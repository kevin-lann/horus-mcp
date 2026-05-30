"""SQLite persistence for watchlist, signals, and alerts."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass
class WatchlistRow:
    id: int
    symbol: str
    added_at: str


@dataclass
class SignalRow:
    id: int
    name: str
    signal_type: str
    params: dict[str, Any]
    ticker_overrides: list[str] | None
    ticker_scope: str
    exchange: str | None
    enabled: bool
    created_at: str
    history_period: str = "1y"
    interval: str = "1d"


@dataclass
class AlertRow:
    id: int
    signal_id: int
    symbol: str
    triggered_at: str
    details: dict[str, Any]


@dataclass
class ScanJobRow:
    id: int
    job_type: str
    status: str
    params: dict[str, Any]
    requested_at: str
    started_at: str | None
    finished_at: str | None
    checked_count: int
    total_count: int | None
    fired_count: int
    result_count: int
    cancel_requested: bool
    result: dict[str, Any] | None
    error: str | None


def _default_db_path() -> Path:
    """Return the default SQLite DB path and ensure its parent exists."""
    p = Path.home() / ".scanner_mcp" / "data.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Store:
    """Thread-safe SQLite store for watchlists, signals, and alert history."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _default_db_path()
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Yield a locked SQLite connection and commit after successful use."""
        with self._lock:
            c = sqlite3.connect(self._path)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    def _init_schema(self) -> None:
        """Create tables and indexes if this is a fresh database."""
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT UNIQUE NOT NULL,
                    added_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    ticker_overrides TEXT,
                    ticker_scope TEXT NOT NULL DEFAULT 'watchlist',
                    exchange TEXT,
                    history_period TEXT NOT NULL DEFAULT '1y',
                    interval TEXT NOT NULL DEFAULT '1d',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    signal_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals (id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (triggered_at DESC);
                CREATE TABLE IF NOT EXISTS scan_jobs (
                    id INTEGER PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    checked_count INTEGER NOT NULL DEFAULT 0,
                    total_count INTEGER,
                    fired_count INTEGER NOT NULL DEFAULT 0,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_scan_jobs_requested_time ON scan_jobs (requested_at DESC);
                """
            )
            self._migrate_signals_columns(c)

    def _migrate_signals_columns(self, conn: sqlite3.Connection) -> None:
        """Add newer signal columns and backfill sane defaults for legacy rows."""
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(signals)")}
        if "ticker_scope" not in cols:
            conn.execute(
                "ALTER TABLE signals ADD COLUMN ticker_scope TEXT NOT NULL DEFAULT 'watchlist'",
            )
            conn.execute("ALTER TABLE signals ADD COLUMN exchange TEXT")
            for row in conn.execute("SELECT id, ticker_overrides FROM signals").fetchall():
                rid = int(row[0])
                ov = row[1]
                if ov:
                    conn.execute(
                        "UPDATE signals SET ticker_scope = 'tickers' WHERE id = ?",
                        (rid,),
                    )
                else:
                    conn.execute(
                        "UPDATE signals SET ticker_scope = 'watchlist' WHERE id = ?",
                        (rid,),
                    )
        if "history_period" not in cols:
            conn.execute(
                "ALTER TABLE signals ADD COLUMN history_period TEXT NOT NULL DEFAULT '1y'",
            )
        if "interval" not in cols:
            conn.execute(
                "ALTER TABLE signals ADD COLUMN interval TEXT NOT NULL DEFAULT '1d'",
            )

    # watchlist
    def watchlist_add(self, symbols: Sequence[str]) -> list[str]:
        """Insert uppercase watchlist symbols and return only newly added ones."""
        now = _utc_now()
        added: list[str] = []
        with self._conn() as c:
            for sym in symbols:
                s = sym.strip().upper()
                if not s:
                    continue
                c.execute(
                    "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
                    (s, now),
                )
                n = c.execute("SELECT changes()").fetchone()[0]
                if n:
                    added.append(s)
        return added

    def watchlist_remove(self, symbols: Sequence[str]) -> int:
        """Remove watchlist symbols and return the number of deleted rows."""
        n = 0
        with self._conn() as c:
            for sym in symbols:
                cur = c.execute("DELETE FROM watchlist WHERE symbol = ?", (sym.strip().upper(),))
                n += cur.rowcount or 0
        return n

    def watchlist_get(self) -> list[WatchlistRow]:
        """Return the current watchlist sorted by symbol."""
        with self._conn() as c:
            rows = c.execute("SELECT id, symbol, added_at FROM watchlist ORDER BY symbol")
            return [WatchlistRow(int(r[0]), r[1], r[2]) for r in rows]

    # signals
    def signal_create(
        self,
        name: str,
        signal_type: str,
        params: dict[str, Any],
        ticker_overrides: list[str] | None,
        ticker_scope: str = "watchlist",
        exchange: str | None = None,
        history_period: str = "1y",
        interval: str = "1d",
    ) -> int:
        """Persist an enabled signal and return its database ID."""
        now = _utc_now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO signals (
                    name, signal_type, params, ticker_overrides,
                    ticker_scope, exchange, history_period, interval, enabled, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    name,
                    signal_type,
                    json.dumps(params),
                    json.dumps(ticker_overrides) if ticker_overrides is not None else None,
                    ticker_scope,
                    exchange.strip().upper() if exchange else None,
                    history_period.strip().lower(),
                    interval.strip().lower(),
                    now,
                ),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def signal_list(self) -> list[SignalRow]:
        """Return all configured signals ordered by ID."""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM signals ORDER BY id")
            return [_row_to_signal(r) for r in rows]

    def signal_get(self, signal_id: int) -> SignalRow | None:
        """Return one signal by ID, or None if it does not exist."""
        with self._conn() as c:
            r = c.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
            return _row_to_signal(r) if r else None

    def signal_delete(self, signal_id: int) -> bool:
        """Delete a signal and its alert history; return whether it existed."""
        with self._conn() as c:
            c.execute("DELETE FROM alerts WHERE signal_id = ?", (signal_id,))
            cur = c.execute("DELETE FROM signals WHERE id = ?", (signal_id,))
            return (cur.rowcount or 0) > 0

    def signal_set_enabled(self, signal_id: int, enabled: bool) -> bool:
        """Enable or disable a signal by ID; return whether it existed."""
        with self._conn() as c:
            cur = c.execute("UPDATE signals SET enabled = ? WHERE id = ?", (1 if enabled else 0, signal_id))
            return (cur.rowcount or 0) > 0

    # alerts
    def alert_insert(
        self,
        signal_id: int,
        symbol: str,
        details: dict[str, Any],
    ) -> int:
        """Persist one triggered alert and return its database ID."""
        now = _utc_now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO alerts (signal_id, symbol, triggered_at, details) VALUES (?, ?, ?, ?)",
                (signal_id, symbol.upper(), now, json.dumps(details)),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def alerts_recent(self, limit: int = 50) -> list[AlertRow]:
        """Return the newest alert rows, newest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT ?",
                (limit,),
            )
            return [_row_to_alert(r) for r in rows]

    # scan jobs
    def scan_job_create(self, job_type: str, params: dict[str, Any]) -> int:
        """Create a queued scan job and return its ID."""
        now = _utc_now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO scan_jobs (job_type, status, params, requested_at)
                VALUES (?, 'queued', ?, ?)
                """,
                (job_type, json.dumps(params), now),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def scan_job_get(self, job_id: int) -> ScanJobRow | None:
        """Return one scan job by ID, or None if missing."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM scan_jobs WHERE id = ?", (job_id,)).fetchone()
            return _row_to_scan_job(row) if row else None

    def scan_jobs_recent(self, limit: int = 20) -> list[ScanJobRow]:
        """Return recent scan jobs, newest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scan_jobs ORDER BY requested_at DESC LIMIT ?",
                (limit,),
            )
            return [_row_to_scan_job(r) for r in rows]

    def scan_job_mark_running(self, job_id: int, *, total_count: int | None = None) -> bool:
        """Transition a queued job to running unless it was already cancelled."""
        now = _utc_now()
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'running', started_at = ?, total_count = COALESCE(?, total_count)
                WHERE id = ? AND status = 'queued' AND cancel_requested = 0
                """,
                (now, total_count, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_update_progress(
        self,
        job_id: int,
        *,
        checked_count: int,
        fired_count: int,
        result_count: int,
        total_count: int | None = None,
    ) -> bool:
        """Persist running job progress counters."""
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET checked_count = ?, fired_count = ?, result_count = ?, total_count = COALESCE(?, total_count)
                WHERE id = ? AND status = 'running'
                """,
                (checked_count, fired_count, result_count, total_count, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_complete(self, job_id: int, result: dict[str, Any]) -> bool:
        """Mark a job completed and persist its final result payload."""
        now = _utc_now()
        checked_count = int(result.get("checked_count", 0))
        total_count = result.get("total_count")
        fired_count = int(result.get("triggered_count", result.get("fired_count", result.get("count", 0))))
        result_count = int(result.get("count", 0))
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'completed',
                    finished_at = ?,
                    checked_count = ?,
                    total_count = COALESCE(?, total_count),
                    fired_count = ?,
                    result_count = ?,
                    result_json = ?,
                    error = NULL
                WHERE id = ?
                """,
                (now, checked_count, total_count, fired_count, result_count, json.dumps(result), job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_fail(self, job_id: int, error: str) -> bool:
        """Mark a job failed and store the error string."""
        now = _utc_now()
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'failed', finished_at = ?, error = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (now, error, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_request_cancel(self, job_id: int) -> bool:
        """Request cancellation for a queued or running job."""
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET cancel_requested = 1
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (job_id,),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_is_cancel_requested(self, job_id: int) -> bool:
        """Return whether cancellation has been requested for a job."""
        with self._conn() as c:
            row = c.execute("SELECT cancel_requested FROM scan_jobs WHERE id = ?", (job_id,)).fetchone()
            return bool(row[0]) if row else False

    def scan_job_mark_cancelled(
        self,
        job_id: int,
        *,
        checked_count: int,
        fired_count: int,
        result_count: int,
        total_count: int | None = None,
    ) -> bool:
        """Mark a job cancelled with the latest progress counters."""
        now = _utc_now()
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'cancelled',
                    finished_at = ?,
                    checked_count = ?,
                    total_count = COALESCE(?, total_count),
                    fired_count = ?,
                    result_count = ?,
                    error = NULL
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (now, checked_count, total_count, fired_count, result_count, job_id),
            )
            return (cur.rowcount or 0) > 0


def _row_to_signal(r: sqlite3.Row) -> SignalRow:
    """Convert a SQLite signal row into a typed dataclass."""
    ov = r["ticker_overrides"]
    keys = r.keys()
    scope = str(r["ticker_scope"]) if "ticker_scope" in keys else "watchlist"
    ex = r["exchange"] if "exchange" in keys else None
    history_period = str(r["history_period"]).strip().lower() if "history_period" in keys and r["history_period"] else "1y"
    interval = str(r["interval"]).strip().lower() if "interval" in keys and r["interval"] else "1d"
    ov_list = json.loads(ov) if ov else None
    if "ticker_scope" not in keys:
        scope = "tickers" if ov_list else "watchlist"
    return SignalRow(
        id=int(r["id"]),
        name=str(r["name"]),
        signal_type=str(r["signal_type"]),
        params=json.loads(r["params"]),
        ticker_overrides=ov_list,
        ticker_scope=scope,
        exchange=str(ex).strip().upper() if ex else None,
        enabled=bool(r["enabled"]),
        created_at=str(r["created_at"]),
        history_period=history_period,
        interval=interval,
    )


def _row_to_alert(r: sqlite3.Row) -> AlertRow:
    """Convert a SQLite alert row into a typed dataclass."""
    d = r["details"]
    return AlertRow(
        id=int(r["id"]),
        signal_id=int(r["signal_id"]),
        symbol=str(r["symbol"]),
        triggered_at=str(r["triggered_at"]),
        details=json.loads(d) if d else {},
    )


def _row_to_scan_job(r: sqlite3.Row) -> ScanJobRow:
    """Convert a SQLite scan job row into a typed dataclass."""
    payload = r["result_json"]
    return ScanJobRow(
        id=int(r["id"]),
        job_type=str(r["job_type"]),
        status=str(r["status"]),
        params=json.loads(r["params"]),
        requested_at=str(r["requested_at"]),
        started_at=str(r["started_at"]) if r["started_at"] else None,
        finished_at=str(r["finished_at"]) if r["finished_at"] else None,
        checked_count=int(r["checked_count"]),
        total_count=int(r["total_count"]) if r["total_count"] is not None else None,
        fired_count=int(r["fired_count"]),
        result_count=int(r["result_count"]),
        cancel_requested=bool(r["cancel_requested"]),
        result=json.loads(payload) if payload else None,
        error=str(r["error"]) if r["error"] else None,
    )


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()

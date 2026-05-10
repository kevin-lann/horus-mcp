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


@dataclass
class AlertRow:
    id: int
    signal_id: int
    symbol: str
    triggered_at: str
    details: dict[str, Any]


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
                """
            )
            self._migrate_signals_columns()

    def _migrate_signals_columns(self) -> None:
        """Add ticker_scope / exchange and backfill from legacy ticker_overrides."""
        with self._conn() as c:
            cols = {str(r[1]) for r in c.execute("PRAGMA table_info(signals)")}
            if "ticker_scope" not in cols:
                c.execute(
                    "ALTER TABLE signals ADD COLUMN ticker_scope TEXT NOT NULL DEFAULT 'watchlist'",
                )
                c.execute("ALTER TABLE signals ADD COLUMN exchange TEXT")
                for row in c.execute("SELECT id, ticker_overrides FROM signals").fetchall():
                    rid = int(row[0])
                    ov = row[1]
                    if ov:
                        c.execute(
                            "UPDATE signals SET ticker_scope = 'tickers' WHERE id = ?",
                            (rid,),
                        )
                    else:
                        c.execute(
                            "UPDATE signals SET ticker_scope = 'watchlist' WHERE id = ?",
                            (rid,),
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
    ) -> int:
        """Persist an enabled signal and return its database ID."""
        now = _utc_now()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO signals (
                    name, signal_type, params, ticker_overrides,
                    ticker_scope, exchange, enabled, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    name,
                    signal_type,
                    json.dumps(params),
                    json.dumps(ticker_overrides) if ticker_overrides is not None else None,
                    ticker_scope,
                    exchange.strip().upper() if exchange else None,
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


def _row_to_signal(r: sqlite3.Row) -> SignalRow:
    """Convert a SQLite signal row into a typed dataclass."""
    ov = r["ticker_overrides"]
    keys = r.keys()
    scope = str(r["ticker_scope"]) if "ticker_scope" in keys else "watchlist"
    ex = r["exchange"] if "exchange" in keys else None
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


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()

"""Daily EOD scan using APScheduler."""

from __future__ import annotations

import logging
import os
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from scanner_mcp.data.exchange_universe import fetch_exchange_tickers
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.notify.notifier import notify_desktop
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal

log = logging.getLogger(__name__)


def _tickers_for_signal(srow: SignalRow, watch: list[str]) -> list[str]:
    if srow.ticker_scope == "tickers":
        return list(srow.ticker_overrides or [])
    if srow.ticker_scope == "exchange":
        if not srow.exchange:
            return []
        try:
            return fetch_exchange_tickers(srow.exchange)
        except ValueError:
            return []
    return list(watch)


def _scan_job(store: Store, provider: DataProvider) -> None:
    """APScheduler entrypoint that logs and swallows scan failures."""
    try:
        run_full_scan(store, provider, notify=True)
    except Exception:  # noqa: BLE001
        log.exception("scan job failed")


def run_full_scan(
    store: Store,
    provider: DataProvider,
    *,
    notify: bool = True,
) -> dict[str, Any]:
    """Evaluate all enabled signals and persist alerts for triggered results.

    Each signal uses `ticker_scope`: fixed tickers, global watchlist, or all
    symbols on a configured US/crypto exchange (Yahoo screener). The result
    counts every fetched symbol/signal pair checked and includes only fired
    alerts in the `alerts` list.
    """
    result: dict[str, Any] = {"checked": 0, "fired": 0, "alerts": []}
    try:
        user_ids = store.all_user_ids()
    except Exception:  # noqa: BLE001
        log.exception("user list")
        return result

    for user_id in user_ids:
        try:
            sig_rows = [s for s in store.signal_list(user_id) if s.enabled]
        except Exception:  # noqa: BLE001
            log.exception("signal list for user %s", user_id)
            continue
        watch = [w.symbol for w in store.watchlist_get(user_id)]

        for srow in sig_rows:
            tickers = _tickers_for_signal(srow, watch)
            if not tickers:
                log.debug("Signal %s has no symbols to scan", srow.id)
                continue
            asig = ActiveSignal(
                id=srow.id,
                name=srow.name,
                signal_type=srow.signal_type,
                params=srow.params,
                ticker_overrides=srow.ticker_overrides,
                history_period=srow.history_period,
                interval=srow.interval,
            )
            for sym in tickers:
                result["checked"] += 1
                try:
                    df = provider.get_history(sym, period=asig.history_period, interval=asig.interval)
                except Exception as e:  # noqa: BLE001
                    log.debug("history %s: %s", sym, e)
                    continue
                if df is None or df.empty:
                    continue
                try:
                    trig, det = evaluate(asig, df)
                except Exception as e:  # noqa: BLE001
                    log.debug("eval %s: %s", sym, e)
                    continue
                if trig:
                    result["fired"] += 1
                    result["alerts"].append(
                        {
                            "user_id": user_id,
                            "signal_id": srow.id,
                            "name": srow.name,
                            "symbol": sym,
                            "details": det,
                        }
                    )
                    try:
                        store.alert_insert(user_id, srow.id, sym, det)
                    except Exception as e:  # noqa: BLE001
                        log.error("alert_insert: %s", e)
                    if notify:
                        notify_desktop(
                            "Signal Scanner",
                            f"{srow.name} — {sym} triggered",
                        )
    return result


def start_scheduler(
    store: Store,
    provider: DataProvider,
) -> BackgroundScheduler:
    """Start the daily end-of-day scan scheduler.

    `SCAN_TIME` may be set as `HH:MM` in Eastern time; invalid values fall back
    to 16:30 ET.
    """
    sched = BackgroundScheduler(
        timezone=ZoneInfo("America/New_York"),
        daemon=True,
    )
    t = os.environ.get("SCAN_TIME", "16:30")
    try:
        parts = t.split(":")
        h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        h, m = 16, 30
    sched.add_job(
        lambda: _scan_job(store, provider),
        CronTrigger(hour=h, minute=m, timezone=ZoneInfo("America/New_York")),
        id="daily_scan",
        replace_existing=True,
    )
    sched.start()
    log.info("Scheduler started, daily at %s ET", f"{h:02d}:{m:02d}")
    return sched

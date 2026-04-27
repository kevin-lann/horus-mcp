"""Daily EOD scan using APScheduler."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from scanner_mcp.data.provider import YFinanceProvider
from scanner_mcp.db.store import Store
from scanner_mcp.notify.notifier import notify_desktop
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal

log = logging.getLogger(__name__)


def _scan_job(store: Store, provider: YFinanceProvider) -> None:
    try:
        run_full_scan(store, provider, notify=True)
    except Exception:  # noqa: BLE001
        log.exception("scan job failed")


def run_full_scan(
    store: Store,
    provider: YFinanceProvider,
    *,
    notify: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {"checked": 0, "fired": 0, "alerts": []}
    try:
        sig_rows = [s for s in store.signal_list() if s.enabled]
    except Exception:  # noqa: BLE001
        log.exception("signal list")
        return result
    watch = [w.symbol for w in store.watchlist_get()]

    for srow in sig_rows:
        tickers = srow.ticker_overrides or watch
        if not tickers:
            log.debug("Signal %s has no tickers (empty watchlist and no overrides)", srow.id)
            continue
        asig = ActiveSignal(
            id=srow.id,
            name=srow.name,
            signal_type=srow.signal_type,
            params=srow.params,
            ticker_overrides=srow.ticker_overrides,
        )
        for sym in tickers:
            result["checked"] += 1
            try:
                df = provider.get_history(sym, period="1y", interval="1d")
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
                        "signal_id": srow.id,
                        "name": srow.name,
                        "symbol": sym,
                        "details": det,
                    }
                )
                try:
                    store.alert_insert(srow.id, sym, det)
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
    provider: YFinanceProvider,
) -> BackgroundScheduler:
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

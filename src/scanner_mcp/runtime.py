"""Application runtime helpers for shared provider/store state and startup."""

from __future__ import annotations

import logging
import os
import signal
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from apscheduler.schedulers.base import BaseScheduler
from fastmcp import FastMCP

from scanner_mcp.data.provider import CompositeDataProvider, DataProvider
from scanner_mcp.db.store import Store
from scanner_mcp.scanner import scheduler as scan_sched
from scanner_mcp.signals.service import ScanCancelledError, execute_scan

log = logging.getLogger(__name__)

_store: Store | None = None
_provider: DataProvider | None = None
_sched: BaseScheduler | None = None
_scan_executor: ThreadPoolExecutor | None = None
_scan_futures: dict[int, Future[None]] = {}


def shutdown_scheduler() -> None:
    """Stop the background scheduler if it was started."""
    global _sched
    if _sched is None:
        return
    try:
        if _sched.running:
            _sched.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("Scheduler shutdown skipped: %s", exc)
    finally:
        _sched = None


def shutdown_scan_executor() -> None:
    """Stop the background scan executor."""
    global _scan_executor
    if _scan_executor is None:
        return
    try:
        _scan_executor.shutdown(wait=False, cancel_futures=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("Scan executor shutdown skipped: %s", exc)
    finally:
        _scan_executor = None
        _scan_futures.clear()


def get_store() -> Store:
    """Lazily create the SQLite store, honoring SCANNER_MCP_DB."""
    global _store
    if _store is None:
        path = os.environ.get("SCANNER_MCP_DB")
        _store = Store(path)
    return _store


def get_provider() -> DataProvider:
    """Lazily create the shared market data provider."""
    global _provider
    if _provider is None:
        _provider = CompositeDataProvider.default()
    return _provider

def scan_job_payload(job_id: int) -> dict[str, Any]:
    row = get_store().scan_job_get(job_id)
    if row is None:
        return {"error": "scan job not found", "job_id": job_id}
    return {
        "job_id": row.id,
        "job_type": row.job_type,
        "status": row.status,
        "requested_at": row.requested_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "checked_count": row.checked_count,
        "total_count": row.total_count,
        "fired_count": row.fired_count,
        "result_count": row.result_count,
        "cancel_requested": row.cancel_requested,
        "params": row.params,
        "error": row.error,
    }


def _scan_executor_instance() -> ThreadPoolExecutor:
    global _scan_executor
    if _scan_executor is None:
        workers = max(1, int(os.environ.get("SCANNER_MCP_SCAN_WORKERS", "2")))
        _scan_executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="scanner-mcp-scan",
        )
    return _scan_executor


def start_scan_job(
    *,
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> int:
    """Create and dispatch a background scan job, returning its persistent job ID."""
    store = get_store()
    params = {
        "signal_id": signal_id,
        "tickers": tickers,
        "all_signal_types": all_signal_types,
        "symbol": symbol,
        "exchange": exchange,
    }
    job_id = store.scan_job_create("run_scan", params)

    def _run() -> None:
        if not store.scan_job_mark_running(job_id):
            store.scan_job_mark_cancelled(
                job_id,
                checked_count=0,
                fired_count=0,
                result_count=0,
                total_count=0,
            )
            return
        try:
            result = execute_scan(
                store,
                get_provider(),
                signal_id=signal_id,
                tickers=tickers,
                all_signal_types=all_signal_types,
                symbol=symbol,
                exchange=exchange,
                progress_callback=lambda checked_count, fired_count, result_count, total_count: store.scan_job_update_progress(
                    job_id,
                    checked_count=checked_count,
                    fired_count=fired_count,
                    result_count=result_count,
                    total_count=total_count,
                ),
                cancel_check=lambda: store.scan_job_is_cancel_requested(job_id),
            )
            store.scan_job_complete(job_id, result)
        except ScanCancelledError:
            latest = store.scan_job_get(job_id)
            checked_count = latest.checked_count if latest else 0
            fired_count = latest.fired_count if latest else 0
            result_count = latest.result_count if latest else 0
            total_count = latest.total_count if latest else None
            store.scan_job_mark_cancelled(
                job_id,
                checked_count=checked_count,
                fired_count=fired_count,
                result_count=result_count,
                total_count=total_count,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("background scan job %s failed", job_id)
            store.scan_job_fail(job_id, str(exc))
        finally:
            _scan_futures.pop(job_id, None)

    fut = _scan_executor_instance().submit(_run)
    _scan_futures[job_id] = fut
    return job_id


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """FastMCP lifespan hook that starts and stops the scan scheduler."""
    global _sched
    st = get_store()
    provider = get_provider()
    try:
        _sched = scan_sched.start_scheduler(st, provider)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not start scheduler: %s", exc)
    try:
        yield {"store": st, "provider": provider, "scheduler": _sched}
    finally:
        shutdown_scheduler()
        shutdown_scan_executor()


def configure_logging() -> None:
    """Configure process logging from environment variables."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("SCANNER_MCP_LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def install_signal_handlers() -> tuple[Any, Any]:
    """Install SIGINT/SIGTERM handlers and return the previous handlers."""

    def _handle_stop(signum: int, _: Any) -> None:
        log.info("Received signal %s, shutting down", signum)
        shutdown_scheduler()
        shutdown_scan_executor()
        logging.shutdown()
        raise SystemExit(128 + signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    return old_sigint, old_sigterm


def restore_signal_handlers(old_sigint: Any, old_sigterm: Any) -> None:
    """Restore the previous process signal handlers."""
    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)

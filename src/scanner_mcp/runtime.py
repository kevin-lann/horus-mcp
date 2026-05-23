"""Application runtime helpers for shared provider/store state and startup."""

from __future__ import annotations

import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from apscheduler.schedulers.base import BaseScheduler
from fastmcp import FastMCP

from scanner_mcp.data.provider import CompositeDataProvider, DataProvider
from scanner_mcp.db.store import Store
from scanner_mcp.scanner import scheduler as scan_sched

log = logging.getLogger(__name__)

_store: Store | None = None
_provider: DataProvider | None = None
_sched: BaseScheduler | None = None


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
        logging.shutdown()
        os._exit(128 + signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    return old_sigint, old_sigterm


def restore_signal_handlers(old_sigint: Any, old_sigterm: Any) -> None:
    """Restore the previous process signal handlers."""
    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)

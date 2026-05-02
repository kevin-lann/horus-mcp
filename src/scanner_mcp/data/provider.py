"""Market data provider abstraction; yfinance implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import yfinance as yf

from scanner_mcp.data.cache import TTLCache

log = logging.getLogger(__name__)

_IFTTL = 15 * 60.0 # 15 minutes
_HISTORY_DAILY_TTL = 6 * 60 * 60.0 # 6 hours


class DataProvider(ABC):
    """Interface used by tools, charts, and scans to fetch market data."""

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV history for a symbol, or an empty DataFrame on failure."""
        ...

    @abstractmethod
    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        """Return quote-like metadata for a symbol, or an empty dict on failure."""
        ...

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: str | None) -> dict[str, Any]:
        """Return option chain frames and expiries, or an error dictionary."""
        ...


def _empty_df() -> pd.DataFrame:
    """Return a fresh empty history frame for provider failure paths."""
    return pd.DataFrame()


class YFinanceProvider(DataProvider):
    """YFinance-backed provider with short-lived in-memory caching."""

    def __init__(self) -> None:
        self._cache: TTLCache[Any] = TTLCache(300.0)
        self._hist_cache: TTLCache[pd.DataFrame] = TTLCache(_HISTORY_DAILY_TTL)

    def get_history(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data and cache by symbol, period, and interval.

        Intraday intervals get a shorter TTL than daily/weekly style history. The
        returned DataFrame is copied so callers cannot mutate cached data.
        """
        sym = symbol.strip().upper()
        key = ("hist", sym, period, interval)
        hit = self._hist_cache.get(key)
        if hit is not None:
            log.debug("history cache hit %s %s %s rows=%s", sym, period, interval, len(hit))
            return hit.copy()
        t = yf.Ticker(sym)
        try:
            df = t.history(period=period, interval=interval, auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            log.exception("history failed %s %s %s: %s", sym, period, interval, e)
            return _empty_df()
        if df is None or df.empty:
            log.debug("history empty %s %s %s", sym, period, interval)
            return _empty_df()
        ttl = _IFTTL if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "1h") else _HISTORY_DAILY_TTL
        self._hist_cache.set(key, df, ttl=ttl)
        log.debug("history fetched %s %s %s rows=%s cols=%s", sym, period, interval, len(df), list(df.columns))
        return df.copy()

    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        """Fetch yfinance fast_info with a one-minute cache."""
        sym = symbol.strip().upper()
        key = ("fi", sym)
        hit = self._cache.get(key)
        if hit is not None:
            log.debug("fast_info cache hit %s keys=%s", sym, sorted(hit.keys()))
            return hit
        t = yf.Ticker(sym)
        try:
            raw = t.fast_info
            fi = dict(raw) if raw else {}
        except Exception as e:  # noqa: BLE001
            log.exception("fast_info failed %s: %s", sym, e)
            fi = {}
        self._cache.set(key, fi, ttl=60.0)
        log.debug("fast_info fetched %s keys=%s", sym, sorted(fi.keys()))
        return fi

    def get_option_chain(self, symbol: str, expiry: str | None) -> dict[str, Any]:
        """Fetch calls, puts, and available expiries for a symbol."""
        sym = symbol.strip().upper()
        t = yf.Ticker(sym)
        try:
            if expiry:
                chain = t.option_chain(expiry)
            else:
                exps = t.options
                if exps is None or len(exps) == 0:
                    return {"error": "No options data", "calls": None, "puts": None, "expiries": []}
                chain = t.option_chain(exps[0])
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "calls": None, "puts": None, "expiries": list(getattr(t, "options", []) or [])}
        return {
            "calls": chain.calls,
            "puts": chain.puts,
            "expiries": list(getattr(t, "options", []) or []),
        }

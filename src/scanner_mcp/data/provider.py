"""Market data provider abstraction; yfinance and fundamentals implementations."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

from scanner_mcp.data.cache import TTLCache

log = logging.getLogger(__name__)

_IFTTL = 15 * 60.0 # 15 minutes
_HISTORY_DAILY_TTL = 6 * 60 * 60.0 # 6 hours
_FUNDAMENTALS_TTL = 12 * 60 * 60.0 # 12 hours
_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
_ALPHA_VANTAGE_API_KEY_ENV = "ALPHA_VANTAGE_API_KEY"
_ENV_FILE_PATH = Path(__file__).resolve().parents[3] / ".env"


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

    @abstractmethod
    def get_fundamental_series(self, symbol: str, metric: str, frequency: str) -> pd.Series:
        """Return revenue or earnings history indexed by statement date."""
        ...

    @abstractmethod
    def get_historical_pe_series(self, symbol: str, close: pd.Series) -> pd.Series:
        """Return P/E history aligned to a close-price series."""
        ...


def _empty_df() -> pd.DataFrame:
    """Return a fresh empty history frame for provider failure paths."""
    return pd.DataFrame()


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


def _source_series(values: pd.Series, source: str) -> pd.Series:
    values.attrs["source"] = source
    return values


def _read_env_file_key() -> str | None:
    try:
        lines = _ENV_FILE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != _ALPHA_VANTAGE_API_KEY_ENV:
            continue
        value = value.strip().strip("'\"")
        if value:
            return value
    return None


def _statement_frequency(frequency: str) -> str:
    freq = str(frequency).strip().lower()
    if freq in {"annual", "yearly", "year"}:
        return "annual"
    if freq in {"quarterly", "quarter", "q"}:
        return "quarterly"
    raise ValueError("frequency must be quarterly or annual")


def _to_naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx.normalize()).astype("datetime64[ns]")


def _merge_asof_price_over_eps(
    close: pd.Series,
    anchor_dates: pd.DatetimeIndex,
    eps_values: np.ndarray,
) -> pd.Series:
    df_sorted = close.sort_index().astype(float)
    hist = pd.DataFrame(
        {
            "asof": _to_naive_dates(pd.DatetimeIndex(df_sorted.index)),
            "close": df_sorted.values,
        }
    )
    right = pd.DataFrame(
        {
            "anchor": _to_naive_dates(pd.DatetimeIndex(anchor_dates)),
            "eps": eps_values.astype(float),
        }
    ).sort_values("anchor")
    merged = pd.merge_asof(hist, right, left_on="asof", right_on="anchor", direction="backward")
    denom = merged["eps"].to_numpy(dtype=float)
    num = merged["close"].to_numpy(dtype=float)
    pe_vals = np.where(np.isfinite(denom) & (denom > 0), num / denom, np.nan)
    return pd.Series(pe_vals, index=df_sorted.index, dtype=float).reindex(close.index)


def _series_from_alpha_vantage_rows(rows: list[dict[str, Any]], value_keys: tuple[str, ...]) -> pd.Series:
    parsed: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        raw_date = row.get("fiscalDateEnding") or row.get("reportedDate")
        if raw_date is None:
            continue
        date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(date):
            continue
        for key in value_keys:
            if key not in row:
                continue
            value = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(value):
                parsed.append((pd.Timestamp(date), float(value)))
                break
    if not parsed:
        return _empty_series()
    out = pd.Series({date: value for date, value in parsed}, dtype=float).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def _statement_metric_series(stmt: pd.DataFrame, metric: str) -> pd.Series:
    if stmt is None or stmt.empty:
        return _empty_series()
    metric_key = str(metric).strip().lower()
    row_candidates = {
        "revenue": ("Total Revenue", "Operating Revenue"),
        "earnings": (
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing Operation Net Minority Interest",
        ),
    }
    if metric_key not in row_candidates:
        raise ValueError("metric must be revenue or earnings")
    for row_name in row_candidates[metric_key]:
        if row_name in stmt.index:
            out = stmt.loc[row_name].dropna().sort_index().astype(float)
            out.index = pd.DatetimeIndex(out.index)
            return out
    return _empty_series()


class YFinanceProvider(DataProvider):
    """YFinance-backed provider with Alpha Vantage fundamentals and caching."""

    def __init__(self) -> None:
        self._cache: TTLCache[Any] = TTLCache(300.0)
        self._hist_cache: TTLCache[pd.DataFrame] = TTLCache(_HISTORY_DAILY_TTL)
        self._fundamentals_cache: TTLCache[pd.Series] = TTLCache(_FUNDAMENTALS_TTL)

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

    def get_fundamental_series(self, symbol: str, metric: str, frequency: str) -> pd.Series:
        """Fetch revenue or earnings history, preferring Alpha Vantage and falling back to Yahoo."""
        metric_key = str(metric).strip().lower()
        if metric_key not in {"revenue", "earnings"}:
            raise ValueError("metric must be revenue or earnings")
        alpha_vantage = self._alpha_vantage_income_statement_series(symbol, metric_key, frequency)
        if not alpha_vantage.empty:
            return alpha_vantage
        return self._yahoo_fundamental_series(symbol, metric_key, frequency)

    def get_historical_pe_series(self, symbol: str, close: pd.Series) -> pd.Series:
        """Fetch historical P/E from Alpha Vantage EPS, falling back to Yahoo EPS-derived P/E."""
        pe = self._alpha_vantage_historical_pe_series(symbol, close, "quarterly")
        if not pe.empty and np.isfinite(pe.to_numpy(dtype=float)).any():
            return pe
        pe = self._alpha_vantage_historical_pe_series(symbol, close, "annual")
        if not pe.empty and np.isfinite(pe.to_numpy(dtype=float)).any():
            return pe
        return self._yahoo_historical_pe_series(symbol, close)

    def _alpha_vantage_api_key(self) -> str | None:
        key = os.environ.get(_ALPHA_VANTAGE_API_KEY_ENV)
        key = key.strip() if key else ""
        return key or _read_env_file_key()

    def _alpha_vantage_get(self, function: str, symbol: str) -> dict[str, Any]:
        api_key = self._alpha_vantage_api_key()
        if api_key is None:
            return {}
        sym = symbol.strip().upper()
        request_params: dict[str, str] = {"function": function, "symbol": sym}
        cache_key = ("alpha-vantage", function, sym)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        request_params["apikey"] = api_key
        try:
            response = httpx.get(_ALPHA_VANTAGE_URL, params=request_params, timeout=15.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as e:  # noqa: BLE001
            log.exception("Alpha Vantage request failed %s %s: %s", function, sym, e)
            return {}
        if not isinstance(payload, dict):
            log.debug("Alpha Vantage returned non-dict payload for %s %s", function, sym)
            return {}
        if "Note" in payload or "Information" in payload or "Error Message" in payload:
            log.debug("Alpha Vantage returned message for %s %s: %s", function, sym, payload)
            return {}
        self._cache.set(cache_key, payload, ttl=_FUNDAMENTALS_TTL)
        return payload

    def _alpha_vantage_income_statement_series(self, symbol: str, metric: str, frequency: str) -> pd.Series:
        sym = symbol.strip().upper()
        period = _statement_frequency(frequency)
        cache_key = ("alpha-vantage-income-series", sym, metric, period)
        hit = self._fundamentals_cache.get(cache_key)
        if hit is not None:
            return hit.copy()
        payload = self._alpha_vantage_get("INCOME_STATEMENT", sym)
        report_key = "annualReports" if period == "annual" else "quarterlyReports"
        rows_raw = payload.get(report_key, [])
        rows = [row for row in rows_raw if isinstance(row, dict)] if isinstance(rows_raw, list) else []
        value_keys = ("totalRevenue",) if metric == "revenue" else ("netIncome",)
        series = _series_from_alpha_vantage_rows(rows, value_keys)
        if not series.empty:
            series = _source_series(series, "Alpha Vantage")
            self._fundamentals_cache.set(cache_key, series, ttl=_FUNDAMENTALS_TTL)
        return series.copy()

    def _alpha_vantage_historical_pe_series(self, symbol: str, close: pd.Series, frequency: str) -> pd.Series:
        sym = symbol.strip().upper()
        period = _statement_frequency(frequency)
        cache_key = ("alpha-vantage-eps", sym, period)
        hit = self._fundamentals_cache.get(cache_key)
        if hit is not None:
            eps = hit.copy()
        else:
            payload = self._alpha_vantage_get("EARNINGS", sym)
            report_key = "annualEarnings" if period == "annual" else "quarterlyEarnings"
            rows_raw = payload.get(report_key, [])
            rows = [row for row in rows_raw if isinstance(row, dict)] if isinstance(rows_raw, list) else []
            eps = _series_from_alpha_vantage_rows(rows, ("reportedEPS",))
            if period == "quarterly" and len(eps) >= 4:
                eps = eps.rolling(window=4, min_periods=4).sum().dropna()
            if not eps.empty:
                eps = _source_series(eps, "Alpha Vantage EPS")
                self._fundamentals_cache.set(cache_key, eps, ttl=_FUNDAMENTALS_TTL)
        if eps.empty:
            return _empty_series()
        pe = _merge_asof_price_over_eps(close, pd.DatetimeIndex(eps.index), eps.to_numpy(dtype=float))
        return _source_series(pe, "Alpha Vantage EPS")

    def _yahoo_fundamental_series(self, symbol: str, metric: str, frequency: str) -> pd.Series:
        sym = symbol.strip().upper()
        freq = str(frequency).strip().lower()
        ticker = yf.Ticker(sym)
        try:
            if freq in {"annual", "yearly", "year"}:
                stmt = ticker.incomestmt
            elif freq in {"quarterly", "quarter", "q"}:
                stmt = ticker.quarterly_incomestmt
            else:
                raise ValueError("frequency must be quarterly or annual")
        except Exception as e:  # noqa: BLE001
            log.exception("Yahoo income statement failed %s %s: %s", sym, frequency, e)
            return _empty_series()
        out = _statement_metric_series(stmt, metric)
        return _source_series(out, "Yahoo") if not out.empty else out

    def _yahoo_historical_pe_series(self, symbol: str, close: pd.Series) -> pd.Series:
        pe_q = self._yahoo_quarterly_ttm_pe_series(symbol, close)
        pe_a = self._yahoo_annual_fy_eps_pe_series(symbol, close)
        out = pe_q.combine_first(pe_a)
        return _source_series(out, "Yahoo EPS fallback")

    def _yahoo_quarterly_ttm_pe_series(self, symbol: str, close: pd.Series) -> pd.Series:
        sym = symbol.strip().upper()
        try:
            stmt = yf.Ticker(sym).quarterly_incomestmt
        except Exception as e:  # noqa: BLE001
            log.exception("Yahoo quarterly income statement failed %s: %s", sym, e)
            return pd.Series(np.nan, index=close.index, dtype=float)
        if stmt is None or stmt.empty:
            return pd.Series(np.nan, index=close.index, dtype=float)
        if "Diluted EPS" in stmt.index:
            qeps = stmt.loc["Diluted EPS"]
        elif "Basic EPS" in stmt.index:
            qeps = stmt.loc["Basic EPS"]
        else:
            return pd.Series(np.nan, index=close.index, dtype=float)
        qeps = qeps.dropna().sort_index().astype(float)
        if len(qeps) < 4:
            return pd.Series(np.nan, index=close.index, dtype=float)
        ttm = qeps.rolling(window=4, min_periods=4).sum().dropna()
        if ttm.empty:
            return pd.Series(np.nan, index=close.index, dtype=float)
        return _merge_asof_price_over_eps(close, pd.DatetimeIndex(ttm.index), ttm.values)

    def _yahoo_annual_fy_eps_pe_series(self, symbol: str, close: pd.Series) -> pd.Series:
        sym = symbol.strip().upper()
        try:
            stmt = yf.Ticker(sym).incomestmt
        except Exception as e:  # noqa: BLE001
            log.exception("Yahoo annual income statement failed %s: %s", sym, e)
            return pd.Series(np.nan, index=close.index, dtype=float)
        if stmt is None or stmt.empty:
            return pd.Series(np.nan, index=close.index, dtype=float)
        if "Diluted EPS" in stmt.index:
            fy = stmt.loc["Diluted EPS"]
        elif "Basic EPS" in stmt.index:
            fy = stmt.loc["Basic EPS"]
        else:
            return pd.Series(np.nan, index=close.index, dtype=float)
        fy = fy.dropna().sort_index().astype(float)
        if fy.empty:
            return pd.Series(np.nan, index=close.index, dtype=float)
        return _merge_asof_price_over_eps(close, pd.DatetimeIndex(fy.index), fy.values)

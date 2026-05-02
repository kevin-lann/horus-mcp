"""Technical indicator calculations on OHLCV DataFrames."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scanner_mcp.indicators import ta

log = logging.getLogger(__name__)


class Indicators:
    """Latest-value indicator facade over an OHLCV DataFrame.

    Methods return plain floats/dicts for MCP JSON serialization and use `None`
    when there is not enough data to compute a meaningful value.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        if df is not None and not df.empty and "Close" in df.columns:
            self._close = df["Close"].astype(float)
        else:
            self._close = pd.Series(dtype=float)

    @property
    def empty(self) -> bool:
        """Whether the input lacks enough close data for indicator calculations."""
        return self._df is None or self._df.empty or len(self._close) < 2

    def rsi(self, period: int = 14) -> float | None:
        """Return the latest RSI value for `period`, or None if unavailable."""
        if self.empty or len(self._close) < period + 1:
            return None
        s = ta.rsi(self._close, length=period)
        if s is None or s.empty:
            return None
        v = s.iloc[-1]
        if pd.isna(v):
            return None
        return float(v)

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, float | None] | None:
        """Return latest MACD line, signal line, histogram, and prior histogram."""
        if self.empty or len(self._close) < slow + 5:
            return None
        out = ta.macd(self._close, fast=fast, slow=slow, signal=signal)
        if out is None or out.empty:
            return None
        cols = list(out.columns)
        macd_col = next((c for c in cols if c.startswith("MACD_") and not c.startswith("MACDs") and not c.startswith("MACDh")), None)
        sig_col = next((c for c in cols if c.startswith("MACDs_")), None)
        hist_col = next((c for c in cols if c.startswith("MACDh_")), None)

        def _f(col: str | None) -> float | None:
            if not col or col not in out.columns:
                return None
            v = out[col].iloc[-1]
            if pd.isna(v):
                return None
            return float(v)

        prev_hist: float | None
        if hist_col and hist_col in out.columns:
            s = out[hist_col].shift(1)
            v = s.iloc[-1] if not s.empty else None
            prev_hist = None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)
        else:
            prev_hist = None
        return {
            "macd": _f(macd_col),
            "signal": _f(sig_col),
            "hist": _f(hist_col),
            "hist_prev": prev_hist,
        }

    def bbands(self, period: int = 20, std: float = 2.0) -> dict[str, float | None] | None:
        """Return latest Bollinger Bands and percent-B position."""
        if self.empty or len(self._close) < period:
            return None
        out = ta.bbands(self._close, length=period, std=std)
        if out is None or out.empty:
            return None
        row = out.iloc[-1]
        bbl = next((c for c in out.columns if str(c).startswith("BBL_")), None)
        bbm = next((c for c in out.columns if str(c).startswith("BBM_")), None)
        bbu = next((c for c in out.columns if str(c).startswith("BBU_")), None)
        if not bbl or not bbm or not bbu:
            return None
        lo = row[bbl]
        midv = row[bbm]
        up = row[bbu]
        if any(pd.isna(x) for x in (lo, up, midv)):
            return None
        lo, midv, up = float(lo), float(midv), float(up)
        c = float(self._close.iloc[-1])
        pct_b = 0.5 if up == lo else (c - lo) / (up - lo)
        return {"upper": up, "mid": midv, "lower": lo, "pct_b": float(pct_b)}

    def sma(self, period: int) -> float | None:
        """Return the latest simple moving average for `period` bars."""
        if self.empty or len(self._close) < period:
            return None
        s = ta.sma(self._close, length=period)
        if s is None or s.empty:
            return None
        v = s.iloc[-1]
        if pd.isna(v):
            return None
        return float(v)

    def ema(self, period: int) -> float | None:
        """Return the latest exponential moving average for `period` bars."""
        if self.empty or len(self._close) < period:
            return None
        s = ta.ema(self._close, length=period)
        if s is None or s.empty:
            return None
        v = s.iloc[-1]
        if pd.isna(v):
            return None
        return float(v)

    def ath_distance(self) -> float | None:
        """Return percent distance from the all-time high as a negative/zero value."""
        if self.empty:
            return None
        hi = self._df["High"] if "High" in self._df else self._close
        ath = float(hi.max())
        if ath == 0:
            return None
        last = float(self._close.iloc[-1])
        return (last - ath) / ath * 100.0

    def last_close(self) -> float | None:
        """Return the latest close value as a float."""
        if self.empty:
            return None
        v = self._close.iloc[-1]
        if pd.isna(v):
            return None
        return float(v)


def beta_from_returns(
    sym_returns: pd.Series,
    bench_returns: pd.Series,
) -> float | None:
    """Compute beta of a symbol's returns against benchmark returns.

    Returns None when the aligned return series is too short or benchmark
    variance is effectively zero.
    """
    a = sym_returns.dropna()
    b = bench_returns.dropna()
    joined = a.align(b, join="inner")
    if joined[0] is None or len(joined[0]) < 20:
        return None
    x = joined[0].values
    y = joined[1].values
    if len(x) < 20 or np.var(y) < 1e-12:
        return None
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(y))

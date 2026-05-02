"""Evaluate whether a bar pattern / threshold signal is triggered."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scanner_mcp.indicators import ta
from scanner_mcp.indicators.core import Indicators
from scanner_mcp.signals.catalog import merge_params
from scanner_mcp.signals.models import ActiveSignal

log = logging.getLogger(__name__)


def evaluate(signal: ActiveSignal, df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Evaluate one active signal against one symbol's OHLCV history.

    Returns `(triggered, details)`. Details describe the latest computed values
    even when the signal is false, which is useful for diagnostics. Exceptions
    inside an individual signal evaluator are caught and returned as details so
    scans can continue across other symbols and signals.
    """
    st = signal.signal_type
    p = merge_params(st, signal.params)
    ind = Indicators(df)
    if df is None or df.empty or "Close" not in df.columns:
        return False, {"error": "no_data"}

    close = df["Close"].astype(float)

    try:
        if st == "golden_cross":
            return _cross_sma(close, p["fast"], p["slow"], bullish=True)
        if st == "death_cross":
            return _cross_sma(close, p["fast"], p["slow"], bullish=False)
        if st == "macd_bullish_crossover":
            return _cross_macd(close, p, bullish=True)
        if st == "macd_bearish_crossover":
            return _cross_macd(close, p, bullish=False)
        if st == "rsi_oversold":
            return _rsi_threshold(ind, p["period"], p["threshold"], below=True)
        if st == "rsi_overbought":
            return _rsi_threshold(ind, p["period"], p["threshold"], below=False)
        if st == "pct_from_ma":
            return _pct_from_ma(close, p)
        if st == "pct_from_ath":
            return _pct_from_ath(df, p["min_pct_below_ath"])
        if st == "bbands_breakout":
            return _bb_breakout(close, p)
        if st == "bull_flag":
            return _bull_flag(df, p)
    except Exception as e:  # noqa: BLE001
        log.exception("evaluate %s: %s", st, e)
        return False, {"error": str(e)}

    return False, {"error": "unknown_signal"}


def _cross_sma(
    close: pd.Series, fast: int, slow: int, bullish: bool
) -> tuple[bool, dict[str, Any]]:
    """Detect a latest-bar fast/slow SMA crossover."""
    if len(close) < slow + 2:
        return False, {"reason": "insufficient_bars", "need": slow + 2}
    a = ta.sma(close, length=fast)
    b = ta.sma(close, length=slow)
    if a is None or b is None or a.empty or b.empty:
        return False, {"reason": "sma_fail"}
    a0, a1 = a.iloc[-1], a.iloc[-2]
    b0, b1 = b.iloc[-1], b.iloc[-2]
    if any(pd.isna(x) for x in (a0, a1, b0, b1)):
        return False, {"reason": "nan"}
    if bullish:
        trig = a1 <= b1 and a0 > b0
    else:
        trig = a1 >= b1 and a0 < b0
    return trig, {
        f"sma_{fast}": float(a0),
        f"sma_{slow}": float(b0),
    }


def _cross_macd(
    close: pd.Series, p: dict[str, Any], bullish: bool
) -> tuple[bool, dict[str, Any]]:
    """Detect a latest-bar MACD/signal line crossover."""
    out = ta.macd(close, fast=p["fast"], slow=p["slow"], signal=p["signal"])
    if out is None or out.empty or len(out) < 2:
        return False, {"reason": "macd_fail"}
    try:
        macd_col = next(
            c
            for c in out.columns
            if c.startswith("MACD_") and not c.startswith("MACDs") and not c.startswith("MACDh")
        )
        sig_col = next(c for c in out.columns if c.startswith("MACDs_"))
    except StopIteration:
        return False, {"reason": "macd_col"}
    m0, m1 = out[macd_col].iloc[-1], out[macd_col].iloc[-2]
    s0, s1 = out[sig_col].iloc[-1], out[sig_col].iloc[-2]
    if any(pd.isna(x) for x in (m0, m1, s0, s1)):
        return False, {"reason": "nan"}
    if bullish:
        trig = m1 <= s1 and m0 > s0
    else:
        trig = m1 >= s1 and m0 < s0
    return trig, {"macd": float(m0), "signal": float(s0)}


def _rsi_threshold(
    ind: Indicators, period: int, thr: float, below: bool
) -> tuple[bool, dict[str, Any]]:
    """Check whether the latest RSI is beyond a configured threshold."""
    v = ind.rsi(period=period)
    if v is None:
        return False, {"reason": "rsi_nan"}
    if below:
        trig = v < thr
    else:
        trig = v > thr
    return trig, {"rsi": v, "threshold": thr}


def _pct_from_ma(close: pd.Series, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check whether price is within a percent band around an SMA or EMA."""
    n = int(p["ma_period"])
    if len(close) < n + 1:
        return False, {"reason": "insufficient_bars"}
    if p.get("ma_type", "sma") == "ema":
        line = ta.ema(close, length=n)
    else:
        line = ta.sma(close, length=n)
    if line is None or line.empty:
        return False, {"reason": "ma_fail"}
    pr = float(close.iloc[-1])
    ma = float(line.iloc[-1])
    if ma == 0 or pd.isna(ma):
        return False, {"reason": "ma_nan"}
    diff = abs(pr - ma) / abs(ma) * 100.0
    trig = diff <= float(p["pct"])
    return trig, {"price": pr, f'{p.get("ma_type", "sma")}_{n}': ma, "diff_pct": diff}


def _pct_from_ath(df: pd.DataFrame, min_pct: float) -> tuple[bool, dict[str, Any]]:
    """Check whether latest close is at least N percent below the series high."""
    hi = df["High"] if "High" in df else df["Close"]
    ath = float(hi.max())
    pr = float(df["Close"].iloc[-1])
    if ath == 0:
        return False, {"reason": "ath_zero"}
    dist = (pr - ath) / ath * 100.0
    trig = dist <= -min_pct
    return trig, {"ath": ath, "close": pr, "pct_from_ath": dist}


def _bb_breakout(close: pd.Series, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Check whether the latest close is outside the configured Bollinger Band."""
    out = ta.bbands(close, length=int(p["length"]), std=float(p["std"]))
    if out is None or out.empty:
        return False, {"reason": "bb_fail"}
    row = out.iloc[-1]
    try:
        bbl = next(c for c in out.columns if str(c).startswith("BBL_"))
        bbu = next(c for c in out.columns if str(c).startswith("BBU_"))
    except StopIteration:
        return False, {"reason": "bb_cols"}
    c = float(close.iloc[-1])
    lo, up = float(row[bbl]), float(row[bbu])
    side = p.get("side", "either")
    below = c < lo
    above = c > up
    if side == "lower":
        trig = below
    elif side == "upper":
        trig = above
    else:
        trig = below or above
    return trig, {"close": c, "lower": lo, "upper": up, "broke": "lower" if below else ("upper" if above else "none")}


def _bull_flag(df: pd.DataFrame, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Detect a simplified bull-flag pattern using prior move and consolidation."""
    close = df["Close"].astype(float)
    L = int(p["prior_lookback"])
    consol = int(p["consol_days"])
    if len(close) < L + consol + 1:
        return False, {"reason": "insufficient_bars"}
    prior = close.iloc[-(consol + L) : -consol]
    if prior.empty or prior.iloc[0] == 0:
        return False, {"reason": "prior_zero"}
    move = (prior.iloc[-1] - prior.iloc[0]) / abs(float(prior.iloc[0])) * 100.0
    window = close.iloc[-consol:]
    rng = float(window.max() - window.min())
    pr = float(window.iloc[-1])
    range_pct = rng / pr * 100.0 if pr else 999.0
    trig = move >= p["prior_move_pct"] and range_pct <= p["max_range_pct"]
    return trig, {"prior_move_pct": move, "consol_range_pct": range_pct}

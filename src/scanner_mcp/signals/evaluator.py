"""Evaluate whether a bar pattern / threshold signal is triggered."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from scanner_mcp.indicators import ta
from scanner_mcp.indicators.core import Indicators
from scanner_mcp.signals import calculations as calc
from scanner_mcp.signals.catalog import merge_params
from scanner_mcp.signals.confidence import with_confidence, with_confidence_result
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
        return False, with_confidence(st, False, {"error": "no_data"})

    close = df["Close"].astype(float)

    try:
        if st == "golden_cross":
            return with_confidence_result(st, *_cross_sma(close, p["fast"], p["slow"], bullish=True))
        if st == "death_cross":
            return with_confidence_result(st, *_cross_sma(close, p["fast"], p["slow"], bullish=False))
        if st == "macd_bullish_crossover":
            return with_confidence_result(st, *_cross_macd(close, p, bullish=True))
        if st == "macd_bearish_crossover":
            return with_confidence_result(st, *_cross_macd(close, p, bullish=False))
        if st == "rsi_oversold":
            return with_confidence_result(st, *_rsi_threshold(ind, p["period"], p["threshold"], below=True))
        if st == "rsi_overbought":
            return with_confidence_result(st, *_rsi_threshold(ind, p["period"], p["threshold"], below=False))
        if st == "pct_from_ma":
            return with_confidence_result(st, *_pct_from_ma(close, p))
        if st == "pct_from_ath":
            return with_confidence_result(st, *_pct_from_ath(df, p["min_pct_below_ath"]))
        if st == "bbands_breakout":
            return with_confidence_result(st, *_bb_breakout(close, p))
        if st == "bull_flag":
            return with_confidence_result(st, *_bull_flag(df, p))
    except Exception as e:
        log.exception("evaluate %s: %s", st, e)
        return False, with_confidence(st, False, {"error": str(e)})

    return False, with_confidence(st, False, {"error": "unknown_signal"})


def _cross_sma(
    close: pd.Series, fast: int, slow: int, bullish: bool
) -> tuple[bool, dict[str, Any]]:
    """Detect a latest-bar fast/slow SMA crossover."""
    if len(close) < slow + 2:
        return False, {"reason": "insufficient_bars", "need": slow + 2}
    _, a = calc.moving_average(close, fast, "sma")
    _, b = calc.moving_average(close, slow, "sma")
    if a is None or b is None or a.empty or b.empty:
        return False, {"reason": "sma_fail"}
    vals = calc.latest_cross_values(a, b)
    if vals is None:
        return False, {"reason": "nan"}
    trig = calc.crossed(
        vals.lhs_prev,
        vals.lhs_cur,
        vals.rhs_prev,
        vals.rhs_cur,
        "bullish" if bullish else "bearish",
    )
    return trig, {
        f"sma_{fast}": vals.lhs_cur,
        f"sma_{slow}": vals.rhs_cur,
    }


def _cross_macd(
    close: pd.Series, p: dict[str, Any], bullish: bool
) -> tuple[bool, dict[str, Any]]:
    """Detect a latest-bar MACD/signal line crossover."""
    macd = calc.macd(close, fast=int(p["fast"]), slow=int(p["slow"]), signal=int(p["signal"]))
    if macd is None:
        return False, {"reason": "macd_fail"}
    out, cols = macd
    if len(out) < 2:
        return False, {"reason": "macd_fail"}
    if cols is None:
        return False, {"reason": "macd_col"}
    vals = calc.latest_cross_values(out[cols.macd], out[cols.signal])
    if vals is None:
        return False, {"reason": "nan"}
    trig = calc.crossed(
        vals.lhs_prev,
        vals.lhs_cur,
        vals.rhs_prev,
        vals.rhs_cur,
        "bullish" if bullish else "bearish",
    )
    return trig, {"macd": vals.lhs_cur, "signal": vals.rhs_cur}


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
    ma_type, line = calc.moving_average(close, n, str(p.get("ma_type", "sma")))
    if line is None or line.empty:
        return False, {"reason": "ma_fail"}
    pr = float(close.iloc[-1])
    ma = float(line.iloc[-1])
    if ma == 0 or pd.isna(ma):
        return False, {"reason": "ma_nan"}
    diff = float(calc.pct_distance_from_ma(close, line).iloc[-1])
    threshold = float(p["pct"])
    trig = diff <= threshold
    return trig, {"price": pr, f"{ma_type}_{n}": ma, "diff_pct": diff, "threshold_pct": threshold}


def _pct_from_ath(df: pd.DataFrame, min_pct: float) -> tuple[bool, dict[str, Any]]:
    """Check whether latest close is at least N percent below the series high."""
    hi = df["High"] if "High" in df else df["Close"]
    ath = float(hi.max())
    pr = float(df["Close"].iloc[-1])
    if ath == 0:
        return False, {"reason": "ath_zero"}
    dist = (pr - ath) / ath * 100.0
    trig = dist <= -min_pct
    return trig, {"ath": ath, "close": pr, "pct_from_ath": dist, "threshold_pct": float(min_pct)}


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

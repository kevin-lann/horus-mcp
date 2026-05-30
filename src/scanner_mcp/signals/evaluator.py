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
        if st == "cup_and_handle":
            return with_confidence_result(st, *_cup_and_handle(df, p))
        if st == "golden_pocket":
            return with_confidence_result(st, *_golden_pocket(df, p))
        if st == "head_and_shoulders":
            return with_confidence_result(st, *_head_and_shoulders(df, p, inverse=False))
        if st == "inverse_head_and_shoulders":
            return with_confidence_result(st, *_head_and_shoulders(df, p, inverse=True))
        if st == "double_bottom":
            return with_confidence_result(st, *_double_reversal(df, p, bullish=True))
        if st == "double_top":
            return with_confidence_result(st, *_double_reversal(df, p, bullish=False))
        if st == "buyable_gap_up":
            return with_confidence_result(st, *_buyable_gap_up(df, p))
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


def _cup_and_handle(df: pd.DataFrame, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Detect a simplified cup-and-handle nearing or clearing a breakout."""
    close = df["Close"].astype(float)
    lookback = int(p["lookback"])
    handle_days = int(p["handle_days"])
    if len(close) < lookback or handle_days < 2 or lookback <= handle_days + 4:
        return False, {"reason": "insufficient_bars"}

    window = close.iloc[-lookback:]
    cup = window.iloc[:-handle_days]
    handle = window.iloc[-handle_days:]
    if cup.empty or handle.empty:
        return False, {"reason": "insufficient_bars"}

    third = max(len(cup) // 3, 1)
    left_zone = cup.iloc[:third]
    mid_zone = cup.iloc[third : len(cup) - third]
    right_zone = cup.iloc[len(cup) - third :]
    if left_zone.empty or mid_zone.empty or right_zone.empty:
        return False, {"reason": "insufficient_bars"}

    left_peak = float(left_zone.max())
    cup_low = float(mid_zone.min())
    right_peak = float(right_zone.max())
    peak_avg = max((left_peak + right_peak) / 2.0, 1e-9)
    peak_tolerance_pct = abs(left_peak - right_peak) / peak_avg * 100.0
    cup_depth_pct = (peak_avg - cup_low) / peak_avg * 100.0
    handle_high = float(handle.max())
    handle_low = float(handle.min())
    handle_pullback_pct = (right_peak - handle_low) / max(abs(right_peak), 1e-9) * 100.0
    breakout_gap_pct = (float(window.iloc[-1]) - handle_high) / max(abs(handle_high), 1e-9) * 100.0

    trig = (
        cup_depth_pct >= float(p["min_cup_depth_pct"])
        and peak_tolerance_pct <= float(p["peak_tolerance_pct"])
        and handle_pullback_pct <= float(p["max_handle_pullback_pct"])
        and breakout_gap_pct >= -float(p["breakout_buffer_pct"])
    )
    return trig, {
        "left_peak": left_peak,
        "right_peak": right_peak,
        "cup_low": cup_low,
        "cup_depth_pct": cup_depth_pct,
        "peak_tolerance_pct": peak_tolerance_pct,
        "handle_pullback_pct": handle_pullback_pct,
        "breakout_gap_pct": breakout_gap_pct,
    }


def _golden_pocket(df: pd.DataFrame, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Detect whether the latest close sits inside a bullish golden-pocket retracement."""
    close = df["Close"].astype(float)
    lookback = int(p["lookback"])
    if len(close) < lookback:
        return False, {"reason": "insufficient_bars"}

    window = close.iloc[-lookback:]
    low_idx = window.idxmin()
    low_pos = window.index.get_loc(low_idx)
    if low_pos >= len(window) - 2:
        return False, {"reason": "swing_not_found"}

    upswing = window.iloc[low_pos:]
    high_idx = upswing.idxmax()
    high_pos = window.index.get_loc(high_idx)
    if high_pos <= low_pos:
        return False, {"reason": "swing_not_found"}

    swing_low = float(window.iloc[low_pos])
    swing_high = float(window.iloc[high_pos])
    if swing_low <= 0 or swing_high <= swing_low:
        return False, {"reason": "swing_not_found"}

    current = float(window.iloc[-1])
    swing_range = swing_high - swing_low
    zone_top = swing_high - swing_range * float(p["retrace_low"])
    zone_bottom = swing_high - swing_range * float(p["retrace_high"])
    retracement_pct = (swing_high - current) / swing_range if swing_range else 0.0
    swing_pct = (swing_high - swing_low) / swing_low * 100.0

    trig = (
        swing_pct >= float(p["min_swing_pct"])
        and current >= zone_bottom
        and current <= zone_top
        and high_pos < len(window) - 1
    )
    return trig, {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "swing_pct": swing_pct,
        "close": current,
        "golden_pocket_low": zone_bottom,
        "golden_pocket_high": zone_top,
        "retracement_pct": retracement_pct * 100.0,
    }


def _head_and_shoulders(df: pd.DataFrame, p: dict[str, Any], *, inverse: bool) -> tuple[bool, dict[str, Any]]:
    """Detect a segmented head-and-shoulders or inverse head-and-shoulders pattern."""
    close = df["Close"].astype(float)
    lookback = int(p["lookback"])
    if len(close) < lookback or lookback < 10:
        return False, {"reason": "insufficient_bars"}

    window = close.iloc[-lookback:]
    seg = max(lookback // 5, 1)
    parts = [window.iloc[i * seg : (i + 1) * seg] for i in range(4)]
    parts.append(window.iloc[4 * seg :])
    if any(part.empty for part in parts):
        return False, {"reason": "insufficient_bars"}

    if inverse:
        left_shoulder = float(parts[0].min())
        trough1 = float(parts[1].max())
        head = float(parts[2].min())
        trough2 = float(parts[3].max())
        right_shoulder = float(parts[4].min())
        shoulder_avg = max(abs((left_shoulder + right_shoulder) / 2.0), 1e-9)
        shoulder_tolerance_pct = abs(left_shoulder - right_shoulder) / shoulder_avg * 100.0
        head_margin_pct = (shoulder_avg - head) / shoulder_avg * 100.0
        neckline = (trough1 + trough2) / 2.0
        breakout_pct = (float(window.iloc[-1]) - neckline) / max(abs(neckline), 1e-9) * 100.0
        trig = (
            shoulder_tolerance_pct <= float(p["shoulder_tolerance_pct"])
            and head_margin_pct >= float(p["min_head_margin_pct"])
            and float(window.iloc[-1]) > neckline
        )
    else:
        left_shoulder = float(parts[0].max())
        trough1 = float(parts[1].min())
        head = float(parts[2].max())
        trough2 = float(parts[3].min())
        right_shoulder = float(parts[4].max())
        shoulder_avg = max(abs((left_shoulder + right_shoulder) / 2.0), 1e-9)
        shoulder_tolerance_pct = abs(left_shoulder - right_shoulder) / shoulder_avg * 100.0
        head_margin_pct = (head - shoulder_avg) / shoulder_avg * 100.0
        neckline = (trough1 + trough2) / 2.0
        breakout_pct = (neckline - float(window.iloc[-1])) / max(abs(neckline), 1e-9) * 100.0
        trig = (
            shoulder_tolerance_pct <= float(p["shoulder_tolerance_pct"])
            and head_margin_pct >= float(p["min_head_margin_pct"])
            and float(window.iloc[-1]) < neckline
        )

    return trig, {
        "left_shoulder": left_shoulder,
        "head": head,
        "right_shoulder": right_shoulder,
        "neckline": neckline,
        "shoulder_tolerance_pct": shoulder_tolerance_pct,
        "head_margin_pct": head_margin_pct,
        "breakout_pct": breakout_pct,
    }


def _double_reversal(df: pd.DataFrame, p: dict[str, Any], *, bullish: bool) -> tuple[bool, dict[str, Any]]:
    """Detect a simplified double-bottom or double-top reversal and neckline break."""
    close = df["Close"].astype(float)
    lookback = int(p["lookback"])
    if len(close) < lookback or lookback < 6:
        return False, {"reason": "insufficient_bars"}

    window = close.iloc[-lookback:]
    half = lookback // 2
    left = window.iloc[:half]
    right = window.iloc[half:]
    if left.empty or right.empty:
        return False, {"reason": "insufficient_bars"}

    if bullish:
        low1 = float(left.min())
        low2 = float(right.min())
        low_avg = max(abs((low1 + low2) / 2.0), 1e-9)
        tolerance_pct = abs(low1 - low2) / low_avg * 100.0
        mid_peak = float(window.iloc[half // 2 : half + max(len(right) // 2, 1)].max())
        rebound_pct = (mid_peak - min(low1, low2)) / max(abs(min(low1, low2)), 1e-9) * 100.0
        breakout_pct = (float(window.iloc[-1]) - mid_peak) / max(abs(mid_peak), 1e-9) * 100.0
        trig = (
            tolerance_pct <= float(p["peak_tolerance_pct"])
            and rebound_pct >= float(p["min_rebound_pct"])
            and float(window.iloc[-1]) > mid_peak
        )
        return trig, {
            "first_low": low1,
            "second_low": low2,
            "neckline": mid_peak,
            "tolerance_pct": tolerance_pct,
            "rebound_pct": rebound_pct,
            "breakout_pct": breakout_pct,
        }

    high1 = float(left.max())
    high2 = float(right.max())
    high_avg = max(abs((high1 + high2) / 2.0), 1e-9)
    tolerance_pct = abs(high1 - high2) / high_avg * 100.0
    mid_trough = float(window.iloc[half // 2 : half + max(len(right) // 2, 1)].min())
    pullback_pct = (max(high1, high2) - mid_trough) / max(abs(max(high1, high2)), 1e-9) * 100.0
    breakout_pct = (mid_trough - float(window.iloc[-1])) / max(abs(mid_trough), 1e-9) * 100.0
    trig = (
        tolerance_pct <= float(p["peak_tolerance_pct"])
        and pullback_pct >= float(p["min_pullback_pct"])
        and float(window.iloc[-1]) < mid_trough
    )
    return trig, {
        "first_high": high1,
        "second_high": high2,
        "neckline": mid_trough,
        "tolerance_pct": tolerance_pct,
        "pullback_pct": pullback_pct,
        "breakout_pct": breakout_pct,
    }


def _buyable_gap_up(df: pd.DataFrame, p: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Detect a latest-session buyable gap-up style breakout."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return False, {"reason": "missing_ohlcv"}

    lookback = int(p["volume_lookback"])
    if len(df) < max(lookback + 1, 2):
        return False, {"reason": "insufficient_bars"}

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    prev_high = float(previous["High"])
    if prev_high <= 0:
        return False, {"reason": "prior_zero"}

    gap_pct = (float(latest["Open"]) - prev_high) / prev_high * 100.0
    day_range = max(float(latest["High"]) - float(latest["Low"]), 1e-9)
    close_position = (float(latest["Close"]) - float(latest["Low"])) / day_range
    avg_volume = float(df["Volume"].astype(float).iloc[-(lookback + 1) : -1].mean())
    if avg_volume <= 0:
        return False, {"reason": "volume_zero"}
    volume_ratio = float(latest["Volume"]) / avg_volume

    trig = (
        gap_pct >= float(p["min_gap_pct"])
        and close_position >= float(p["min_close_position"])
        and volume_ratio >= float(p["min_volume_ratio"])
    )
    return trig, {
        "gap_pct": gap_pct,
        "close_position": close_position,
        "volume_ratio": volume_ratio,
        "prev_high": prev_high,
        "open": float(latest["Open"]),
        "close": float(latest["Close"]),
    }

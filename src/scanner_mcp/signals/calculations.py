"""Shared signal calculation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from scanner_mcp.indicators import ta


CrossDirection = Literal["bullish", "bearish"]


@dataclass(frozen=True)
class CrossValues:
    """Previous and current values around a two-series crossover."""

    lhs_prev: float
    lhs_cur: float
    rhs_prev: float
    rhs_cur: float


@dataclass(frozen=True)
class MacdColumns:
    """Column names for MACD line, signal line, and histogram outputs."""

    macd: str
    signal: str
    hist: str | None = None


def crossed(
    lhs_prev: float,
    lhs_cur: float,
    rhs_prev: float,
    rhs_cur: float,
    direction: CrossDirection,
) -> bool:
    """Return whether two series crossed in the requested direction."""
    if direction == "bullish":
        return lhs_prev <= rhs_prev and lhs_cur > rhs_cur
    return lhs_prev >= rhs_prev and lhs_cur < rhs_cur


def latest_cross_values(lhs: pd.Series, rhs: pd.Series) -> CrossValues | None:
    """Return previous/current values for the latest two bars of two aligned series."""
    if lhs is None or rhs is None or len(lhs) < 2 or len(rhs) < 2:
        return None
    vals = (lhs.iloc[-2], lhs.iloc[-1], rhs.iloc[-2], rhs.iloc[-1])
    if any(pd.isna(x) for x in vals):
        return None
    return CrossValues(*(float(x) for x in vals))


def cross_indexes(lhs: pd.Series, rhs: pd.Series, direction: CrossDirection) -> list[int]:
    """Return indexes where two series cross in the requested direction."""
    out: list[int] = []
    for i in range(1, min(len(lhs), len(rhs))):
        vals = (lhs.iloc[i - 1], lhs.iloc[i], rhs.iloc[i - 1], rhs.iloc[i])
        if any(pd.isna(x) for x in vals):
            continue
        if crossed(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), direction):
            out.append(i)
    return out


def moving_average(close: pd.Series, period: int, ma_type: str = "sma") -> tuple[str, pd.Series]:
    """Return a normalized MA type and its series."""
    normalized = str(ma_type).lower()
    if normalized == "ema":
        return "ema", ta.ema(close, length=period)
    return "sma", ta.sma(close, length=period)


def macd(
    close: pd.Series,
    *,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.DataFrame, MacdColumns | None] | None:
    """Return MACD output and resolved line columns."""
    out = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if out is None or out.empty:
        return None
    return out, macd_columns(out)


def pct_distance_from_ma(close: pd.Series, ma: pd.Series) -> pd.Series:
    """Return absolute percent distance between price and a moving average."""
    return (close - ma).abs() / ma.abs() * 100.0


def rsi_threshold_cross_indexes(
    close: pd.Series,
    *,
    period: int,
    threshold: float,
    below: bool,
) -> list[int]:
    """Return indexes where RSI crosses into an overbought/oversold zone."""
    r = ta.rsi(close, length=period)
    if r is None or r.empty:
        return []
    prev = r.shift(1)
    out: list[int] = []
    for i in range(1, len(r)):
        if pd.isna(r.iloc[i]) or pd.isna(prev.iloc[i]):
            continue
        if below and prev.iloc[i] >= threshold and r.iloc[i] < threshold:
            out.append(i)
        if not below and prev.iloc[i] <= threshold and r.iloc[i] > threshold:
            out.append(i)
    return out


def macd_columns(out: pd.DataFrame | None) -> MacdColumns | None:
    """Return MACD output column names regardless of parameter suffix."""
    if out is None or out.empty:
        return None
    cols = list(out.columns)
    macd_col = next(
        (
            c
            for c in cols
            if str(c).startswith("MACD_")
            and not str(c).startswith("MACDs")
            and not str(c).startswith("MACDh")
        ),
        None,
    )
    sig_col = next((c for c in cols if str(c).startswith("MACDs_")), None)
    hist_col = next((c for c in cols if str(c).startswith("MACDh_")), None)
    if not macd_col or not sig_col:
        return None
    return MacdColumns(macd=str(macd_col), signal=str(sig_col), hist=str(hist_col) if hist_col else None)

"""Shared chart data preparation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scanner_mcp.charts.params import as_bool, positive_int
from scanner_mcp.data.provider import DataProvider


def close_series(provider: DataProvider, symbol: str, period: str, interval: str = "1d") -> pd.Series:
    """Fetch one close series with a normalized symbol label."""
    sym = str(symbol).strip().upper()
    df = provider.get_history(sym, period=period, interval=interval)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"No price data for {sym}")
    close = df["Close"].astype(float).dropna()
    if close.empty:
        raise ValueError(f"No close data for {sym}")
    close.name = sym
    return close


def aligned_close_frame(
    provider: DataProvider,
    symbols: list[str],
    period: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Return close series aligned on a shared date index."""
    if not symbols:
        raise ValueError("symbols must not be empty")
    frames = [close_series(provider, symbol, period, interval) for symbol in symbols]
    aligned = pd.concat(frames, axis=1, join="inner").dropna(how="any")
    if aligned.empty:
        raise ValueError("No overlapping price history for requested symbols")
    return aligned


def normalize_to_100(series: pd.Series) -> pd.Series:
    """Normalize a series to an index value of 100 at the first point."""
    base = float(series.iloc[0])
    if not np.isfinite(base) or base == 0:
        raise ValueError(f"Cannot normalize series {series.name or ''}: invalid starting value")
    return series / base * 100.0


def contiguous_true_spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Collapse a boolean mask into contiguous x-axis spans."""
    if mask.empty:
        return []
    clean_mask = mask.fillna(False).astype(bool)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for ts, is_true in clean_mask.items():
        ts = pd.Timestamp(ts)
        if is_true and start is None:
            start = ts
        if not is_true and start is not None and prev is not None:
            spans.append((start, prev))
            start = None
        prev = ts
    if start is not None and prev is not None:
        spans.append((start, prev))
    return spans


def price_history_has_fib(params: dict[str, Any]) -> bool:
    """Return whether Fibonacci retracement overlays are enabled."""
    return as_bool(params.get("show_fib_retracement", params.get("fib_retracement", False)))


def fib_x_padding(index: pd.Index) -> tuple[list[Any], Any] | None:
    """Return an extended x-axis range and label anchor beyond the latest bar."""
    if len(index) < 2:
        return None
    if isinstance(index, pd.DatetimeIndex):
        start = index[0]
        end = index[-1]
        span = end - start
        diffs = index.to_series().diff().dropna()
        step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
        pad = max(span * 0.08, step * 8)
        padded_end = end + pad
        return [start, padded_end], end + (pad * 0.9)
    try:
        start = float(index[0])
        end = float(index[-1])
    except (TypeError, ValueError):
        return None
    span = end - start
    if not np.isfinite(span) or span <= 0:
        return None
    pad = span * 0.08
    padded_end = end + pad
    return [start, padded_end], end + (pad * 0.9)


def fib_label_x(index: pd.Index) -> Any:
    """Return the x coordinate used for Fibonacci labels."""
    padding = fib_x_padding(index)
    if padding is None:
        return index[-1]
    return padding[1]


def price_history_indicator_lookback_bars(params: dict[str, Any]) -> int:
    """Return the largest warm-up window required by enabled overlays."""
    lookbacks: list[int] = []
    if as_bool(params.get("show_bollinger_bands", params.get("bollinger_bands", False))):
        lookbacks.append(positive_int(params.get("bb_period", 20), 20))
    if as_bool(params.get("show_ma", params.get("ma", False))):
        lookbacks.append(positive_int(params.get("ma_period", 50), 50))
    if as_bool(params.get("show_ema", params.get("ema", False))):
        lookbacks.append(positive_int(params.get("ema_period", 21), 21))
    if as_bool(params.get("show_ma_cloud", params.get("ma_cloud", False))):
        lookbacks.extend(
            [
                positive_int(params.get("ma_cloud_fast", 50), 50),
                positive_int(params.get("ma_cloud_slow", 200), 200),
            ]
        )
    return max(lookbacks, default=0)


def price_history_preroll_delta(interval: str, bars: int) -> pd.Timedelta:
    """Approximate a bar-based lookback as calendar time for yfinance start dates."""
    if bars <= 0:
        return pd.Timedelta(0)
    interval_key = str(interval).strip().lower()
    if interval_key in {"1wk", "1w", "wk", "week", "weekly"}:
        return pd.Timedelta(days=(bars * 9) + 14)
    if interval_key in {"1mo", "3mo", "1mth", "month", "monthly"}:
        return pd.Timedelta(days=(bars * 35) + 31)
    if interval_key in {"5d"}:
        return pd.Timedelta(days=(bars * 7) + 7)
    return pd.Timedelta(days=(bars * 2) + 7)


def price_history_frames(
    provider: DataProvider,
    symbol: str,
    period: str,
    interval: str,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return visible history plus optional pre-roll history for indicator warm-up."""
    visible_df = provider.get_history(symbol, period=period, interval=interval)
    if visible_df.empty:
        return visible_df, visible_df
    lookback_bars = price_history_indicator_lookback_bars(params)
    if lookback_bars <= 0:
        return visible_df, visible_df
    visible_index = pd.DatetimeIndex(visible_df.index)
    preroll_start = visible_index[0] - price_history_preroll_delta(interval, lookback_bars)
    visible_end = visible_index[-1] + pd.Timedelta(days=1)
    full_df = provider.get_history(symbol, interval=interval, start=preroll_start, end=visible_end)
    if full_df.empty:
        return visible_df, visible_df
    return visible_df, full_df

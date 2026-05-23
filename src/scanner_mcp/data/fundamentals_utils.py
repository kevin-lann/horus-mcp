"""Shared fundamentals and valuation time-series helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def empty_series() -> pd.Series:
    """Return a fresh empty float series."""
    return pd.Series(dtype=float)


def source_series(values: pd.Series, source: str) -> pd.Series:
    """Attach source metadata to a series and return it."""
    values.attrs["source"] = source
    return values


def to_naive_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize timestamps to timezone-naive calendar dates."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx.normalize()).astype("datetime64[ns]")


def merge_asof_price_over_eps(
    close: pd.Series,
    anchor_dates: pd.DatetimeIndex,
    eps_values: np.ndarray,
) -> pd.Series:
    """Align a price series to the most recent EPS value known on each date."""
    df_sorted = close.sort_index().astype(float)
    hist = pd.DataFrame(
        {
            "asof": to_naive_dates(pd.DatetimeIndex(df_sorted.index)),
            "close": df_sorted.values,
        }
    )
    right = pd.DataFrame(
        {
            "anchor": to_naive_dates(pd.DatetimeIndex(anchor_dates)),
            "eps": eps_values.astype(float),
        }
    ).sort_values("anchor")
    merged = pd.merge_asof(hist, right, left_on="asof", right_on="anchor", direction="backward")
    denom = merged["eps"].to_numpy(dtype=float)
    num = merged["close"].to_numpy(dtype=float)
    pe_vals = np.where(np.isfinite(denom) & (denom > 0), num / denom, np.nan)
    return pd.Series(pe_vals, index=df_sorted.index, dtype=float).reindex(close.index)


def series_from_alpha_vantage_rows(rows: list[dict[str, Any]], value_keys: tuple[str, ...]) -> pd.Series:
    """Parse Alpha Vantage rows into a dated numeric series."""
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
        return empty_series()
    out = pd.Series({date: value for date, value in parsed}, dtype=float).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def statement_metric_series(stmt: pd.DataFrame, metric: str) -> pd.Series:
    """Extract a supported metric from a yfinance-style income statement."""
    if stmt is None or stmt.empty:
        return empty_series()
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
    return empty_series()

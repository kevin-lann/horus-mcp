"""Fundamental overlay chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner_mcp.charts.rendering import fig_to_b64
from scanner_mcp.data.provider import DataProvider


def format_large_axis(value: float) -> str:
    """Format large values for axis tick labels."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:g}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:g}K"
    return f"{value:g}"


def large_axis_tick_values(values: pd.Series) -> tuple[list[float], list[str]]:
    """Build evenly spaced tick values for large-scale fundamentals axes."""
    vals = values.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [], []
    low = min(0.0, float(vals.min()))
    high = float(vals.max())
    if high <= low:
        return [high], [format_large_axis(high)]
    ticks = np.linspace(low, high, 5)
    return [float(value) for value in ticks], [format_large_axis(float(value)) for value in ticks]


def fundamental_overlay(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Overlay price history with revenue or earnings bars from income statements."""
    symbol = str(params.get("symbol", "AAPL")).strip().upper()
    period = params.get("period", "5y")
    interval = params.get("interval", "1d")
    metric = str(params.get("metric", "revenue")).strip().lower()
    frequency = str(params.get("frequency", "quarterly")).strip().lower()
    price_style = str(params.get("price_style", "candlestick")).strip().lower()

    df = provider.get_history(symbol, period=str(period), interval=str(interval))
    if df.empty:
        raise ValueError("No price data")
    fundamentals = provider.get_fundamental_series(symbol, metric, frequency)
    if fundamentals.empty:
        raise ValueError(f"No {metric} data from Alpha Vantage or Yahoo fallback")
    source = str(fundamentals.attrs.get("source", "provider"))

    start = pd.Timestamp(df.index[0])
    end = pd.Timestamp(df.index[-1])
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    if end.tzinfo is not None:
        end = end.tz_convert(None)
    fundamentals_index = pd.DatetimeIndex(fundamentals.index)
    if fundamentals_index.tz is not None:
        fundamentals_index = fundamentals_index.tz_convert("UTC").tz_localize(None)
    fundamentals = pd.Series(fundamentals.to_numpy(dtype=float), index=fundamentals_index)
    visible = fundamentals[(fundamentals.index >= start) & (fundamentals.index <= end)]
    if visible.empty:
        raise ValueError(f"No {metric} data within visible price period")
    ticks, ticktext = large_axis_tick_values(visible)

    metric_title = "Revenue" if metric == "revenue" else "Earnings"
    frequency_title = "Annual" if frequency in {"annual", "yearly", "year"} else "Quarterly"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_color = "#0f766e" if metric == "revenue" else "#7c2d12"
    fig.add_trace(go.Bar(x=visible.index, y=visible.values, name=f"{frequency_title} {metric_title} ({source})", marker={"color": bar_color}, opacity=0.34, hovertemplate=f"{frequency_title} {metric_title}<br>%{{x|%Y-%m-%d}}<br>%{{customdata}}<extra></extra>", customdata=[format_large_axis(float(value)) for value in visible.values]), secondary_y=True)
    if price_style == "line":
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].astype(float), name=f"{symbol} Close", mode="lines", line={"color": "#111827", "width": 1.9}), secondary_y=False)
    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=symbol), secondary_y=False)
    fig.update_layout(title=f"{symbol} price vs {frequency_title.lower()} {metric_title.lower()}", xaxis_title="Date", xaxis_rangeslider_visible=False, yaxis_title="Price", yaxis2_title=metric_title, width=1150, height=680, margin={"l": 56, "r": 72, "t": 76, "b": 48}, legend={"orientation": "h", "x": 0.02, "xanchor": "left", "y": 1, "yanchor": "top"}, hovermode="x unified", bargap=0.35, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)", secondary_y=False)
    fig.update_yaxes(showgrid=False, tickmode="array", tickvals=ticks, ticktext=ticktext, secondary_y=True)
    return {"mime": "image/png", "data": fig_to_b64(fig, "fundamental_overlay")}

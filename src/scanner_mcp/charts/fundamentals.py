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


def _naive_series(series: pd.Series) -> pd.Series:
    """Normalize a dated series to a timezone-naive DatetimeIndex."""
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.Series(series.to_numpy(dtype=float), index=index).sort_index()


def _naive_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dated frame to a timezone-naive DatetimeIndex."""
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out = df.copy()
    out.index = index
    return out.sort_index()


def _visible_series(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Clip a dated series to the visible chart range."""
    return series[(series.index >= start) & (series.index <= end)]


def _growth_rate(series: pd.Series, periods: int) -> pd.Series:
    """Return period-over-period growth in percent."""
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan)
    return clean.pct_change(periods=periods) * 100.0


def _fundamental_series_bundle(provider: DataProvider, symbol: str, frequency: str) -> tuple[pd.Series, pd.Series]:
    """Fetch revenue and earnings, preferring a matched bundle when the provider supports it."""
    bundle_fn = getattr(provider, "get_fundamental_bundle", None)
    if callable(bundle_fn):
        bundle = bundle_fn(symbol, ["revenue", "earnings"], frequency)
        revenue = bundle.get("revenue", pd.Series(dtype=float))
        earnings = bundle.get("earnings", pd.Series(dtype=float))
        return revenue, earnings
    return (
        provider.get_fundamental_series(symbol, "revenue", frequency),
        provider.get_fundamental_series(symbol, "earnings", frequency),
    )


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


def fundamental_momentum(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Show price, revenue growth, earnings growth or margin, and valuation."""
    symbol = str(params.get("symbol", "AAPL")).strip().upper()
    period = str(params.get("period", "5y"))
    interval = str(params.get("interval", "1d"))
    frequency = str(params.get("frequency", "quarterly")).strip().lower()
    price_style = str(params.get("price_style", "line")).strip().lower()
    profitability_metric = str(params.get("profitability_metric", "net_margin")).strip().lower()
    if profitability_metric not in {"net_margin", "earnings_growth"}:
        raise ValueError("profitability_metric must be net_margin or earnings_growth")

    df = provider.get_history(symbol, period=period, interval=interval)
    if df.empty:
        raise ValueError("No price data")
    df = _naive_frame(df)
    close = df["Close"].astype(float)
    pe = _naive_series(provider.get_historical_pe_series(symbol, close).astype(float))
    pe = pe.replace([np.inf, -np.inf], np.nan)
    if pe.dropna().empty:
        raise ValueError("No valuation history available")

    revenue, earnings = _fundamental_series_bundle(provider, symbol, frequency)
    if revenue.empty or earnings.empty:
        raise ValueError("Revenue and earnings history are required for fundamental momentum")
    revenue = _naive_series(revenue)
    earnings = _naive_series(earnings)

    start = pd.Timestamp(df.index[0])
    end = pd.Timestamp(df.index[-1])
    if start.tzinfo is not None:
        start = start.tz_convert(None)
    if end.tzinfo is not None:
        end = end.tz_convert(None)
    revenue_visible = _visible_series(revenue, start, end)
    earnings_visible = _visible_series(earnings, start, end)
    if revenue_visible.empty or earnings_visible.empty:
        raise ValueError("No fundamental data within visible price period")

    yoy_periods = 4 if frequency in {"quarterly", "quarter", "q"} else 1
    revenue_growth = _visible_series(_growth_rate(revenue, yoy_periods), start, end).dropna()
    earnings_growth = _visible_series(_growth_rate(earnings, yoy_periods), start, end).dropna()
    margin = _visible_series((earnings / revenue.replace(0.0, np.nan)) * 100.0, start, end).replace([np.inf, -np.inf], np.nan).dropna()

    panel_three = earnings_growth if profitability_metric == "earnings_growth" else margin
    if revenue_growth.empty:
        raise ValueError("Not enough revenue history to compute YoY growth")
    if panel_three.empty:
        raise ValueError("Not enough fundamental history for requested profitability_metric")

    pe_visible = pe.dropna()
    pe_visible = pe_visible[(pe_visible.index >= start) & (pe_visible.index <= end)]
    if pe_visible.empty:
        raise ValueError("No valuation history within visible period")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.38, 0.19, 0.19, 0.24],
    )
    if price_style == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=symbol,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=close, name=symbol, mode="lines", line={"color": "#111827", "width": 2.0}),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=revenue_growth.index,
            y=revenue_growth.values,
            name="Revenue YoY %",
            marker={"color": np.where(revenue_growth.values >= 0, "#0f766e", "#b91c1c")},
        ),
        row=2,
        col=1,
    )

    panel_three_name = "Earnings YoY %" if profitability_metric == "earnings_growth" else "Net Margin %"
    fig.add_trace(
        go.Bar(
            x=panel_three.index,
            y=panel_three.values,
            name=panel_three_name,
            marker={"color": np.where(panel_three.values >= 0, "#1d4ed8", "#c2410c")},
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pe_visible.index,
            y=pe_visible.values,
            name="P/E",
            mode="lines",
            line={"color": "#7c3aed", "width": 1.8},
        ),
        row=4,
        col=1,
    )
    fig.add_hline(y=float(pe_visible.median()), line={"color": "rgba(124,58,237,0.35)", "dash": "dash"}, row=4, col=1)

    fig.update_layout(
        title=f"{symbol} fundamental momentum",
        width=1180,
        height=980,
        margin={"l": 64, "r": 52, "t": 78, "b": 42},
        legend={"orientation": "h", "x": 0.01, "xanchor": "left", "y": 1.03, "yanchor": "top"},
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.25,
        xaxis_rangeslider_visible=False,
    )
    for row in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=True, zerolinecolor="rgba(15,23,42,0.10)", row=row, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Revenue YoY %", row=2, col=1)
    fig.update_yaxes(title_text=panel_three_name, row=3, col=1)
    fig.update_yaxes(title_text="P/E", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    return {"mime": "image/png", "data": fig_to_b64(fig, "fundamental_momentum")}

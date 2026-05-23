"""Price-history chart construction and overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner_mcp.charts.data import fib_label_x, fib_x_padding, price_history_frames, price_history_has_fib
from scanner_mcp.charts.layout import price_history_legend_layout
from scanner_mcp.charts.params import as_bool, positive_int
from scanner_mcp.charts.rendering import fig_to_b64
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.indicators import ta


def trailing_pe_series(provider: DataProvider, symbol: str, close: pd.Series) -> pd.Series:
    """P/E aligned to `close`, provided by the configured data provider."""
    return provider.get_historical_pe_series(str(symbol), close)


def add_price_trace(fig: go.Figure, trace: Any, row: int | None = None) -> None:
    """Add a trace to a figure or subplot."""
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def anchored_vwap(df: pd.DataFrame, anchor: Any = None) -> pd.Series:
    """Return VWAP anchored to the first visible bar, or to `anchor` when supplied."""
    idx = pd.DatetimeIndex(df.index)
    typical = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    typical = typical.where(volume > 0)
    if anchor:
        anchor_ts = pd.Timestamp(anchor)
        if idx.tz is not None:
            anchor_ts = anchor_ts.tz_localize(idx.tz) if anchor_ts.tzinfo is None else anchor_ts.tz_convert(idx.tz)
        elif anchor_ts.tzinfo is not None:
            anchor_ts = anchor_ts.tz_convert(None)
        mask = idx >= anchor_ts
        typical = typical.where(mask)
        volume = volume.where(mask)
    cum_volume = volume.cumsum()
    return (typical * volume).cumsum() / cum_volume.replace(0, np.nan)


def add_fib_retracement_traces(
    fig: go.Figure,
    x: pd.Index,
    high: pd.Series,
    low: pd.Series,
    row: int | None = None,
) -> None:
    """Add Fibonacci retracement levels to a figure."""
    low_idx = low.idxmin()
    high_idx = high.idxmax()
    swing_low = float(low.loc[low_idx])
    swing_high = float(high.loc[high_idx])
    span = swing_high - swing_low
    if not np.isfinite(span) or span <= 0:
        return
    uptrend = low_idx <= high_idx
    label_x = fib_label_x(pd.Index(x))
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    for ratio in ratios:
        level = swing_high - span * ratio if uptrend else swing_low + span * ratio
        label = f"{ratio:g} {level:.2f}"
        add_price_trace(
            fig,
            go.Scatter(
                x=[x[0], label_x],
                y=[level, level],
                name=f"Fib {ratio:g}",
                mode="lines",
                line={"color": "rgba(75, 85, 99, 0.35)", "width": 1},
                hovertemplate=f"{label}<extra></extra>",
                showlegend=False,
            ),
            row,
        )
        fig.add_annotation(
            x=0.995,
            xref="paper",
            xanchor="right",
            y=level + 2,
            yref="y" if row in (None, 1) else f"y{row}",
            text=label,
            showarrow=False,
            font={"color": "#4b5563", "size": 8},
            align="right",
            bgcolor="rgba(255,255,255,0.0)",
        )


def apply_fib_x_padding(fig: go.Figure, df: pd.DataFrame, params: dict[str, Any], rows: int = 1) -> None:
    """Extend the x-axis when Fibonacci overlays are enabled."""
    if not price_history_has_fib(params):
        return
    padding = fib_x_padding(pd.Index(df.index))
    if padding is None:
        return
    x_range = padding[0]
    if rows == 1:
        fig.update_xaxes(range=x_range)
        return
    for row in range(1, rows + 1):
        fig.update_xaxes(range=x_range, row=row, col=1)


def add_price_history_main_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    symbol: str,
    params: dict[str, Any],
    row: int | None = None,
    indicator_df: pd.DataFrame | None = None,
) -> None:
    """Add price-history candlesticks plus optional overlays to a figure."""
    source_df = indicator_df if indicator_df is not None else df
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    x = df.index
    source_close = source_df["Close"].astype(float)

    add_price_trace(
        fig,
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=symbol,
        ),
        row,
    )

    if as_bool(params.get("show_bollinger_bands", params.get("bollinger_bands", False))):
        period = positive_int(params.get("bb_period", 20), 20)
        std = float(params.get("bb_std", 2.0))
        bands = ta.bbands(source_close, length=period, std=std)
        lower = bands[f"BBL_{period}_{std}"].reindex(x)
        mid = bands[f"BBM_{period}_{std}"].reindex(x)
        upper = bands[f"BBU_{period}_{std}"].reindex(x)
        add_price_trace(
            fig,
            go.Scatter(x=x, y=lower, name=f"BB lower ({period}, {std:g})", mode="lines", line={"color": "rgba(37, 99, 235, 0.55)", "width": 1}, connectgaps=False),
            row,
        )
        add_price_trace(
            fig,
            go.Scatter(x=x, y=upper, name=f"BB upper ({period}, {std:g})", mode="lines", line={"color": "rgba(37, 99, 235, 0.55)", "width": 1}, fill="tonexty", fillcolor="rgba(37, 99, 235, 0.08)", connectgaps=False),
            row,
        )
        add_price_trace(
            fig,
            go.Scatter(x=x, y=mid, name=f"BB mid ({period})", mode="lines", line={"color": "#2563eb", "width": 1, "dash": "dot"}, connectgaps=False),
            row,
        )

    if as_bool(params.get("show_ma_cloud", params.get("ma_cloud", False))):
        fast = positive_int(params.get("ma_cloud_fast", 50), 50)
        slow = positive_int(params.get("ma_cloud_slow", 200), 200)
        fast_ma = ta.sma(source_close, length=fast).reindex(x)
        slow_ma = ta.sma(source_close, length=slow).reindex(x)
        add_price_trace(fig, go.Scatter(x=x, y=slow_ma, name=f"SMA {slow}", mode="lines", line={"color": "rgba(107, 114, 128, 0.75)", "width": 1.1}, connectgaps=False), row)
        add_price_trace(fig, go.Scatter(x=x, y=fast_ma, name=f"SMA {fast} cloud", mode="lines", line={"color": "rgba(14, 165, 164, 0.85)", "width": 1.1}, fill="tonexty", fillcolor="rgba(14, 165, 164, 0.10)", connectgaps=False), row)

    if as_bool(params.get("show_ma", params.get("ma", False))):
        period = positive_int(params.get("ma_period", 50), 50)
        ma = ta.sma(source_close, length=period).reindex(x)
        add_price_trace(fig, go.Scatter(x=x, y=ma, name=f"SMA {period}", mode="lines", line={"color": "#f59e0b", "width": 1.6}, connectgaps=False), row)

    if as_bool(params.get("show_ema", params.get("ema", False))):
        period = positive_int(params.get("ema_period", 21), 21)
        ema = ta.ema(source_close, length=period).reindex(x)
        add_price_trace(fig, go.Scatter(x=x, y=ema, name=f"EMA {period}", mode="lines", line={"color": "#7c3aed", "width": 1.5}, connectgaps=False), row)

    if as_bool(params.get("show_avwap", params.get("avwap", False))):
        anchor = params.get("avwap_anchor") if params.get("avwap_anchor") else x[0]
        avwap = anchored_vwap(source_df, anchor).reindex(x)
        add_price_trace(fig, go.Scatter(x=x, y=avwap, name="aVWAP", mode="lines", line={"color": "#dc2626", "width": 1.5}, connectgaps=False), row)

    if price_history_has_fib(params):
        add_fib_retracement_traces(fig, x, high, low, row)


def price_history(provider: DataProvider, params: dict[str, Any]) -> dict[str, str]:
    """Create a candlestick chart for one symbol over a requested period."""
    symbol = params.get("symbol", "SPY")
    period = params.get("period", "1y")
    interval = params.get("interval", "1d")
    pe_subchart = as_bool(params.get("pe_subchart", False))
    df, indicator_df = price_history_frames(provider, str(symbol), str(period), str(interval), params)
    if df.empty:
        raise ValueError("No price data")
    if not pe_subchart:
        fig = go.Figure()
        add_price_history_main_traces(fig, df, str(symbol), params, indicator_df=indicator_df)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(
            title=f"{symbol} {period} {interval}",
            xaxis_title="Date",
            yaxis_title="Price",
            margin={"l": 56, "r": 32, "t": 82, "b": 56},
            legend=price_history_legend_layout(),
            hovermode="x unified",
        )
        apply_fib_x_padding(fig, df, params)
        return {"mime": "image/png", "data": fig_to_b64(fig, "price_history")}

    close = df["Close"].astype(float)
    pe = trailing_pe_series(provider, symbol, close)
    pe_source = str(pe.attrs.get("source", "provider"))
    if not np.isfinite(pe.to_numpy(dtype=float)).any():
        raise ValueError(
            "No P/E for this symbol (Alpha Vantage did not return EPS history and Yahoo had no "
            "usable quarterly TTM or annual Diluted/Basic EPS; many ETFs and funds have none)."
        )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.68, 0.32])
    add_price_history_main_traces(fig, df, str(symbol), params, row=1, indicator_df=indicator_df)
    fig.add_trace(go.Scatter(x=df.index, y=pe, name=f"P/E ({pe_source})", mode="lines", line={"color": "#2563eb", "width": 1.4}, connectgaps=False), row=2, col=1)
    fig.update_layout(
        title=f"{symbol} {period} {interval}",
        xaxis_rangeslider_visible=False,
        height=720,
        margin={"l": 56, "r": 32, "t": 82, "b": 40},
        legend=price_history_legend_layout(),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    apply_fib_x_padding(fig, df, params, rows=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text=f"P/E ({pe_source})", row=2, col=1)
    return {"mime": "image/png", "data": fig_to_b64(fig, "price_history")}

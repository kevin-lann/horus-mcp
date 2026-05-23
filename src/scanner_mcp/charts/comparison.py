"""Comparison and overlay chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner_mcp.charts.data import aligned_close_frame, contiguous_true_spans, normalize_to_100
from scanner_mcp.charts.layout import clean_layout
from scanner_mcp.charts.params import positive_int
from scanner_mcp.charts.rendering import fig_to_b64
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.indicators import ta


def price_overlay(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Plot multiple symbols on one line chart, optionally normalized to 100."""
    raw_symbols = params.get("symbols") or ["SPY", "QQQ"]
    if isinstance(raw_symbols, str):
        symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]
    else:
        symbols = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    period = params.get("period", "1y")
    normalize = params.get("normalize", True)
    fig = go.Figure()
    for symbol in symbols:
        df = provider.get_history(symbol, period=str(period), interval="1d")
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        y = close / float(close.iloc[0]) * 100.0 if normalize else close
        fig.add_trace(go.Scatter(x=df.index, y=y, name=symbol, mode="lines"))
    if not fig.data:
        raise ValueError("no data available for requested symbols")
    fig.update_layout(title="Price overlay" + (" (normalized % base=100)" if normalize else ""), xaxis_title="Date", yaxis_title="Y")
    return {"mime": "image/png", "data": fig_to_b64(fig, "price_overlay")}


def ratio_chart(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Plot one asset priced as a ratio of another asset."""
    symbol = str(params.get("symbol", "SPY")).strip().upper()
    benchmark = str(params.get("benchmark", "XLP")).strip().upper()
    period = str(params.get("period", "1y"))
    closes = aligned_close_frame(provider, [symbol, benchmark], period)
    ratio = closes[symbol] / closes[benchmark]
    fig = go.Figure(data=[go.Scatter(x=ratio.index, y=ratio, name=f"{symbol}/{benchmark}", mode="lines", line={"color": "#1d4ed8", "width": 2.2})])
    clean_layout(fig, title=f"Ratio chart: {symbol} vs {benchmark}", yaxis_title=f"{symbol}/{benchmark}")
    return {"mime": "image/png", "data": fig_to_b64(fig, "ratio_chart")}


def relative_strength_chart(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Plot stock-vs-benchmark ratio with a moving average and leadership shading."""
    symbol = str(params.get("symbol", "AAPL")).strip().upper()
    benchmark = str(params.get("benchmark", "SPY")).strip().upper()
    period = str(params.get("period", "2y"))
    ma_period = positive_int(params.get("ma_period", 50), 50)
    closes = aligned_close_frame(provider, [symbol, benchmark], period)
    ratio = (closes[symbol] / closes[benchmark]).rename(f"{symbol}/{benchmark}")
    ratio_ma = ta.sma(ratio, length=ma_period)
    plot_frame = pd.DataFrame({"ratio": ratio, "ma": ratio_ma}).dropna()
    if plot_frame.empty:
        raise ValueError(f"Not enough data to compute a {ma_period}-day relative strength average")

    fig = go.Figure()
    for start, end in contiguous_true_spans(plot_frame["ratio"] >= plot_frame["ma"]):
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(22,163,74,0.10)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=plot_frame.index, y=plot_frame["ratio"], name=f"{symbol}/{benchmark}", mode="lines", line={"color": "#0f172a", "width": 2.2}))
    fig.add_trace(go.Scatter(x=plot_frame.index, y=plot_frame["ma"], name=f"{ma_period}D SMA", mode="lines", line={"color": "#dc2626", "width": 1.8, "dash": "dash"}))
    latest_ratio = float(plot_frame["ratio"].iloc[-1])
    latest_ma = float(plot_frame["ma"].iloc[-1])
    status = "Above MA" if latest_ratio >= latest_ma else "Below MA"
    clean_layout(fig, title=f"Relative strength: {symbol} vs {benchmark} ({status})", yaxis_title=f"{symbol}/{benchmark}")
    return {"mime": "image/png", "data": fig_to_b64(fig, "relative_strength")}


def sector_rotation_chart(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Compare normalized sector ETF performance with rolling returns."""
    symbols = [str(symbol).strip().upper() for symbol in (params.get("symbols") or ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"])]
    period = str(params.get("period", "2y"))
    return_window = positive_int(params.get("return_window", 63), 63)
    closes = aligned_close_frame(provider, symbols, period)
    normalized = closes.apply(normalize_to_100, axis=0)
    rolling_returns = (closes / closes.shift(return_window) - 1.0) * 100.0
    rolling_returns = rolling_returns.dropna(how="all")
    if rolling_returns.empty:
        raise ValueError(f"Not enough data to compute {return_window}-day rolling returns")

    latest_norm = normalized.iloc[-1].sort_values(ascending=False)
    top_symbol = str(latest_norm.index[0])
    bottom_symbol = str(latest_norm.index[-1])
    latest_ret = rolling_returns.iloc[-1].dropna().sort_values(ascending=False)
    if latest_ret.empty:
        raise ValueError("No rolling return data available for requested symbols")

    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4f46e5", "#65a30d", "#be123c", "#6d28d9"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38])
    for idx, symbol in enumerate(symbols):
        color = colors[idx % len(colors)]
        width = 3.0 if symbol == top_symbol else 1.4
        opacity = 1.0 if symbol in {top_symbol, bottom_symbol} else 0.8
        dash = "dot" if symbol == bottom_symbol else "solid"
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized[symbol], name=symbol, mode="lines", line={"color": color, "width": width, "dash": dash}, opacity=opacity), row=1, col=1)
        fig.add_trace(go.Scatter(x=rolling_returns.index, y=rolling_returns[symbol], name=f"{symbol} {return_window}D", mode="lines", line={"color": color, "width": width if symbol == top_symbol else 1.2, "dash": dash}, opacity=opacity, showlegend=False), row=2, col=1)

    top_value = float(latest_norm.iloc[0])
    bottom_value = float(latest_norm.iloc[-1])
    fig.add_trace(go.Scatter(x=[normalized.index[-1], normalized.index[-1]], y=[top_value, bottom_value], mode="markers+text", text=[f"Top: {top_symbol}", f"Bottom: {bottom_symbol}"], textposition=["middle right", "middle right"], marker={"color": ["#15803d", "#b91c1c"], "size": [8, 8], "line": {"color": "#ffffff", "width": 1}}, showlegend=False, hoverinfo="skip"), row=1, col=1)

    fig.update_layout(title=f"Sector rotation / ETF comparison ({return_window}-day return panel)", height=840, margin={"l": 60, "r": 120, "t": 78, "b": 48}, legend={"orientation": "h", "x": 0.01, "xanchor": "left", "y": 1.03, "yanchor": "top"}, hovermode="x unified", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    if len(normalized.index) >= 2:
        x_index = pd.DatetimeIndex(normalized.index)
        span = x_index[-1] - x_index[0]
        diffs = x_index.to_series().diff().dropna()
        step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
        right_pad = max(span * 0.18, step * 20)
        x_range = [x_index[0], x_index[-1] + right_pad]
        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False, row=1, col=1, showticklabels=False)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False, row=2, col=1, title_text="Date")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=False, row=1, col=1, title_text="Normalized (base=100)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=True, zerolinecolor="rgba(15,23,42,0.12)", row=2, col=1, title_text=f"{return_window}D return %")
    return {"mime": "image/png", "data": fig_to_b64(fig, "sector_rotation")}


def drawdown_comparison(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Compare percentage drawdowns from each symbol's running high."""
    raw_symbols = params.get("symbols") or ["^GSPC", "QQQ"]
    if isinstance(raw_symbols, str):
        symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]
    else:
        symbols = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    period = params.get("period", "5y")
    fig = go.Figure()
    for symbol in symbols:
        df = provider.get_history(symbol, period=str(period), interval="1d")
        if df is None or df.empty:
            continue
        close = df["Close"].astype(float)
        run_max = close.cummax()
        drawdown = (close - run_max) / run_max * 100.0
        fig.add_trace(go.Scatter(x=df.index, y=drawdown, name=symbol, mode="lines"))
    if not fig.data:
        raise ValueError("no data available for requested symbols")
    fig.update_layout(title="Drawdown % (from running max)", yaxis_title="Drawdown %")
    return {"mime": "image/png", "data": fig_to_b64(fig, "drawdown_comparison")}


def log_cycle(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Plot weekly log10 close values for long-cycle price inspection."""
    symbol = str(params.get("symbol", "BTC-USD"))
    period = params.get("period", "max")
    df = provider.get_history(symbol, period=str(period), interval="1wk")
    if df.empty:
        raise ValueError("No data for log chart")
    close = df["Close"].astype(float)
    y = np.log10(close.replace(0, np.nan))
    fig = go.Figure(data=[go.Scatter(x=df.index, y=y, name=symbol, mode="lines")])
    fig.update_layout(title=f"{symbol} log10(close) weekly", yaxis_title="log10 price")
    return {"mime": "image/png", "data": fig_to_b64(fig, "log_cycle")}


def basket_breadth_chart(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Compare an equal-weight basket against a benchmark with breadth panels."""
    raw_symbols = params.get("symbols") or ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
    if isinstance(raw_symbols, str):
        symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]
    else:
        symbols = [str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()]
    if len(symbols) < 2:
        raise ValueError("symbols must contain at least two basket members")
    benchmark = str(params.get("benchmark", "QQQ")).strip().upper()
    period = str(params.get("period", "1y"))
    sma_period = positive_int(params.get("sma_period", 50), 50)
    corr_window = positive_int(params.get("corr_window", 63), 63)

    closes = aligned_close_frame(provider, [*symbols, benchmark], period)
    basket_closes = closes[symbols]
    basket_returns = basket_closes.pct_change().fillna(0.0).mean(axis=1)
    benchmark_returns = closes[benchmark].pct_change().fillna(0.0)
    basket_index = ((1.0 + basket_returns).cumprod() * 100.0).rename("Basket")
    benchmark_index = normalize_to_100(closes[benchmark]).rename(benchmark)
    rolling_corr = basket_returns.rolling(corr_window).corr(benchmark_returns).dropna()
    if rolling_corr.empty:
        raise ValueError(f"Not enough data to compute {corr_window}-day rolling correlation")

    members_above = (basket_closes >= basket_closes.rolling(sma_period).mean()).sum(axis=1).dropna()
    if members_above.empty:
        raise ValueError(f"Not enough data to compute breadth above {sma_period}-day SMA")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.44, 0.24, 0.32],
    )
    fig.add_trace(go.Scatter(x=basket_index.index, y=basket_index.values, name="Equal-weight basket", mode="lines", line={"color": "#111827", "width": 2.2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_index.index, y=benchmark_index.values, name=benchmark, mode="lines", line={"color": "#2563eb", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr.values, name=f"{corr_window}D correlation", mode="lines", line={"color": "#7c3aed", "width": 1.8}), row=2, col=1)
    fig.add_hline(y=0.0, line={"color": "rgba(15,23,42,0.2)", "dash": "dot"}, row=2, col=1)
    fig.add_trace(go.Bar(x=members_above.index, y=members_above.values, name=f"Members above {sma_period}D SMA", marker={"color": "#059669"}), row=3, col=1)
    fig.add_hline(y=len(symbols) / 2.0, line={"color": "rgba(5,150,105,0.25)", "dash": "dash"}, row=3, col=1)

    fig.update_layout(
        title=f"Basket breadth vs {benchmark}",
        height=920,
        margin={"l": 60, "r": 52, "t": 78, "b": 46},
        legend={"orientation": "h", "x": 0.01, "xanchor": "left", "y": 1.03, "yanchor": "top"},
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.08,
    )
    for row in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=True, zerolinecolor="rgba(15,23,42,0.12)", row=row, col=1)
    fig.update_yaxes(title_text="Normalized (base=100)", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Breadth Count", row=3, col=1, range=[0, len(symbols) + 0.5])
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return {"mime": "image/png", "data": fig_to_b64(fig, "basket_breadth")}


def pairs_spread_chart(provider: DataProvider, params: dict[str, object]) -> dict[str, str]:
    """Show normalized pair prices, ratio/spread, and a z-score panel."""
    symbol = str(params.get("symbol", "KO")).strip().upper()
    benchmark = str(params.get("benchmark", "PEP")).strip().upper()
    period = str(params.get("period", "1y"))
    spread_mode = str(params.get("spread_mode", "ratio")).strip().lower()
    z_window = positive_int(params.get("z_window", 63), 63)
    if spread_mode not in {"ratio", "price_spread"}:
        raise ValueError("spread_mode must be ratio or price_spread")

    closes = aligned_close_frame(provider, [symbol, benchmark], period)
    lhs = closes[symbol]
    rhs = closes[benchmark]
    normalized_lhs = normalize_to_100(lhs)
    normalized_rhs = normalize_to_100(rhs)
    spread = (lhs / rhs) if spread_mode == "ratio" else (lhs - rhs)
    spread_name = f"{symbol}/{benchmark}" if spread_mode == "ratio" else f"{symbol}-{benchmark}"
    rolling_mean = spread.rolling(z_window).mean()
    rolling_std = spread.rolling(z_window).std(ddof=0).replace(0.0, np.nan)
    zscore = ((spread - rolling_mean) / rolling_std).dropna()
    if zscore.empty:
        raise ValueError(f"Not enough data to compute {z_window}-day z-score")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.42, 0.26, 0.32],
    )
    fig.add_trace(go.Scatter(x=normalized_lhs.index, y=normalized_lhs.values, name=symbol, mode="lines", line={"color": "#111827", "width": 2.0}), row=1, col=1)
    fig.add_trace(go.Scatter(x=normalized_rhs.index, y=normalized_rhs.values, name=benchmark, mode="lines", line={"color": "#2563eb", "width": 1.8}), row=1, col=1)
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name=spread_name, mode="lines", line={"color": "#b45309", "width": 1.8}), row=2, col=1)
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore.values, name=f"{z_window}D z-score", mode="lines", line={"color": "#7c3aed", "width": 1.8}), row=3, col=1)
    for level, color in ((2.0, "#b91c1c"), (0.0, "#475569"), (-2.0, "#15803d")):
        fig.add_hline(y=level, line={"color": color, "dash": "dash"}, row=3, col=1)

    fig.update_layout(
        title=f"Pairs spread: {symbol} vs {benchmark}",
        height=900,
        margin={"l": 60, "r": 48, "t": 78, "b": 46},
        legend={"orientation": "h", "x": 0.01, "xanchor": "left", "y": 1.03, "yanchor": "top"},
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    for row in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=True, zerolinecolor="rgba(15,23,42,0.12)", row=row, col=1)
    fig.update_yaxes(title_text="Normalized (base=100)", row=1, col=1)
    fig.update_yaxes(title_text=spread_name, row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return {"mime": "image/png", "data": fig_to_b64(fig, "pairs_spread")}

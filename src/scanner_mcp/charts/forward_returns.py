"""Forward-return chart/table construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner_mcp.charts.rendering import fig_to_b64
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.research.forward_returns import DEFAULT_FORWARD_WINDOWS, compute_event_forward_study, summarize_forward_study
from scanner_mcp.signals.catalog import CATALOG, merge_params


def format_number(value: float) -> str:
    """Format a numeric parameter without a trailing .0 when integral."""
    return f"{value:g}"


def window_label(window: int) -> str:
    """Return compact table/legend labels for trading-bar horizons."""
    month_map = {5: "1 Week", 10: "2 Weeks", 21: "1 Month", 42: "2 Months", 63: "3 Months", 84: "4 Months", 105: "5 Months", 126: "6 Months", 252: "12 Months"}
    return month_map.get(int(window), f"{int(window)}d")


def mix_rgb(start: tuple[int, int, int], end: tuple[int, int, int], weight: float) -> str:
    """Linearly mix two RGB colors and return a CSS rgb string."""
    clamped = max(0.0, min(1.0, weight))
    rgb = tuple(round(a + (b - a) * clamped) for a, b in zip(start, end, strict=True))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def forward_return_cell_color(row_name: str, value: float | None, scale_values: list[float]) -> str:
    """Return a neutral, green, or red table-cell color for one summary metric."""
    if value is None:
        return "#ffffff"
    if row_name == "% Positive":
        signed = value - 50.0
        max_abs = 50.0
    else:
        signed = value
        max_abs = max((abs(v) for v in scale_values), default=0.0)
    if max_abs <= 0 or signed == 0:
        return "#ffffff"
    intensity = min(1.0, abs(signed) / max_abs)
    target = (200, 238, 211) if signed > 0 else (249, 204, 204)
    return mix_rgb((255, 255, 255), target, 0.25 + 0.65 * intensity)


def forward_return_table_fill_colors(
    table_rows: list[str],
    table_numeric: dict[str, list[float | None]],
    window_count: int,
) -> list[list[str]]:
    """Build Plotly table fill colors as columns, with result cells heatmapped."""
    colors: list[list[str]] = [["#ffffff" for _ in table_rows]]
    for col_idx in range(window_count):
        col_colors: list[str] = []
        for row_name in table_rows:
            vals = table_numeric[row_name]
            value = vals[col_idx] if col_idx < len(vals) else None
            scale_values = [v for v in vals if v is not None]
            col_colors.append(forward_return_cell_color(row_name, value, scale_values))
        colors.append(col_colors)
    return colors


def forward_event_title(event_type: str, params: dict[str, Any] | None) -> str:
    """Render a human-readable event description with resolved detector params."""
    resolved = merge_params(event_type, params) if event_type in CATALOG else params or {}
    if event_type == "pct_from_ma":
        return f"price moves within {format_number(float(resolved.get('pct', 2.0)))}% of {int(resolved.get('ma_period', 50))}-day {str(resolved.get('ma_type', 'sma')).upper()}"
    if event_type == "rsi_oversold":
        return f"RSI Oversold ({int(resolved.get('period', 14))}-day RSI crosses below {format_number(float(resolved.get('threshold', 30.0)))})"
    if event_type == "rsi_overbought":
        return f"RSI Overbought ({int(resolved.get('period', 14))}-day RSI crosses above {format_number(float(resolved.get('threshold', 70.0)))})"
    if event_type == "golden_cross":
        return f"Golden Cross ({int(resolved.get('fast', 50))}-day SMA crosses above {int(resolved.get('slow', 200))}-day SMA)"
    if event_type == "macd_bullish_crossover":
        return f"MACD Bullish Crossover ({int(resolved.get('fast', 12))}/{int(resolved.get('slow', 26))} MACD crosses above {int(resolved.get('signal', 9))}-day signal)"
    return event_type.replace("_", " ")


def forward_marker_window(study: Any) -> int:
    """Choose the largest forward window that has usable event results."""
    for window in sorted(study.windows, reverse=True):
        for event in study.events:
            result = event.windows.get(window)
            if result is not None and np.isfinite(result.final_return):
                return int(window)
    raise ValueError("No events or no forward returns; try a different symbol/event_type")


def forward_returns_chart(provider: DataProvider, params: dict[str, Any]) -> dict[str, str]:
    """Build a price/event chart plus forward-return summary table."""
    symbol = str(params.get("symbol", "SPY"))
    event_type = str(params.get("event_type", "rsi_oversold"))
    windows = [int(value) for value in (params.get("windows") or DEFAULT_FORWARD_WINDOWS)]
    event_params = params.get("event_params")
    if event_params is not None and not isinstance(event_params, dict):
        raise ValueError("event_params must be an object")
    event_title = forward_event_title(event_type, event_params)
    study = compute_event_forward_study(provider, symbol, event_type, windows, period="10y", params=event_params)
    summary = summarize_forward_study(study)
    if study.price.empty or not study.events or not any(summary.get(window, {}).get("n") for window in study.windows):
        raise ValueError("No events or no forward returns; try a different symbol/event_type")

    marker_window = forward_marker_window(study)
    marker_dates: dict[str, list[Any]] = {"positive": [], "negative": [], "neutral": []}
    marker_prices: dict[str, list[float]] = {"positive": [], "negative": [], "neutral": []}
    marker_text: dict[str, list[str]] = {"positive": [], "negative": [], "neutral": []}
    guide_x: dict[str, list[Any]] = {"positive": [], "negative": [], "neutral": []}
    guide_y: dict[str, list[float | None]] = {"positive": [], "negative": [], "neutral": []}
    price_min = float(study.price.min())
    for event in study.events:
        result = event.windows.get(marker_window)
        key = "neutral"
        ret_label = "n/a"
        if result is not None:
            final_return = result.final_return
            if math.isfinite(final_return):
                ret_label = f"{final_return:.1f}%"
                if final_return > 0:
                    key = "positive"
                elif final_return < 0:
                    key = "negative"
        marker_dates[key].append(event.date)
        marker_prices[key].append(event.price)
        marker_text[key].append(f"{event.label}<br>{marker_window}d return: {ret_label}")
        guide_x[key].extend([event.date, event.date, None])
        guide_y[key].extend([price_min, event.price, None])

    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "xy"}], [{"type": "table"}]], row_heights=[0.66, 0.34], vertical_spacing=0.035)
    fig.add_trace(go.Scatter(x=study.price.index, y=study.price.values, name=symbol, mode="lines", line={"color": "#171717", "width": 1.8}), row=1, col=1)
    marker_style = {"positive": ("Signal Positive", "#3ca454", "triangle-up"), "negative": ("Signal Negative", "#bd3a30", "triangle-up"), "neutral": ("Signal", "#777777", "triangle-up")}
    for key, (name, color, symbol_name) in marker_style.items():
        if not marker_dates[key]:
            continue
        fig.add_trace(go.Scatter(x=guide_x[key], y=guide_y[key], name=f"{name} guide", mode="lines", line={"color": color, "width": 1}, opacity=0.35, hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=marker_dates[key], y=marker_prices[key], name=f"{name} After {window_label(marker_window)}", mode="markers", marker={"color": color, "size": 11, "symbol": symbol_name}, text=marker_text[key], hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>"), row=1, col=1)

    table_header = [f"Dates of<br>{len(study.events)} Signals", *[window_label(window) + "<br>Later (%)" for window in study.windows]]
    table_rows = ["Mean", "Median", "% Positive", "Avg Max Loss", "Avg Max Gain"]
    table_values: list[list[str]] = [[] for _ in range(len(study.windows) + 1)]
    table_numeric: dict[str, list[float | None]] = {row: [] for row in table_rows}
    for row_name in table_rows:
        table_values[0].append(f"<b>{row_name}</b>")
    for idx, window in enumerate(study.windows, start=1):
        row = summary.get(window, {"n": 0})
        if not row.get("n"):
            table_values[idx].extend(["-", "-", "-", "-", "-"])
            for row_name in table_rows:
                table_numeric[row_name].append(None)
            continue
        values = {"Mean": float(row["mean"]), "Median": float(row["median"]), "% Positive": float(row["positive_pct"]), "Avg Max Loss": float(row["avg_max_loss"]), "Avg Max Gain": float(row["avg_max_gain"])}
        table_values[idx].extend([f"{values['Mean']:.1f}", f"{values['Median']:.1f}", f"{values['% Positive']:.0f}%", f"{values['Avg Max Loss']:.1f}", f"{values['Avg Max Gain']:.1f}"])
        for row_name in table_rows:
            table_numeric[row_name].append(values[row_name])

    fill_colors = forward_return_table_fill_colors(table_rows, table_numeric, len(study.windows))
    fig.add_trace(go.Table(header={"values": [f"<b>{value}</b>" for value in table_header], "align": "center", "font": {"size": 15, "color": "#111111"}, "fill_color": "#ffffff", "line_color": "#d0d0d0", "height": 34}, cells={"values": table_values, "align": "center", "font": {"size": 14, "color": "#111111"}, "fill_color": fill_colors, "line_color": "#eeeeee", "height": 27}), row=2, col=1)
    fig.update_layout(title={"text": f"{symbol} forward returns after {event_title}", "x": 0.02, "xanchor": "left"}, width=1150, height=760, margin={"l": 42, "r": 42, "t": 68, "b": 26}, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", legend={"orientation": "h", "x": 0.02, "y": 1.05})
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, title_text="Price", row=1, col=1)
    return {"mime": "image/png", "data": fig_to_b64(fig, "forward_returns")}

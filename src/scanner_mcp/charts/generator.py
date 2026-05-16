"""Build Plotly charts for MCP responses and optional local debug PNGs."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
import io
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scanner_mcp.data.provider import YFinanceProvider
from scanner_mcp.indicators import ta
from scanner_mcp.signals.catalog import CATALOG, merge_params

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_DIR = _PROJECT_ROOT / "output"


def _save_debug_png(chart_type: str, png_bytes: bytes) -> None:
    """Persist a generated PNG under project-root/output for local debugging."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{chart_type}_{timestamp}_{uuid4().hex[:8]}.png"
    (_OUTPUT_DIR / filename).write_bytes(png_bytes)


def _fig_to_b64(fig: go.Figure, chart_type: str) -> str:
    """Render a Plotly figure to PNG bytes, save a debug copy, and return base64."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", engine="kaleido", scale=1.5)
    buf.seek(0)
    png_bytes = buf.getvalue()
    try:
        _save_debug_png(chart_type, png_bytes)
    except OSError:
        log.exception("Failed to save debug chart image")
    return base64.b64encode(png_bytes).decode("ascii")


def generate_chart(
    provider: YFinanceProvider,
    chart_type: str,
    params: dict[str, Any],
) -> dict[str, str]:
    """Dispatch a supported chart type and return an MCP-safe image payload.

    The returned dictionary is JSON serializable and contains a PNG MIME type plus
    base64 image data. Rendering errors and unknown chart types are intentionally
    raised for the MCP tool wrapper to convert into an error response.
    """
    ct = chart_type.lower().strip()
    if ct == "price_history":
        return _price_history(provider, params)
    if ct == "price_overlay":
        return _price_overlay(provider, params)
    if ct == "forward_returns":
        return _forward_returns_chart(provider, params)
    if ct == "drawdown_comparison":
        return _drawdown_comparison(provider, params)
    if ct == "log_cycle":
        return _log_cycle(provider, params)
    raise ValueError(f"Unknown chart_type: {chart_type}")


def _to_naive_asof_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and normalize to midnight for merge_asof on daily fundamentals."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def _merge_asof_price_over_eps(
    close: pd.Series,
    anchor_dates: pd.DatetimeIndex,
    eps_values: np.ndarray,
) -> pd.Series:
    """For each bar date, use the latest EPS row with anchor date <= bar (naive calendar day)."""
    df_sorted = close.sort_index().astype(float)
    hist = pd.DataFrame(
        {
            "asof": _to_naive_asof_dates(pd.DatetimeIndex(df_sorted.index)),
            "close": df_sorted.values,
        }
    )
    right = pd.DataFrame(
        {
            "anchor": _to_naive_asof_dates(pd.DatetimeIndex(anchor_dates)),
            "eps": eps_values.astype(float),
        }
    ).sort_values("anchor")
    merged = pd.merge_asof(hist, right, left_on="asof", right_on="anchor", direction="backward")
    denom = merged["eps"].to_numpy(dtype=float)
    num = merged["close"].to_numpy(dtype=float)
    pe_vals = np.where(np.isfinite(denom) & (denom > 0), num / denom, np.nan)
    pe_sorted = pd.Series(pe_vals, index=df_sorted.index, dtype=float)
    return pe_sorted.reindex(close.index)


def _quarterly_ttm_pe_series(symbol: str, close: pd.Series) -> pd.Series:
    """P/E from rolling 4-quarter Diluted (or Basic) EPS vs close (Yahoo caps at ~5 quarters)."""
    import yfinance as yf

    sym = str(symbol).strip().upper()
    stmt = yf.Ticker(sym).quarterly_incomestmt
    if stmt is None or stmt.empty:
        return pd.Series(np.nan, index=close.index, dtype=float)
    if "Diluted EPS" in stmt.index:
        qeps = stmt.loc["Diluted EPS"]
    elif "Basic EPS" in stmt.index:
        qeps = stmt.loc["Basic EPS"]
    else:
        return pd.Series(np.nan, index=close.index, dtype=float)
    qeps = qeps.dropna().sort_index().astype(float)
    if len(qeps) < 4:
        return pd.Series(np.nan, index=close.index, dtype=float)
    ttm = qeps.rolling(window=4, min_periods=4).sum()
    ttm = ttm.dropna()
    if ttm.empty:
        return pd.Series(np.nan, index=close.index, dtype=float)
    return _merge_asof_price_over_eps(close, pd.DatetimeIndex(ttm.index), ttm.values)


def _annual_fy_eps_pe_series(symbol: str, close: pd.Series) -> pd.Series:
    """P/E vs fiscal-year Diluted (or Basic) EPS from annual statements (longer history than quarterly cap)."""
    import yfinance as yf

    sym = str(symbol).strip().upper()
    stmt = yf.Ticker(sym).incomestmt
    if stmt is None or stmt.empty:
        return pd.Series(np.nan, index=close.index, dtype=float)
    if "Diluted EPS" in stmt.index:
        fy = stmt.loc["Diluted EPS"]
    elif "Basic EPS" in stmt.index:
        fy = stmt.loc["Basic EPS"]
    else:
        return pd.Series(np.nan, index=close.index, dtype=float)
    fy = fy.dropna().sort_index().astype(float)
    if fy.empty:
        return pd.Series(np.nan, index=close.index, dtype=float)
    return _merge_asof_price_over_eps(close, pd.DatetimeIndex(fy.index), fy.values)


def _trailing_pe_series(symbol: str, close: pd.Series) -> pd.Series:
    """P/E aligned to `close`: prefer quarterly TTM where Yahoo provides enough quarters; else FY EPS.

    Yahoo fundamentals cap quarterly columns (~5), so true TTM only exists for the last ~1–2 years.
    Annual Diluted EPS keyed on fiscal year-end fills earlier dates (FY P/E, not TTM).
    """
    pe_q = _quarterly_ttm_pe_series(symbol, close)
    pe_a = _annual_fy_eps_pe_series(symbol, close)
    return pe_q.combine_first(pe_a)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like MCP/JSON values without treating every non-empty string as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out > 0 else default


def _add_price_trace(fig: go.Figure, trace: Any, row: int | None = None) -> None:
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=1)


def _add_price_history_main_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    symbol: str,
    p: dict[str, Any],
    row: int | None = None,
) -> None:
    """Add price-history candlesticks plus optional overlays to a figure."""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    x = df.index

    _add_price_trace(
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

    if _as_bool(p.get("show_bollinger_bands", p.get("bollinger_bands", False))):
        period = _positive_int(p.get("bb_period", 20), 20)
        std = float(p.get("bb_std", 2.0))
        bands = ta.bbands(close, length=period, std=std)
        lower = bands[f"BBL_{period}_{std}"]
        mid = bands[f"BBM_{period}_{std}"]
        upper = bands[f"BBU_{period}_{std}"]
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=lower,
                name=f"BB lower ({period}, {std:g})",
                mode="lines",
                line={"color": "rgba(37, 99, 235, 0.55)", "width": 1},
                connectgaps=False,
            ),
            row,
        )
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=upper,
                name=f"BB upper ({period}, {std:g})",
                mode="lines",
                line={"color": "rgba(37, 99, 235, 0.55)", "width": 1},
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.08)",
                connectgaps=False,
            ),
            row,
        )
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=mid,
                name=f"BB mid ({period})",
                mode="lines",
                line={"color": "#2563eb", "width": 1, "dash": "dot"},
                connectgaps=False,
            ),
            row,
        )

    if _as_bool(p.get("show_ma_cloud", p.get("ma_cloud", False))):
        fast = _positive_int(p.get("ma_cloud_fast", 50), 50)
        slow = _positive_int(p.get("ma_cloud_slow", 200), 200)
        fast_ma = ta.sma(close, length=fast)
        slow_ma = ta.sma(close, length=slow)
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=slow_ma,
                name=f"SMA {slow}",
                mode="lines",
                line={"color": "rgba(107, 114, 128, 0.75)", "width": 1.1},
                connectgaps=False,
            ),
            row,
        )
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=fast_ma,
                name=f"SMA {fast} cloud",
                mode="lines",
                line={"color": "rgba(14, 165, 164, 0.85)", "width": 1.1},
                fill="tonexty",
                fillcolor="rgba(14, 165, 164, 0.10)",
                connectgaps=False,
            ),
            row,
        )

    if _as_bool(p.get("show_ma", p.get("ma", False))):
        period = _positive_int(p.get("ma_period", 50), 50)
        ma = ta.sma(close, length=period)
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=ma,
                name=f"SMA {period}",
                mode="lines",
                line={"color": "#f59e0b", "width": 1.6},
                connectgaps=False,
            ),
            row,
        )

    if _as_bool(p.get("show_ema", p.get("ema", False))):
        period = _positive_int(p.get("ema_period", 21), 21)
        ema = ta.ema(close, length=period)
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=ema,
                name=f"EMA {period}",
                mode="lines",
                line={"color": "#7c3aed", "width": 1.5},
                connectgaps=False,
            ),
            row,
        )

    if _as_bool(p.get("show_avwap", p.get("avwap", False))):
        avwap = _anchored_vwap(df, p.get("avwap_anchor"))
        _add_price_trace(
            fig,
            go.Scatter(
                x=x,
                y=avwap,
                name="aVWAP",
                mode="lines",
                line={"color": "#dc2626", "width": 1.5},
                connectgaps=False,
            ),
            row,
        )

    if _price_history_has_fib(p):
        _add_fib_retracement_traces(fig, x, high, low, row)


def _anchored_vwap(df: pd.DataFrame, anchor: Any = None) -> pd.Series:
    """Return VWAP anchored to the first visible bar, or to `anchor` when supplied."""
    idx = pd.DatetimeIndex(df.index)
    typical = (df["High"].astype(float) + df["Low"].astype(float) + df["Close"].astype(float)) / 3.0
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    typical = typical.where(volume > 0)
    if anchor:
        anchor_ts = pd.Timestamp(anchor)
        if idx.tz is not None:
            anchor_ts = (
                anchor_ts.tz_localize(idx.tz)
                if anchor_ts.tzinfo is None
                else anchor_ts.tz_convert(idx.tz)
            )
        elif anchor_ts.tzinfo is not None:
            anchor_ts = anchor_ts.tz_convert(None)
        mask = idx >= anchor_ts
        typical = typical.where(mask)
        volume = volume.where(mask)
    cum_volume = volume.cumsum()
    return (typical * volume).cumsum() / cum_volume.replace(0, np.nan)


def _add_fib_retracement_traces(
    fig: go.Figure,
    x: pd.Index,
    high: pd.Series,
    low: pd.Series,
    row: int | None = None,
) -> None:
    low_idx = low.idxmin()
    high_idx = high.idxmax()
    swing_low = float(low.loc[low_idx])
    swing_high = float(high.loc[high_idx])
    span = swing_high - swing_low
    if not np.isfinite(span) or span <= 0:
        return
    uptrend = low_idx <= high_idx
    label_x = _fib_label_x(pd.Index(x))
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    for ratio in ratios:
        level = swing_high - span * ratio if uptrend else swing_low + span * ratio
        label = f"{ratio:g} {level:.2f}"
        _add_price_trace(
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
            y=level+2,
            yref="y" if row in (None, 1) else f"y{row}",
            text=label,
            showarrow=False,
            font={"color": "#4b5563", "size": 8},
            align="right",
            bgcolor="rgba(255,255,255,0.0)",
        )


def _price_history_legend_layout() -> dict[str, Any]:
    """Place price-history legends above the plot area, not over the data."""
    return {
        "orientation": "h",
        "x": 0.02,
        "xanchor": "left",
        "y": 1.1,
        "yanchor": "top",
        "bgcolor": "rgba(255,255,255,0.85)",
    }


def _price_history_has_fib(p: dict[str, Any]) -> bool:
    return _as_bool(p.get("show_fib_retracement", p.get("fib_retracement", False)))


def _fib_x_padding(index: pd.Index) -> tuple[list[Any], Any] | None:
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


def _fib_label_x(index: pd.Index) -> Any:
    padding = _fib_x_padding(index)
    if padding is None:
        return index[-1]
    return padding[1]


def _apply_fib_x_padding(fig: go.Figure, df: pd.DataFrame, p: dict[str, Any], rows: int = 1) -> None:
    if not _price_history_has_fib(p):
        return
    padding = _fib_x_padding(pd.Index(df.index))
    if padding is None:
        return
    x_range = padding[0]
    if rows == 1:
        fig.update_xaxes(range=x_range)
        return
    for row in range(1, rows + 1):
        fig.update_xaxes(range=x_range, row=row, col=1)


def _price_history(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Create a candlestick chart for one symbol over a requested period."""
    sym = p.get("symbol", "SPY")
    period = p.get("period", "1y")
    interval = p.get("interval", "1d")
    pe_subchart = _as_bool(p.get("pe_subchart", False))
    df = provider.get_history(str(sym), period=str(period), interval=str(interval))
    if df.empty:
        raise ValueError("No price data")
    if not pe_subchart:
        fig = go.Figure()
        _add_price_history_main_traces(fig, df, str(sym), p)
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(
            title=f"{sym} {period} {interval}",
            xaxis_title="Date",
            yaxis_title="Price",
            margin={"l": 56, "r": 32, "t": 82, "b": 56},
            legend=_price_history_legend_layout(),
            hovermode="x unified",
        )
        _apply_fib_x_padding(fig, df, p)
        return {"mime": "image/png", "data": _fig_to_b64(fig, "price_history")}

    close = df["Close"].astype(float)
    pe = _trailing_pe_series(sym, close)
    if not np.isfinite(pe.to_numpy(dtype=float)).any():
        raise ValueError(
            "No P/E for this symbol (no usable quarterly TTM or annual Diluted/Basic EPS in "
            "yfinance; many ETFs and funds have none)."
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
    )
    _add_price_history_main_traces(fig, df, str(sym), p, row=1)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=pe,
            name="P/E (TTM or FY EPS)",
            mode="lines",
            line={"color": "#2563eb", "width": 1.4},
            connectgaps=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=f"{sym} {period} {interval}",
        xaxis_rangeslider_visible=False,
        height=720,
        margin={"l": 56, "r": 32, "t": 82, "b": 40},
        legend=_price_history_legend_layout(),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    _apply_fib_x_padding(fig, df, p, rows=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="P/E (TTM or FY EPS)", row=2, col=1)
    return {"mime": "image/png", "data": _fig_to_b64(fig, "price_history")}


def _price_overlay(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Plot multiple symbols on one line chart, optionally normalized to 100."""
    syms: list = p.get("symbols") or ["SPY", "QQQ"]
    period = p.get("period", "1y")
    norm = p.get("normalize", True)
    fig = go.Figure()
    for s in syms:
        df = provider.get_history(str(s), period=str(period), interval="1d")
        if df.empty:
            continue
        c = df["Close"].astype(float)
        y = c / float(c.iloc[0]) * 100.0 if norm else c
        fig.add_trace(go.Scatter(x=df.index, y=y, name=str(s), mode="lines"))
    fig.update_layout(
        title="Price overlay" + (" (normalized % base=100)" if norm else ""),
        xaxis_title="Date",
        yaxis_title="Y",
    )
    return {"mime": "image/png", "data": _fig_to_b64(fig, "price_overlay")}


def _forward_returns_chart(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Build a price/event chart plus forward-return summary table."""
    from scanner_mcp.research.forward_returns import (  # local import
        DEFAULT_FORWARD_WINDOWS,
        compute_event_forward_study,
        summarize_forward_study,
    )

    sym = str(p.get("symbol", "SPY"))
    event = str(p.get("event_type", "rsi_oversold"))
    windows = [int(x) for x in (p.get("windows") or DEFAULT_FORWARD_WINDOWS)]
    event_params_raw = p.get("event_params")
    if event_params_raw is not None and not isinstance(event_params_raw, dict):
        raise ValueError("event_params must be an object")
    event_title = _forward_event_title(event, event_params_raw)
    study = compute_event_forward_study(
        provider,
        sym,
        event,
        windows,
        period="10y",
        params=event_params_raw,
    )
    summary = summarize_forward_study(study)
    if study.price.empty or not study.events or not any(summary.get(w, {}).get("n") for w in study.windows):
        raise ValueError("No events or no forward returns; try a different symbol/event_type")

    marker_window = _forward_marker_window(study)
    marker_dates: dict[str, list[Any]] = {"positive": [], "negative": [], "neutral": []}
    marker_prices: dict[str, list[float]] = {"positive": [], "negative": [], "neutral": []}
    marker_text: dict[str, list[str]] = {"positive": [], "negative": [], "neutral": []}
    guide_x: dict[str, list[Any]] = {"positive": [], "negative": [], "neutral": []}
    guide_y: dict[str, list[float | None]] = {"positive": [], "negative": [], "neutral": []}
    price_min = float(study.price.min())

    for ev in study.events:
        result = ev.windows.get(marker_window)
        key = "neutral"
        ret_label = "n/a"
        if result is not None:
            key = "positive" if result.final_return > 0 else "negative"
            ret_label = f"{result.final_return:.1f}%"
        marker_dates[key].append(ev.date)
        marker_prices[key].append(ev.price)
        marker_text[key].append(f"{ev.label}<br>{marker_window}d return: {ret_label}")
        guide_x[key].extend([ev.date, ev.date, None])
        guide_y[key].extend([price_min, ev.price, None])

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        row_heights=[0.66, 0.34],
        vertical_spacing=0.035,
    )
    fig.add_trace(
        go.Scatter(
            x=study.price.index,
            y=study.price.values,
            name=sym,
            mode="lines",
            line={"color": "#171717", "width": 1.8},
        ),
        row=1,
        col=1,
    )
    marker_style = {
        "positive": ("Signal Positive", "#3ca454", "triangle-up"),
        "negative": ("Signal Negative", "#bd3a30", "triangle-up"),
        "neutral": ("Signal", "#777777", "triangle-up"),
    }
    for key, (name, color, symbol) in marker_style.items():
        if not marker_dates[key]:
            continue
        fig.add_trace(
            go.Scatter(
                x=guide_x[key],
                y=guide_y[key],
                name=f"{name} guide",
                mode="lines",
                line={"color": color, "width": 1},
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=marker_dates[key],
                y=marker_prices[key],
                name=f"{name} After {_window_label(marker_window)}",
                mode="markers",
                marker={"color": color, "size": 11, "symbol": symbol},
                text=marker_text[key],
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    table_header = [
        f"Dates of<br>{len(study.events)} Signals",
        *[_window_label(w) + "<br>Later (%)" for w in study.windows],
    ]
    table_rows = ["Mean", "Median", "% Positive", "Avg Max Loss", "Avg Max Gain"]
    table_values: list[list[str]] = [[] for _ in range(len(study.windows) + 1)]
    table_numeric: dict[str, list[float | None]] = {row: [] for row in table_rows}
    for row_name in table_rows:
        table_values[0].append(f"<b>{row_name}</b>")
    for i, w in enumerate(study.windows, start=1):
        row = summary.get(w, {"n": 0})
        if not row.get("n"):
            table_values[i].extend(["-", "-", "-", "-", "-"])
            for row_name in table_rows:
                table_numeric[row_name].append(None)
            continue
        values = {
            "Mean": float(row["mean"]),
            "Median": float(row["median"]),
            "% Positive": float(row["positive_pct"]),
            "Avg Max Loss": float(row["avg_max_loss"]),
            "Avg Max Gain": float(row["avg_max_gain"]),
        }
        table_values[i].extend(
            [
                f"{values['Mean']:.1f}",
                f"{values['Median']:.1f}",
                f"{values['% Positive']:.0f}%",
                f"{values['Avg Max Loss']:.1f}",
                f"{values['Avg Max Gain']:.1f}",
            ]
        )
        for row_name in table_rows:
            table_numeric[row_name].append(values[row_name])

    fill_colors = _forward_return_table_fill_colors(table_rows, table_numeric, len(study.windows))

    fig.add_trace(
        go.Table(
            header={
                "values": [f"<b>{x}</b>" for x in table_header],
                "align": "center",
                "font": {"size": 15, "color": "#111111"},
                "fill_color": "#ffffff",
                "line_color": "#d0d0d0",
                "height": 34,
            },
            cells={
                "values": table_values,
                "align": "center",
                "font": {"size": 14, "color": "#111111"},
                "fill_color": fill_colors,
                "line_color": "#eeeeee",
                "height": 27,
            },
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": f"{sym} forward returns after {event_title}",
            "x": 0.02,
            "xanchor": "left",
        },
        width=1150,
        height=760,
        margin={"l": 42, "r": 42, "t": 68, "b": 26},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "x": 0.02, "y": 1.05},
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, title_text="Price", row=1, col=1)
    return {"mime": "image/png", "data": _fig_to_b64(fig, "forward_returns")}


def _forward_event_title(event_type: str, params: dict[str, Any] | None) -> str:
    """Render a human-readable event description with resolved detector params."""
    if event_type in CATALOG:
        p = merge_params(event_type, params)
    else:
        p = params or {}

    if event_type == "pct_from_ma":
        ma_period = int(p.get("ma_period", 50))
        ma_type = str(p.get("ma_type", "sma")).upper()
        pct = float(p.get("pct", 2.0))
        return f"price moves within {_format_number(pct)}% of {ma_period}-day {ma_type}"
    if event_type == "rsi_oversold":
        period = int(p.get("period", 14))
        threshold = float(p.get("threshold", 30.0))
        return f"RSI Oversold ({period}-day RSI crosses below {_format_number(threshold)})"
    if event_type == "rsi_overbought":
        period = int(p.get("period", 14))
        threshold = float(p.get("threshold", 70.0))
        return f"RSI Overbought ({period}-day RSI crosses above {_format_number(threshold)})"
    if event_type == "golden_cross":
        fast = int(p.get("fast", 50))
        slow = int(p.get("slow", 200))
        return f"Golden Cross ({fast}-day SMA crosses above {slow}-day SMA)"
    if event_type == "macd_bullish_crossover":
        fast = int(p.get("fast", 12))
        slow = int(p.get("slow", 26))
        signal = int(p.get("signal", 9))
        return f"MACD Bullish Crossover ({fast}/{slow} MACD crosses above {signal}-day signal)"
    return event_type.replace("_", " ")


def _forward_marker_window(study: Any) -> int:
    """Choose the largest forward window that has usable event results."""
    for window in sorted(study.windows, reverse=True):
        for ev in study.events:
            result = ev.windows.get(window)
            if result is not None and np.isfinite(result.final_return):
                return int(window)
    raise ValueError("No events or no forward returns; try a different symbol/event_type")


def _format_number(value: float) -> str:
    """Format a numeric parameter without a trailing .0 when it is integral."""
    return f"{value:g}"


def _window_label(window: int) -> str:
    """Return compact table/legend labels for trading-bar horizons."""
    month_map = {
        5: "1 Week",
        10: "2 Weeks",
        21: "1 Month",
        42: "2 Months",
        63: "3 Months",
        84: "4 Months",
        105: "5 Months",
        126: "6 Months",
        252: "12 Months",
    }
    return month_map.get(int(window), f"{int(window)}d")


def _forward_return_table_fill_colors(
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
            col_colors.append(_forward_return_cell_color(row_name, value, scale_values))
        colors.append(col_colors)
    return colors


def _forward_return_cell_color(
    row_name: str,
    value: float | None,
    scale_values: list[float],
) -> str:
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
    return _mix_rgb((255, 255, 255), target, 0.25 + 0.65 * intensity)


def _mix_rgb(start: tuple[int, int, int], end: tuple[int, int, int], weight: float) -> str:
    """Linearly mix two RGB colors and return a CSS rgb string."""
    w = max(0.0, min(1.0, weight))
    rgb = tuple(round(a + (b - a) * w) for a, b in zip(start, end, strict=True))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _drawdown_comparison(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Compare percentage drawdowns from each symbol's running high."""
    syms: list = p.get("symbols") or ["^GSPC", "QQQ"]
    period = p.get("period", "5y")
    fig = go.Figure()
    for s in syms:
        df = provider.get_history(str(s), period=str(period), interval="1d")
        if df.empty:
            continue
        c = df["Close"].astype(float)
        run_max = c.cummax()
        dd = (c - run_max) / run_max * 100.0
        fig.add_trace(go.Scatter(x=df.index, y=dd, name=str(s), mode="lines"))
    fig.update_layout(title="Drawdown % (from running max)", yaxis_title="Drawdown %")
    return {"mime": "image/png", "data": _fig_to_b64(fig, "drawdown_comparison")}


def _log_cycle(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Plot weekly log10 close values for long-cycle price inspection."""
    sym = str(p.get("symbol", "BTC-USD"))
    period = p.get("period", "max")
    df = provider.get_history(sym, period=str(period), interval="1wk")
    if df.empty:
        raise ValueError("No data for log chart")
    c = df["Close"].astype(float)
    y = np.log10(c.replace(0, np.nan))
    fig = go.Figure(data=[go.Scatter(x=df.index, y=y, name=sym, mode="lines")])
    fig.update_layout(title=f"{sym} log10(close) weekly", yaxis_title="log10 price")
    return {"mime": "image/png", "data": _fig_to_b64(fig, "log_cycle")}

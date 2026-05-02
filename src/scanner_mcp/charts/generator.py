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
from scanner_mcp.indicators.core import Indicators

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


def _price_history(provider: YFinanceProvider, p: dict[str, Any]) -> dict[str, str]:
    """Create a candlestick chart for one symbol over a requested period."""
    sym = p.get("symbol", "SPY")
    period = p.get("period", "1y")
    interval = p.get("interval", "1d")
    df = provider.get_history(str(sym), period=str(period), interval=str(interval))
    if df.empty:
        raise ValueError("No price data")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=sym,
            )
        ]
    )
    fig.update_layout(title=f"{sym} {period} {interval}", xaxis_title="Date", yaxis_title="Price")
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
    """Build histograms of forward returns after a labeled event type."""
    from scanner_mcp.research.forward_returns import compute_event_forward_returns  # local import

    sym = str(p.get("symbol", "SPY"))
    event = str(p.get("event_type", "rsi_oversold"))
    windows = p.get("windows") or [7, 30, 90, 180]
    res = compute_event_forward_returns(provider, sym, event, [int(x) for x in windows], period="10y")
    w_int = [int(x) for x in windows]
    if not res or not any(res.get(ww, []) for ww in w_int):
        raise ValueError("No events or no forward returns; try a different symbol/event_type")
    nrows = max(1, len(windows))
    fig = make_subplots(
        rows=nrows,
        cols=1,
        subplot_titles=[f"{w}d forward %" for w in windows] if windows else [""],
    )
    for i, w in enumerate(w_int):
        rets = res.get(w, [])
        if rets:
            fig.add_trace(go.Histogram(x=rets, name=f"{w}d", nbinsx=30), row=i + 1, col=1)
    fig.update_layout(title=f"Forward returns after {event} — {sym}", showlegend=False)
    return {"mime": "image/png", "data": _fig_to_b64(fig, "forward_returns")}


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

"""Forward returns after labeled events (for charts + research resource)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scanner_mcp.data.provider import YFinanceProvider
from scanner_mcp.indicators import ta

log = logging.getLogger(__name__)


def _events_rsi(
    close: pd.Series, *, oversold: bool, period: int, thr: float
) -> list[int]:
    """Return indexes where RSI crosses into an overbought/oversold zone."""
    r = ta.rsi(close, length=period)
    if r is None or r.empty:
        return []
    shift = r.shift(1)
    out: list[int] = []
    for i in range(1, len(r)):
        if pd.isna(r.iloc[i]) or pd.isna(shift.iloc[i]):
            continue
        if oversold and shift.iloc[i] >= thr and r.iloc[i] < thr:
            out.append(i)
        if not oversold and shift.iloc[i] <= thr and r.iloc[i] > thr:
            out.append(i)
    return out


def compute_event_forward_returns(
    provider: YFinanceProvider,
    symbol: str,
    event_type: str,
    windows: list[int],
    period: str = "10y",
) -> dict[int, list[float]]:
    """Compute forward percentage returns after a supported event type."""
    df = provider.get_history(symbol, period=period, interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    close = df["Close"].astype(float).reset_index(drop=True)
    ev: list[int] = []
    if event_type == "rsi_oversold":
        ev = _events_rsi(close, oversold=True, period=14, thr=30.0)
    elif event_type == "rsi_overbought":
        ev = _events_rsi(close, oversold=False, period=14, thr=70.0)
    else:
        return {}

    if not ev:
        return {w: [] for w in windows}

    out: dict[int, list[float]] = {w: [] for w in windows}
    for idx in ev:
        base = float(close.iloc[idx])
        if base == 0 or not np.isfinite(base):
            continue
        for w in windows:
            j = idx + w
            if j >= len(close):
                continue
            nxt = float(close.iloc[j])
            if not np.isfinite(nxt):
                continue
            out[w].append((nxt - base) / base * 100.0)
    return out


def forward_returns_markdown(
    provider: YFinanceProvider,
    symbol: str,
    event_type: str,
) -> str:
    """Render forward-return summary statistics as a Markdown table."""
    windows = [7, 30, 90]
    res = compute_event_forward_returns(
        provider, symbol, event_type, windows, period="10y"
    )
    lines = [f"## Forward returns after {event_type}\n", f"**Symbol:** {symbol}\n", ""]
    lines.append("| Window | n | mean % | med % |")
    lines.append("|--------|---|--------|-------|")
    for w in windows:
        xs = res.get(w, [])
        if not xs:
            lines.append(f"| {w}d | 0 | — | — |")
            continue
        a = np.array(xs, dtype=float)
        lines.append(
            f"| {w}d | {len(a)} | {a.mean():.2f} | {float(np.median(a)):.2f} |"
        )
    return "\n".join(lines) + "\n"

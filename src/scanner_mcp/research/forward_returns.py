"""Forward returns after labeled events (for charts + research resource)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scanner_mcp.data.provider import DataProvider
from scanner_mcp.signals import calculations as calc
from scanner_mcp.signals.catalog import CATALOG, merge_params

log = logging.getLogger(__name__)

DEFAULT_FORWARD_WINDOWS = [5, 10, 21, 42, 63, 126, 252]


@dataclass(frozen=True)
class SignalEvent:
    """One historical signal instance on a price series."""

    index: int
    label: str


@dataclass(frozen=True)
class ForwardWindowResult:
    """Forward path statistics for one event and one horizon."""

    final_return: float
    max_loss: float
    max_gain: float


@dataclass(frozen=True)
class ForwardEvent:
    """One signal event enriched with per-horizon forward return stats."""

    index: int
    date: Any
    price: float
    label: str
    event_type: str
    windows: dict[int, ForwardWindowResult] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardStudy:
    """Reusable event-study payload for chart and research renderers."""

    symbol: str
    event_type: str
    windows: list[int]
    price: pd.Series
    events: list[ForwardEvent]


Detector = Callable[[pd.DataFrame, dict[str, Any]], list[SignalEvent]]


def _events_rsi(
    close: pd.Series, *, oversold: bool, period: int, thr: float
) -> list[int]:
    """Return indexes where RSI crosses into an overbought/oversold zone."""
    return calc.rsi_threshold_cross_indexes(close, period=period, threshold=thr, below=oversold)


def _detect_rsi_oversold(df: pd.DataFrame, params: dict[str, Any]) -> list[SignalEvent]:
    close = df["Close"].astype(float).reset_index(drop=True)
    period = int(params.get("period", 14))
    threshold = float(params.get("threshold", 30.0))
    return [
        SignalEvent(index=i, label="RSI Oversold")
        for i in _events_rsi(close, oversold=True, period=period, thr=threshold)
    ]


def _detect_rsi_overbought(df: pd.DataFrame, params: dict[str, Any]) -> list[SignalEvent]:
    close = df["Close"].astype(float).reset_index(drop=True)
    period = int(params.get("period", 14))
    threshold = float(params.get("threshold", 70.0))
    return [
        SignalEvent(index=i, label="RSI Overbought")
        for i in _events_rsi(close, oversold=False, period=period, thr=threshold)
    ]


def _detect_golden_cross(df: pd.DataFrame, params: dict[str, Any]) -> list[SignalEvent]:
    close = df["Close"].astype(float).reset_index(drop=True)
    fast = int(params.get("fast", 50))
    slow = int(params.get("slow", 200))
    _, fast_line = calc.moving_average(close, fast, "sma")
    _, slow_line = calc.moving_average(close, slow, "sma")
    return _cross_events(fast_line, slow_line, label=f"Golden Cross ({fast}/{slow})", bullish=True)


def _detect_macd_bullish_crossover(df: pd.DataFrame, params: dict[str, Any]) -> list[SignalEvent]:
    close = df["Close"].astype(float).reset_index(drop=True)
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    signal = int(params.get("signal", 9))
    macd = calc.macd(close, fast=fast, slow=slow, signal=signal)
    if macd is None:
        return []
    out, cols = macd
    if cols is None:
        return []
    return _cross_events(
        out[cols.macd],
        out[cols.signal],
        label=f"MACD Bullish Cross ({fast}/{slow}/{signal})",
        bullish=True,
    )


def _detect_pct_from_ma(df: pd.DataFrame, params: dict[str, Any]) -> list[SignalEvent]:
    close = df["Close"].astype(float).reset_index(drop=True)
    ma_period = int(params.get("ma_period", 50))
    pct = float(params.get("pct", 2.0))
    ma_type, ma = calc.moving_average(close, ma_period, str(params.get("ma_type", "sma")))

    diff = calc.pct_distance_from_ma(close, ma)
    out: list[SignalEvent] = []
    label = f"Within {pct:g}% of {ma_type.upper()} {ma_period}"
    for i in range(1, len(diff)):
        prev = diff.iloc[i - 1]
        cur = diff.iloc[i]
        if any(pd.isna(x) for x in (prev, cur)):
            continue
        if prev > pct and cur <= pct:
            out.append(SignalEvent(index=i, label=label))
    return out


def _cross_events(
    lhs: pd.Series,
    rhs: pd.Series,
    *,
    label: str,
    bullish: bool,
) -> list[SignalEvent]:
    direction: calc.CrossDirection = "bullish" if bullish else "bearish"
    return [SignalEvent(index=i, label=label) for i in calc.cross_indexes(lhs, rhs, direction)]


DETECTORS: dict[str, Detector] = {
    "golden_cross": _detect_golden_cross,
    "macd_bullish_crossover": _detect_macd_bullish_crossover,
    "pct_from_ma": _detect_pct_from_ma,
    "rsi_oversold": _detect_rsi_oversold,
    "rsi_overbought": _detect_rsi_overbought,
}


def _clean_windows(windows: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for w in windows:
        wi = int(w)
        if wi <= 0 or wi in seen:
            continue
        out.append(wi)
        seen.add(wi)
    return out


def _coerce_signal_dates(signal_dates: Iterable[Any]) -> list[pd.Timestamp]:
    """Parse and deduplicate user-supplied signal dates."""
    out: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for raw in signal_dates:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid signal date: {raw}")
        ts = pd.Timestamp(parsed).normalize().tz_localize(None)
        if ts in seen:
            continue
        seen.add(ts)
        out.append(ts)
    return sorted(out)


def _resolve_custom_events(
    close: pd.Series,
    signal_dates: Iterable[Any],
    *,
    label: str = "Custom Signal",
) -> list[SignalEvent]:
    """Map requested calendar dates onto the next available trading session."""
    if close.empty:
        return []
    requested = _coerce_signal_dates(signal_dates)
    if not requested:
        return []

    trading_index = pd.DatetimeIndex(pd.to_datetime(close.index)).tz_localize(None).normalize()
    out: list[SignalEvent] = []
    seen_indexes: set[int] = set()
    for requested_date in requested:
        position = trading_index.searchsorted(requested_date, side="left")
        if position >= len(trading_index):
            continue
        idx = int(position)
        if idx in seen_indexes:
            continue
        seen_indexes.add(idx)
        out.append(SignalEvent(index=idx, label=label))
    return out


def compute_event_forward_study_from_history(
    df: pd.DataFrame,
    symbol: str,
    event_type: str,
    windows: list[int],
    params: dict[str, Any] | None = None,
    detectors: dict[str, Detector] | None = None,
) -> ForwardStudy:
    """Compute event-study statistics from an already-fetched OHLCV history."""
    w_int = _clean_windows(windows)
    if df is None or df.empty or "Close" not in df.columns:
        return ForwardStudy(symbol=symbol, event_type=event_type, windows=w_int, price=pd.Series(dtype=float), events=[])

    registry = DETECTORS if detectors is None else detectors
    detector = registry.get(event_type)
    if detector is None:
        return ForwardStudy(symbol=symbol, event_type=event_type, windows=w_int, price=pd.Series(dtype=float), events=[])

    close = df["Close"].astype(float)
    close_pos = close.reset_index(drop=True)
    detector_params = merge_params(event_type, params) if event_type in CATALOG else (params or {})
    raw_events = detector(df, detector_params)
    events: list[ForwardEvent] = []

    for raw in raw_events:
        idx = int(raw.index)
        if idx < 0 or idx >= len(close_pos):
            continue
        base = float(close_pos.iloc[idx])
        if base == 0 or not np.isfinite(base):
            continue

        per_window: dict[int, ForwardWindowResult] = {}
        for w in w_int:
            j = idx + w
            if j >= len(close_pos):
                continue
            path = close_pos.iloc[idx : j + 1].astype(float)
            if path.empty or not np.isfinite(path).all():
                continue
            pct_path = (path - base) / base * 100.0
            per_window[w] = ForwardWindowResult(
                final_return=float(pct_path.iloc[-1]),
                max_loss=float(pct_path.min()),
                max_gain=float(pct_path.max()),
            )

        if per_window:
            events.append(
                ForwardEvent(
                    index=idx,
                    date=close.index[idx],
                    price=base,
                    label=raw.label,
                    event_type=event_type,
                    windows=per_window,
                )
            )

    return ForwardStudy(symbol=symbol, event_type=event_type, windows=w_int, price=close, events=events)


def compute_custom_date_forward_study_from_history(
    df: pd.DataFrame,
    symbol: str,
    signal_dates: list[Any],
    windows: list[int],
    *,
    label: str = "Custom Signal",
) -> ForwardStudy:
    """Compute forward returns for user-supplied signal dates."""
    w_int = _clean_windows(windows)
    if df is None or df.empty or "Close" not in df.columns:
        return ForwardStudy(symbol=symbol, event_type="custom_dates", windows=w_int, price=pd.Series(dtype=float), events=[])

    synthetic_detector: dict[str, Detector] = {
        "custom_dates": lambda frame, _: _resolve_custom_events(frame["Close"].astype(float), signal_dates, label=label)
    }
    return compute_event_forward_study_from_history(
        df,
        symbol,
        "custom_dates",
        w_int,
        params=None,
        detectors=synthetic_detector,
    )


def compute_custom_date_forward_study(
    provider: DataProvider,
    symbol: str,
    signal_dates: list[Any],
    windows: list[int] | None = None,
    period: str = "10y",
    *,
    label: str = "Custom Signal",
) -> ForwardStudy:
    """Compute forward returns for explicit signal dates instead of detector-derived events."""
    w_int = _clean_windows(windows or DEFAULT_FORWARD_WINDOWS)
    df = provider.get_history(symbol, period=period, interval="1d")
    return compute_custom_date_forward_study_from_history(df, symbol, signal_dates, w_int, label=label)


def compute_event_forward_study(
    provider: DataProvider,
    symbol: str,
    event_type: str,
    windows: list[int] | None = None,
    period: str = "10y",
    params: dict[str, Any] | None = None,
) -> ForwardStudy:
    """Compute forward return event-study records after a supported signal type."""
    w_int = _clean_windows(windows or DEFAULT_FORWARD_WINDOWS)
    df = provider.get_history(symbol, period=period, interval="1d")
    return compute_event_forward_study_from_history(df, symbol, event_type, w_int, params=params)


def summarize_forward_study(study: ForwardStudy) -> dict[int, dict[str, float | int]]:
    """Aggregate event-study records by forward horizon."""
    out: dict[int, dict[str, float | int]] = {}
    for w in study.windows:
        vals = [ev.windows[w] for ev in study.events if w in ev.windows]
        if not vals:
            out[w] = {"n": 0}
            continue
        finals = np.array([x.final_return for x in vals], dtype=float)
        losses = np.array([x.max_loss for x in vals], dtype=float)
        gains = np.array([x.max_gain for x in vals], dtype=float)
        out[w] = {
            "n": int(len(finals)),
            "mean": float(finals.mean()),
            "median": float(np.median(finals)),
            "positive_pct": float((finals > 0).mean() * 100.0),
            "avg_max_loss": float(losses.mean()),
            "avg_max_gain": float(gains.mean()),
        }
    return out


def compute_event_forward_returns(
    provider: DataProvider,
    symbol: str,
    event_type: str,
    windows: list[int],
    period: str = "10y",
) -> dict[int, list[float]]:
    """Compute forward percentage returns after a supported event type.

    Kept as a compact compatibility wrapper around the richer event-study API.
    """
    study = compute_event_forward_study(provider, symbol, event_type, windows, period=period)
    out: dict[int, list[float]] = {w: [] for w in study.windows}
    for ev in study.events:
        for w, result in ev.windows.items():
            out[w].append(result.final_return)
    return out


def forward_returns_markdown(
    provider: DataProvider,
    symbol: str,
    event_type: str,
) -> str:
    """Render forward-return summary statistics as a Markdown table."""
    windows = [7, 30, 90]
    study = compute_event_forward_study(provider, symbol, event_type, windows, period="10y")
    summary = summarize_forward_study(study)
    lines = [f"## Forward returns after {event_type}\n", f"**Symbol:** {symbol}\n", ""]
    lines.append("| Window | n | mean % | med % |")
    lines.append("|--------|---|--------|-------|")
    for w in windows:
        row = summary.get(w, {"n": 0})
        if not row.get("n"):
            lines.append(f"| {w}d | 0 | — | — |")
            continue
        lines.append(
            f"| {w}d | {row['n']} | {float(row['mean']):.2f} | {float(row['median']):.2f} |"
        )
    return "\n".join(lines) + "\n"

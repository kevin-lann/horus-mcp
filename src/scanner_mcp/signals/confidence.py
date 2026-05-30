from __future__ import annotations

from typing import Any

from scanner_mcp.signals.catalog import confidence_grade, clampconfidence_score


def with_confidence_result(signal_type: str, triggered: bool, details: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Attach confidence metadata to a signal result."""
    return triggered, with_confidence(signal_type, triggered, details)


def with_confidence(signal_type: str, triggered: bool, details: dict[str, Any]) -> dict[str, Any]:
    """Attach confidence score and letter grade to evaluator details."""
    score = confidence_score(signal_type, triggered, details)
    return {
        **details,
        "confidence_score": score,
        "confidence_grade": confidence_grade(score),
    }


def confidence_score(signal_type: str, triggered: bool, details: dict[str, Any]) -> int:
    """Estimate a 0-100 confidence score from the latest signal measurements."""
    if details.get("error") or details.get("reason") in {
        "insufficient_bars",
        "no_data",
        "nan",
        "rsi_nan",
        "ma_fail",
        "ma_nan",
        "macd_fail",
        "macd_col",
        "bb_fail",
        "bb_cols",
        "sma_fail",
        "ath_zero",
        "prior_zero",
    }:
        return 0

    if signal_type in {"golden_cross", "death_cross"}:
        values = [float(v) for k, v in details.items() if k.startswith("sma_")]
        if len(values) != 2:
            return 0
        spread_pct = abs(values[0] - values[1]) / max(abs(values[1]), 1e-9) * 100.0
        base = 70 if triggered else 25
        return clampconfidence_score(base + min(30.0, spread_pct * (12.0 if triggered else 6.0)))

    if signal_type in {"macd_bullish_crossover", "macd_bearish_crossover"}:
        spread = abs(float(details["macd"]) - float(details["signal"]))
        denom = max(abs(float(details["macd"])), abs(float(details["signal"])), 1.0)
        spread_pct = spread / denom * 100.0
        base = 70 if triggered else 25
        return clampconfidence_score(base + min(30.0, spread_pct * (0.6 if triggered else 0.3)))

    if signal_type in {"rsi_oversold", "rsi_overbought"}:
        distance = abs(float(details["rsi"]) - float(details["threshold"]))
        base = 70 if triggered else 30
        return clampconfidence_score(base + min(30.0, distance * (3.0 if triggered else 1.5)))

    if signal_type == "pct_from_ma":
        diff = abs(float(details["diff_pct"]))
        threshold = max(abs(float(details["threshold_pct"])), 1e-9)
        if triggered:
            closeness = max(0.0, 1.0 - (diff / threshold))
            return clampconfidence_score(70.0 + closeness * 30.0)
        overshoot = max(0.0, (diff - threshold) / threshold)
        return clampconfidence_score(65.0 - min(65.0, overshoot * 40.0))

    if signal_type == "pct_from_ath":
        drawdown_pct = abs(float(details["pct_from_ath"]))
        threshold = max(abs(float(details["threshold_pct"])), 1e-9)
        ratio = drawdown_pct / threshold
        if triggered:
            return clampconfidence_score(70.0 + min(30.0, (ratio - 1.0) * 30.0))
        return clampconfidence_score(min(69.0, ratio * 69.0))

    if signal_type == "bbands_breakout":
        close = float(details["close"])
        lower = float(details["lower"])
        upper = float(details["upper"])
        width = max(upper - lower, 1e-9)
        outside = max(lower - close, close - upper, 0.0)
        if triggered:
            return clampconfidence_score(70.0 + min(30.0, outside / width * 200.0))
        gap = min(abs(close - lower), abs(close - upper))
        return clampconfidence_score(30.0 + max(0.0, 30.0 - gap / width * 100.0))

    if signal_type == "bull_flag":
        move = max(float(details["prior_move_pct"]), 0.0)
        range_pct = max(float(details["consol_range_pct"]), 0.0)
        tightness = max(0.0, 100.0 - range_pct * 25.0)
        impulse = min(100.0, move * 5.0)
        score = (tightness + impulse) / 2.0
        if triggered:
            score = max(score, 70.0)
        else:
            score *= 0.5
        return clampconfidence_score(score)

    return 0
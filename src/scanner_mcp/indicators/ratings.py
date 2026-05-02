"""Map indicator outputs to buy / hold / sell and aggregate consensus."""

from __future__ import annotations

from typing import Any, Literal

Rating = Literal["buy", "hold", "sell"]


def _vote_weight(r: Rating) -> int:
    """Map a rating to a numeric vote used for consensus."""
    return 1 if r == "buy" else (-1 if r == "sell" else 0)


def consensus(ratings: list[Rating]) -> Rating:
    """Aggregate indicator ratings by simple majority vote weight."""
    if not ratings:
        return "hold"
    s = sum(_vote_weight(r) for r in ratings)
    if s > 0:
        return "buy"
    if s < 0:
        return "sell"
    return "hold"


def rate_rsi(value: float | None) -> Rating:
    """Rate RSI with classic oversold/overbought thresholds."""
    if value is None:
        return "hold"
    if value < 30:
        return "buy"
    if value > 70:
        return "sell"
    return "hold"


def rate_macd(m: dict[str, float | None] | None) -> Rating:
    """Rate MACD by histogram sign and whether momentum is increasing."""
    if not m:
        return "hold"
    h = m.get("hist")
    ph = m.get("hist_prev")
    if h is None:
        return "hold"
    if h > 0 and (ph is not None) and h > ph:
        return "buy"
    if h < 0 and (ph is not None) and h < ph:
        return "sell"
    return "hold"


def rate_bbands(m: dict[str, float | None] | None) -> Rating:
    """Rate Bollinger Bands using percent-B extremes."""
    if not m or m.get("pct_b") is None:
        return "hold"
    pb = float(m["pct_b"])
    if pb < 0.2:
        return "buy"
    if pb > 0.8:
        return "sell"
    return "hold"


def rate_price_vs_ma(price: float | None, ma: float | None) -> Rating:
    """Rate price relative to a moving average with a 1 percent deadband."""
    if price is None or ma is None or ma == 0:
        return "hold"
    d = (price - ma) / abs(ma) * 100.0
    if d > 1.0:
        return "buy"
    if d < -1.0:
        return "sell"
    return "hold"


def rate_ath_distance(pct: float | None) -> Rating:
    """Rate percent distance from all-time high as dip/extended context."""
    if pct is None:
        return "hold"
    if pct < -30:
        return "buy"
    if pct > -3:
        return "sell"
    return "hold"


def rate_single(indicator: str, payload: Any) -> Rating:
    """Dispatch rating logic for one indicator payload."""
    k = indicator.lower().split(":", 1)[0]
    if k == "rsi":
        return rate_rsi(payload if isinstance(payload, (int, float)) else None)
    if k == "macd" and isinstance(payload, dict):
        return rate_macd(payload)  # type: ignore[arg-type]
    if k == "bbands" and isinstance(payload, dict):
        return rate_bbands(payload)  # type: ignore[arg-type]
    if k in ("sma", "ema") and isinstance(payload, dict):
        return rate_price_vs_ma(  # type: ignore[arg-type]
            float(payload.get("price")) if payload.get("price") is not None else None,
            float(payload.get("value")) if payload.get("value") is not None else None,
        )
    if k == "ath_distance" and isinstance(payload, (int, float)):
        return rate_ath_distance(float(payload))
    if k == "beta":
        return "hold"
    return "hold"

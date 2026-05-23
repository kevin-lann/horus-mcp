"""Market computation and quote helpers used by MCP handlers."""

from __future__ import annotations

import logging
from typing import Any

from scanner_mcp.data.provider import DataProvider
from scanner_mcp.indicators import ratings
from scanner_mcp.indicators.core import Indicators, beta_from_returns

log = logging.getLogger(__name__)

MARKET_SNAPSHOT: dict[str, list[str]] = {
    "us_indices": ["^GSPC", "^IXIC", "^DJI", "^RUT"],
    "etfs": ["SPY", "QQQ", "IWM", "GLD", "TLT", "USO", "XLE"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "volatility": ["^VIX"],
}


def parse_indicator(name: str) -> tuple[str, dict[str, Any]]:
    """Parse indicator specs like `rsi:14` into a key and parameter dict."""
    cleaned = name.strip().lower()
    if ":" in cleaned:
        key, rest = cleaned.split(":", 1)
        key = key.strip()
        if key == "rsi":
            return "rsi", {"period": int(rest)}
        if key in {"sma", "ema"}:
            return key, {"period": int(rest)}
    return cleaned, {}


def as_float(value: Any) -> float | None:
    """Best-effort float conversion that treats NaN and invalid input as None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def first_present(data: dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value among several possible dictionary keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def quote_from_history(provider: DataProvider, symbol: str) -> dict[str, float | None]:
    """Fallback quote calculation from recent daily closes."""
    df = provider.get_history(symbol, period="10d", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        cols = [] if df is None else list(df.columns)
        log.debug("quote history fallback unavailable %s empty=%s cols=%s", symbol, df is None or df.empty, cols)
        return {"last_price": None, "previous_close": None, "day_change_pct": None}

    close = df["Close"].dropna()
    if close.empty:
        log.debug("quote history fallback close empty %s rows=%s", symbol, len(df))
        return {"last_price": None, "previous_close": None, "day_change_pct": None}

    last = as_float(close.iloc[-1])
    prev = as_float(close.iloc[-2]) if len(close) > 1 else None
    chg_pct = None
    if last is not None and prev not in (None, 0):
        chg_pct = (last - prev) / prev * 100.0
    log.debug("quote history fallback %s last=%s prev=%s pct=%s rows=%s", symbol, last, prev, chg_pct, len(close))
    return {"last_price": last, "previous_close": prev, "day_change_pct": chg_pct}


def quote_snapshot(provider: DataProvider, symbol: str) -> dict[str, Any]:
    """Resolve last price and daily change from fast_info with history fallback."""
    info = provider.get_fast_info(symbol) or {}
    log.debug("quote snapshot fast_info %s keys=%s", symbol, sorted(info.keys()))
    last = as_float(first_present(info, "last_price", "lastPrice"))
    prev = as_float(
        first_present(
            info,
            "previous_close",
            "previousClose",
            "regular_market_previous_close",
            "regularMarketPreviousClose",
        )
    )
    chg_pct = as_float(first_present(info, "regular_market_change_percent", "regularMarketChangePercent"))

    if chg_pct is None and last is not None and prev not in (None, 0):
        chg_pct = (last - prev) / prev * 100.0

    if last is None or chg_pct is None:
        log.debug("quote snapshot %s falling back to history last=%s prev=%s pct=%s", symbol, last, prev, chg_pct)
        hist = quote_from_history(provider, symbol)
        last = last if last is not None else hist["last_price"]
        prev = prev if prev is not None else hist["previous_close"]
        chg_pct = chg_pct if chg_pct is not None else hist["day_change_pct"]

    log.debug("quote snapshot %s resolved last=%s prev=%s pct=%s", symbol, last, prev, chg_pct)
    return {"last_price": last, "previous_close": prev, "day_change_pct": chg_pct}


def compute_indicators(
    provider: DataProvider,
    symbol: str,
    names: list[str],
    period: str,
) -> dict[str, Any]:
    """Compute requested indicators and attach per-indicator ratings."""
    df = provider.get_history(symbol, period=period, interval="1d")
    if df is None or df.empty:
        return {"error": "no_history", "symbol": symbol}
    if len(df) < 5:
        return {"error": "not_enough_bars", "symbol": symbol}
    ind = Indicators(df)
    out: dict[str, Any] = {}
    ratings_list: list[ratings.Rating] = []

    for raw in names:
        key, extra = parse_indicator(raw)
        key = key.strip()

        if key == "rsi":
            period_value = int(extra.get("period", 14))
            value = ind.rsi(period=period_value)
            rating = ratings.rate_rsi(value)
            label = f"rsi_{period_value}" if ":" in raw else "rsi"
            out[label] = {"value": value, "rating": rating}
            ratings_list.append(rating)
        elif key == "macd":
            macd = ind.macd()
            rating = ratings.rate_macd(macd)
            block: dict[str, Any] = {}
            if macd:
                for field in ("macd", "signal", "hist", "hist_prev"):
                    if macd.get(field) is not None:
                        block[field] = macd[field]
            out["macd"] = {**block, "rating": rating} if macd else {"value": None, "rating": "hold"}
            ratings_list.append(rating)
        elif key == "bbands":
            bands = ind.bbands()
            rating = ratings.rate_bbands(bands)
            out["bbands"] = {**(bands or {}), "rating": rating}
            ratings_list.append(rating)
        elif key in {"sma", "ema"}:
            period_value = int(extra.get("period", 50 if key == "sma" else 20))
            value = ind.sma(period_value) if key == "sma" else ind.ema(period_value)
            price = ind.last_close()
            rating = ratings.rate_price_vs_ma(price, value)
            name = f"{key}_{period_value}"
            out[name] = {"value": value, "price": price, "rating": rating}
            ratings_list.append(rating)
        elif key == "ath_distance":
            value = ind.ath_distance()
            rating = ratings.rate_ath_distance(value)
            out["ath_distance"] = {"value": value, "rating": rating}
            ratings_list.append(rating)
        elif key == "beta":
            beta_history = provider.get_history(symbol, period="1y", interval="1d")
            benchmark_history = provider.get_history("SPY", period="1y", interval="1d")
            beta_value = None
            if not beta_history.empty and not benchmark_history.empty:
                r1 = beta_history["Close"].pct_change()
                r2 = benchmark_history["Close"].pct_change()
                beta_value = beta_from_returns(r1, r2)
            out["beta"] = {"value": beta_value, "rating": "hold"}
        else:
            out[raw] = {"error": f"unknown indicator: {key}"}

    out["consensus"] = ratings.consensus(ratings_list) if ratings_list else "hold"
    out["symbol"] = symbol.upper()
    return out


def market_snapshot(provider: DataProvider) -> dict[str, Any]:
    """Build a snapshot payload for the configured symbol buckets."""
    out: dict[str, Any] = {}
    for category, tickers in MARKET_SNAPSHOT.items():
        out[category] = []
        for ticker in tickers:
            quote = quote_snapshot(provider, ticker)
            out[category].append(
                {
                    "symbol": ticker,
                    "price": quote["last_price"],
                    "day_change_pct": quote["day_change_pct"],
                }
            )
    return out

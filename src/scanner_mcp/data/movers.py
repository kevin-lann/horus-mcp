"""Top gainers/losers and crypto movers via yfinance screener + quotes."""

from __future__ import annotations

import logging
from typing import Any, Literal

import yfinance as yf

log = logging.getLogger(__name__)

Exchange = Literal["NYSE", "NASDAQ", "AMEX", "CRYPTO"]

# Yahoo Finance exchange short codes
_EXCHANGE_SETS: dict[Exchange, frozenset[str]] = {
    "NYSE": frozenset({"NYQ", "NYS", "PCX", "SNY"}),
    "NASDAQ": frozenset({"NMS", "NCM", "NGM", "NAS", "NMQ", "NGS"}),
    "AMEX": frozenset({"ASE", "AMQ", "AAS", "PHE", "BTS"}),
}

# Major spot pairs to rank for CRYPTO mode
CRYPTO_TICKERS = _CRYPTO_TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "AVAX-USD",
    "DOT-USD",
    "MATIC-USD",
    "LINK-USD",
    "LTC-USD",
    "BCH-USD",
    "ATOM-USD",
    "UNI-USD",
    "XLM-USD",
    "NEAR-USD",
    "ARB-USD",
    "OP-USD",
    "SUI-USD",
    "PEPE-USD",
]


def _quote_to_row(q: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one Yahoo screener quote into the mover row schema."""
    sym = q.get("symbol")
    if not sym:
        return None
    name = q.get("longName") or q.get("shortName") or sym
    price = q.get("regularMarketPrice") or q.get("regularMarketPreviousClose")
    if price is None:
        return None
    chg = q.get("regularMarketChangePercent")
    if chg is None:
        return None
    vol = q.get("regularMarketVolume")
    ex = q.get("exchange", "")
    return {
        "symbol": sym,
        "name": name,
        "price": float(price),
        "change_pct": float(chg),
        "volume": int(vol) if vol is not None else 0,
        "exchange": ex,
    }


def _filter_quotes(
    quotes: list[dict[str, Any]],
    exchange: Exchange,
) -> list[dict[str, Any]]:
    """Keep only screener quotes that belong to the requested equity exchange."""
    if exchange == "CRYPTO":
        return []
    codes = _EXCHANGE_SETS[exchange]
    rows: list[dict[str, Any]] = []
    for q in quotes:
        ex = q.get("exchange") or ""
        if ex in codes:
            r = _quote_to_row(q)
            if r:
                rows.append(r)
    return rows


def screen_movers(
    which: Literal["gainers", "losers"],
    exchange: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return top gainers or losers for an equity exchange or crypto universe."""
    ex = exchange.upper() if exchange else "NYSE"
    if ex not in _EXCHANGE_SETS and ex != "CRYPTO":
        raise ValueError(f"Invalid exchange: {exchange}. Use NYSE, NASDAQ, AMEX, or CRYPTO.")
    ex_t = ex  # type: ignore[assignment]

    if ex_t == "CRYPTO":
        return _crypto_movers(which, limit)

    qname = "day_gainers" if which == "gainers" else "day_losers"
    try:
        r = yf.screen(qname, count=min(250, max(limit * 5, 50)))
    except Exception as e:  # noqa: BLE001
        log.exception("yfinance.screen failed: %s", e)
        return []

    raw = r.get("quotes") if isinstance(r, dict) else None
    if not raw:
        return []
    rows = _filter_quotes(raw, ex_t)  # type: ignore[arg-type]
    if which == "losers":
        rows.sort(key=lambda x: x["change_pct"])
    else:
        rows.sort(key=lambda x: x["change_pct"], reverse=True)
    return rows[:limit]


def _crypto_movers(
    which: Literal["gainers", "losers"],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank a fixed list of liquid crypto pairs by daily percentage change."""
    tks = yf.Tickers(" ".join(_CRYPTO_TICKERS))
    m = tks.tickers
    rows: list[dict[str, Any]] = []
    for sym in _CRYPTO_TICKERS:
        try:
            tick = m[sym] if sym in m else yf.Ticker(sym)
            q: dict = dict(tick.fast_info) if tick.fast_info else {}
            if not q:
                continue
            name = str(sym)
            last = q.get("last_price")
            prev = q.get("previous_close", last)
            if last is None or prev in (0, None):
                continue
            chg = (float(last) - float(prev)) / float(prev) * 100.0
            r = {
                "symbol": sym,
                "name": name,
                "price": float(last),
                "change_pct": float(chg),
                "volume": int(q.get("last_volume", 0) or 0),
                "exchange": "CRYPTO",
            }
            rows.append(r)
        except Exception:  # noqa: BLE001
            log.debug("crypto quote failed for %s", sym, exc_info=True)
    rev = which == "gainers"
    rows.sort(key=lambda x: x["change_pct"], reverse=rev)
    return rows[:limit]

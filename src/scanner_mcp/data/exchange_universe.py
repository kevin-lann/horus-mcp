"""Resolve full ticker universes for US exchanges via Yahoo screener (paginated)."""

from __future__ import annotations

import logging
import os
import time
import yfinance as yf
from yfinance import EquityQuery

from scanner_mcp.data.movers import CRYPTO_TICKERS, Exchange

log = logging.getLogger(__name__)

# Yahoo EquityQuery valid US MIC codes (yfinance.const EQUITY_SCREENER_EQ_MAP), grouped by venue.
_MIC: dict[Exchange, list[str]] = {
    "NASDAQ": ["NMS", "NCM", "NGM"],
    "NYSE": ["NYQ", "PCX"],
    "AMEX": ["ASE", "BTS"],
}

_CACHE_TTL = float(os.environ.get("SCANNER_MCP_EXCHANGE_LIST_TTL", "3600")) # 1 hour
_CACHE: dict[str, tuple[float, list[str]]] = {}


def _exchange_query(user_ex: Exchange) -> EquityQuery:
    """Build region=us AND OR(exchange=mic) for one listed US venue."""
    codes = _MIC[user_ex]
    ors = [EquityQuery("eq", ["exchange", c]) for c in codes]
    return EquityQuery(
        "and",
        [EquityQuery("eq", ["region", "us"]), EquityQuery("or", ors)],
    )


def fetch_exchange_tickers(
    exchange: str,
    *,
    max_symbols: int | None = None,
    use_cache: bool = True,
) -> list[str]:
    """Return Yahoo Finance equity symbols for NYSE / NASDAQ / AMEX, or spot pairs for CRYPTO.

    Pagination uses screener ``offset`` + ``size`` (max 250 per request). Results are cached
    in-process for ``SCANNER_MCP_EXCHANGE_LIST_TTL`` seconds (default 3600).
    ``max_symbols`` truncates after deduplication (default: env
    ``SCANNER_MCP_EXCHANGE_MAX_SYMBOLS``, or unlimited when unset/empty).
    """
    ex: Exchange = exchange.upper()  # type: ignore[assignment]
    if ex not in _MIC and ex != "CRYPTO":
        raise ValueError(f"Invalid exchange: {exchange}. Use NYSE, NASDAQ, AMEX, or CRYPTO.")

    env_max = os.environ.get("SCANNER_MCP_EXCHANGE_MAX_SYMBOLS", "").strip()
    cap: int | None
    if max_symbols is not None:
        cap = max_symbols
    elif env_max:
        try:
            cap = int(env_max)
        except ValueError:
            cap = None
    else:
        cap = None

    cache_key = ex
    now = time.monotonic()
    if use_cache and cache_key in _CACHE:
        ts, syms = _CACHE[cache_key]
        if now - ts < _CACHE_TTL:
            out = list(syms)
            if cap is not None:
                out = out[:cap]
            return out

    if ex == "CRYPTO":
        out = [s.upper() for s in CRYPTO_TICKERS]
        _CACHE[cache_key] = (now, out)
        if cap is not None:
            return out[:cap]
        return list(out)

    q = _exchange_query(ex)
    symbols: list[str] = []
    seen: set[str] = set()
    offset = 0

    while True:
        try:
            r = yf.screen(q, offset=offset, size=250, sortField="ticker", sortAsc=True)
        except Exception as e:  # noqa: BLE001
            log.exception("exchange screen failed %s offset=%s: %s", ex, offset, e)
            break
        quotes = r.get("quotes") or []
        if not quotes:
            break
        for row in quotes:
            sym = row.get("symbol")
            if not sym:
                continue
            u = str(sym).upper()
            if u not in seen:
                seen.add(u)
                symbols.append(u)
                if cap is not None and len(symbols) >= cap:
                    _CACHE[cache_key] = (now, symbols)
                    return list(symbols)

        offset += len(quotes)
        if len(quotes) < 250:
            break

    _CACHE[cache_key] = (now, symbols)
    return list(symbols)

"""FastMCP entry point: market tools, signals, charts, resources."""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from io import StringIO
from typing import Annotated, Any, AsyncIterator, Literal

from pydantic import Field

from scanner_mcp.mcp_schemas import YFINANCE_PERIOD_DESC

from apscheduler.schedulers.base import BaseScheduler
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from scanner_mcp.charts import generator as chartgen
from scanner_mcp.data.exchange_universe import fetch_exchange_tickers
from scanner_mcp.data.movers import screen_movers
from scanner_mcp.data.provider import YFinanceProvider
from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.indicators.core import Indicators, beta_from_returns
from scanner_mcp.indicators import ratings
from scanner_mcp.research.forward_returns import forward_returns_markdown
from scanner_mcp.scanner import scheduler as scan_sched
from scanner_mcp.signals.catalog import CATALOG, list_catalog_entries, merge_params
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal

log = logging.getLogger(__name__)

_store: Store | None = None
_provider: YFinanceProvider | None = None
_sched: BaseScheduler | None = None

_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "CRYPTO"})

MARKET_SNAPSHOT: dict[str, list[str]] = {
    "us_indices": ["^GSPC", "^IXIC", "^DJI", "^RUT"],
    "etfs": ["SPY", "QQQ", "IWM", "GLD", "TLT", "USO", "XLE"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "volatility": ["^VIX"],
}


def _shutdown_scheduler() -> None:
    """Stop the background scheduler if it was started."""
    global _sched
    if _sched is None:
        return
    try:
        if _sched.running:
            _sched.shutdown(wait=False)
    except Exception as e:  # noqa: BLE001
        log.debug("Scheduler shutdown skipped: %s", e)
    finally:
        _sched = None


def _get_store() -> Store:
    """Lazily create the SQLite store, honoring SCANNER_MCP_DB."""
    global _store
    if _store is None:
        path = os.environ.get("SCANNER_MCP_DB")
        _store = Store(path)
    return _store


def _get_provider() -> YFinanceProvider:
    """Lazily create the shared market data provider."""
    global _provider
    if _provider is None:
        _provider = YFinanceProvider()
    return _provider


def _parse_ind(name: str) -> tuple[str, dict[str, Any]]:
    """Parse indicator specs like `rsi:14` into a key and parameter dict."""
    s = name.strip().lower()
    if ":" in s:
        k, rest = s.split(":", 1)
        k = k.strip()
        if k == "rsi":
            return "rsi", {"period": int(rest)}
        if k in ("sma", "ema"):
            return k, {"period": int(rest)}
    return s, {}


@asynccontextmanager
async def _lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """FastMCP lifespan hook that starts and stops the scan scheduler."""
    global _sched
    st = _get_store()
    pr = _get_provider()
    try:
        _sched = scan_sched.start_scheduler(st, pr)
    except Exception as e:  # noqa: BLE001
        log.error("Could not start scheduler: %s", e)
    try:
        yield {"store": st, "provider": pr, "scheduler": _sched}
    finally:
        _shutdown_scheduler()


mcp = FastMCP("ScannerMCP", lifespan=_lifespan)


def _compute_indicators(
    sym: str,
    names: list[str],
    period: str,
) -> dict[str, Any]:
    """Compute requested indicators and attach per-indicator ratings.

    Unknown indicator names are represented as errors in the output instead of
    failing the whole tool response.
    """
    pr = _get_provider()
    df = pr.get_history(sym, period=period, interval="1d")
    if df is None or df.empty:
        return {"error": "no_history", "symbol": sym}
    if len(df) < 5:
        return {"error": "not_enough_bars", "symbol": sym}
    ind = Indicators(df)
    out: dict[str, Any] = {}
    rlist: list[ratings.Rating] = []

    for raw in names:
        key, extra = _parse_ind(raw)
        key = key.strip()

        if key == "rsi":
            p = int(extra.get("period", 14))
            v = ind.rsi(period=p)
            ra = ratings.rate_rsi(v)
            label = f"rsi_{p}" if ":" in raw else "rsi"
            out[label] = {"value": v, "rating": ra}
            rlist.append(ra)
        elif key == "macd":
            m = ind.macd()
            ra = ratings.rate_macd(m)
            block: dict[str, Any] = {}
            if m:
                for k2 in ("macd", "signal", "hist", "hist_prev"):
                    if m.get(k2) is not None:
                        block[k2] = m[k2]
            out["macd"] = {**block, "rating": ra} if m else {"value": None, "rating": "hold"}
            rlist.append(ra)
        elif key == "bbands":
            b = ind.bbands()
            ra = ratings.rate_bbands(b)
            out["bbands"] = {**(b or {}), "rating": ra}
            rlist.append(ra)
        elif key in ("sma", "ema"):
            p = int(extra.get("period", 50 if key == "sma" else 20))
            v = ind.sma(p) if key == "sma" else ind.ema(p)
            prc = ind.last_close()
            ra = ratings.rate_price_vs_ma(prc, v)
            name = f"{key}_{p}"
            out[name] = {"value": v, "price": prc, "rating": ra}
            rlist.append(ra)
        elif key == "ath_distance":
            v = ind.ath_distance()
            ra = ratings.rate_ath_distance(v)
            out["ath_distance"] = {"value": v, "rating": ra}
            rlist.append(ra)
        elif key == "beta":
            bdf = pr.get_history(sym, period="1y", interval="1d")
            bnb = pr.get_history("SPY", period="1y", interval="1d")
            b = None
            if not bdf.empty and not bnb.empty:
                r1 = bdf["Close"].pct_change()
                r2 = bnb["Close"].pct_change()
                b = beta_from_returns(r1, r2)
            out["beta"] = {"value": b, "rating": "hold"}
        else:
            out[raw] = {"error": f"unknown indicator: {key}"}

    out["consensus"] = ratings.consensus(rlist) if rlist else "hold"
    out["symbol"] = sym.upper()
    return out


def _as_float(value: Any) -> float | None:
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


def _resolve_signal_universe(srow: SignalRow, watch: list[str]) -> list[str]:
    """Tickers to scan for one persisted signal: overrides, watchlist, or full exchange list."""
    scope = srow.ticker_scope
    if scope == "tickers":
        return list(srow.ticker_overrides or [])
    if scope == "exchange":
        if not srow.exchange:
            return []
        try:
            return fetch_exchange_tickers(srow.exchange)
        except ValueError:
            return []
    return list(watch)


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    """Return the first non-None value among several possible dictionary keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _quote_from_history(p: YFinanceProvider, symbol: str) -> dict[str, float | None]:
    """Fallback quote calculation from recent daily closes."""
    df = p.get_history(symbol, period="10d", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        cols = [] if df is None else list(df.columns)
        log.debug("quote history fallback unavailable %s empty=%s cols=%s", symbol, df is None or df.empty, cols)
        return {"last_price": None, "previous_close": None, "day_change_pct": None}

    close = df["Close"].dropna()
    if close.empty:
        log.debug("quote history fallback close empty %s rows=%s", symbol, len(df))
        return {"last_price": None, "previous_close": None, "day_change_pct": None}

    last = _as_float(close.iloc[-1])
    prev = _as_float(close.iloc[-2]) if len(close) > 1 else None
    chg_pct = None
    if last is not None and prev not in (None, 0):
        chg_pct = (last - prev) / prev * 100.0
    log.debug("quote history fallback %s last=%s prev=%s pct=%s rows=%s", symbol, last, prev, chg_pct, len(close))
    return {"last_price": last, "previous_close": prev, "day_change_pct": chg_pct}


def _quote_snapshot(p: YFinanceProvider, symbol: str) -> dict[str, Any]:
    """Resolve last price and daily change from fast_info with history fallback."""
    f = p.get_fast_info(symbol) or {}
    log.debug("quote snapshot fast_info %s keys=%s", symbol, sorted(f.keys()))
    last = _as_float(_first_present(f, "last_price", "lastPrice"))
    prev = _as_float(
        _first_present(
            f,
            "previous_close",
            "previousClose",
            "regular_market_previous_close",
            "regularMarketPreviousClose",
        )
    )
    chg_pct = _as_float(_first_present(f, "regular_market_change_percent", "regularMarketChangePercent"))

    if chg_pct is None and last is not None and prev not in (None, 0):
        chg_pct = (last - prev) / prev * 100.0

    if last is None or chg_pct is None:
        log.debug("quote snapshot %s falling back to history last=%s prev=%s pct=%s", symbol, last, prev, chg_pct)
        hist = _quote_from_history(p, symbol)
        last = last if last is not None else hist["last_price"]
        prev = prev if prev is not None else hist["previous_close"]
        chg_pct = chg_pct if chg_pct is not None else hist["day_change_pct"]

    log.debug("quote snapshot %s resolved last=%s prev=%s pct=%s", symbol, last, prev, chg_pct)
    return {"last_price": last, "previous_close": prev, "day_change_pct": chg_pct}


def _chart_tool_result(chart_type: str, params: dict[str, Any]) -> Image | str:
    """Run chart generation: MCP image block on success, JSON text with `error` on failure."""
    pr = _get_provider()
    try:
        r = chartgen.generate_chart(pr, chart_type, params)
        if not isinstance(r, dict):
            return json.dumps({"error": "unexpected chart response"})
        b64 = r.get("data")
        if not isinstance(b64, str):
            return json.dumps({"error": "chart response missing image data"})
        return Image(data=base64.b64decode(b64))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


@mcp.tool()
def debug_quote(symbol: str) -> str:
    """Debug yfinance `fast_info` keys and daily-history fallback for one symbol.

    `symbol`: yfinance ticker (e.g. `AAPL`, `^GSPC`, `BTC-USD`).
    """
    p = _get_provider()
    f = p.get_fast_info(symbol) or {}
    df = p.get_history(symbol, period="10d", interval="1d")
    q = _quote_snapshot(p, symbol)
    hist_tail: dict[str, Any] | None = None
    if df is not None and not df.empty:
        row = df.tail(1).iloc[0]
        hist_tail = {str(k): _as_float(v) for k, v in row.items()}
    return json.dumps(
        {
            "symbol": symbol.upper(),
            "server_file": __file__,
            "fast_info_keys": sorted(f.keys()),
            "fast_info_sample": {
                "last_price": _first_present(f, "last_price", "lastPrice"),
                "previous_close": _first_present(f, "previous_close", "previousClose"),
                "regular_market_previous_close": _first_present(
                    f,
                    "regular_market_previous_close",
                    "regularMarketPreviousClose",
                ),
                "regular_market_change_percent": _first_present(
                    f,
                    "regular_market_change_percent",
                    "regularMarketChangePercent",
                ),
                "last_volume": _first_present(f, "last_volume", "lastVolume"),
            },
            "history": {
                "empty": df is None or df.empty,
                "rows": 0 if df is None else len(df),
                "columns": [] if df is None else list(df.columns),
                "last_index": None if df is None or df.empty else str(df.index[-1]),
                "tail": hist_tail,
            },
            "resolved": q,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def get_price(symbol: str) -> str:
    """Current / last price, day change, volume, and market cap from yfinance `fast_info` (with history fallback).

    `symbol`: yfinance ticker string (e.g. `AAPL`, `^VIX`, `BTC-USD`).
    """
    p = _get_provider()
    f = p.get_fast_info(symbol)
    q = _quote_snapshot(p, symbol)
    if not f and q["last_price"] is None:
        return json.dumps({"error": "no data", "symbol": symbol})
    last = q["last_price"]
    prev = q["previous_close"]
    chg = _as_float(_first_present(f, "day_change", "dayChange")) if f else None
    if chg is None and last is not None and prev is not None:
        chg = last - prev
    chg_p = q["day_change_pct"]
    o = {
        "symbol": symbol.upper(),
        "last_price": last,
        "previous_close": prev,
        "day_change": chg,
        "day_change_pct": chg_p,
        "volume": _first_present(f, "last_volume", "lastVolume", "shares_unlisted", "sharesUnlisted"),
        "market_cap": _first_present(f, "market_cap", "marketCap"),
    }
    return json.dumps(o, indent=2, default=str)


@mcp.tool()
def get_indicators(
    symbol: str,
    indicators: list[str],
    period: str = "6mo",
) -> str:
    """Compute indicators with buy/hold/sell rating each plus consensus.

    `symbol`: yfinance ticker.
    `indicators`: each element is one token — `rsi` or `rsi:<n>` (length `n`, default 14); `macd`;
    `bbands`; `sma` or `sma:<n>` (default 50); `ema` or `ema:<n>` (default 20); `ath_distance`; `beta`.
    `period`: yfinance history window for daily bars (`1d` interval), e.g. `1mo`, `6mo`, `1y`, `5y`, `max` (default `6mo`).
    """
    r = _compute_indicators(symbol, indicators, period)
    return json.dumps(r, indent=2, default=str)


@mcp.tool()
def get_ath_distance(symbol: str) -> str:
    """Percent distance of last daily close below the running all-time high (full `max` history window).

    `symbol`: yfinance ticker.
    """
    pr = _get_provider()
    df = pr.get_history(symbol, period="max", interval="1d")
    if df.empty:
        return json.dumps({"error": "no data"})
    ind = Indicators(df)
    d = ind.ath_distance()
    return json.dumps({"symbol": symbol.upper(), "pct_from_ath": d})


@mcp.tool()
def get_option_chain(symbol: str, expiry: str | None = None) -> str:
    """Options chain preview as plain text (up to ~20 rows per calls/puts).

    `symbol`: underlying yfinance ticker.
    `expiry`: `YYYY-MM-DD` for a specific expiry, or omit / null to use the first listed expiry.
    """
    p = _get_provider()
    r = p.get_option_chain(symbol, expiry)
    if r.get("error"):
        return f"Error: {r['error']}\nExpiries: {r.get('expiries', [])[:10]}"
    b = StringIO()
    c = r.get("calls")
    u = r.get("puts")
    if c is not None and not c.empty:
        b.write("=== CALLS ===\n")
        b.write(c.head(20).to_string())
        b.write("\n\n")
    if u is not None and not u.empty:
        b.write("=== PUTS ===\n")
        b.write(u.head(20).to_string())
    return b.getvalue() or "empty chain"


@mcp.tool()
def market_snapshot() -> str:
    """Major US indices, ETFs, crypto, VIX: `last_price` and `day_change_pct` per symbol (no arguments).

    Buckets: `us_indices`, `etfs`, `crypto`, `volatility` (fixed symbol lists in server config).
    """
    p = _get_provider()
    out: dict[str, Any] = {}
    for cat, tickers in MARKET_SNAPSHOT.items():
        out[cat] = []
        for t in tickers:
            q = _quote_snapshot(p, t)
            out[cat].append(
                {
                    "symbol": t,
                    "price": q["last_price"],
                    "day_change_pct": q["day_change_pct"],
                }
            )
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def top_gainers(exchange: str, limit: int = 20) -> str:
    """Top daily percentage gainers for an equity exchange or a fixed crypto pair list.

    `exchange`: `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (case-insensitive; `CRYPTO` ranks ~20 liquid USD pairs).
    `limit`: max rows (default 20).
    """
    try:
        rows = screen_movers("gainers", exchange, limit=limit)
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


@mcp.tool()
def top_losers(exchange: str, limit: int = 20) -> str:
    """Top daily percentage losers; same `exchange` and `limit` semantics as `top_gainers`."""
    try:
        rows = screen_movers("losers", exchange, limit=limit)
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_signal_catalog() -> str:
    """List predefined `signal_type` keys with descriptions and `default_params` (input for `create_signal`)."""
    return json.dumps(list_catalog_entries(), indent=2, default=str)


@mcp.tool()
def create_signal(
    name: str,
    signal_type: str,
    params: dict[str, Any] | None = None,
    ticker_scope: Literal["tickers", "watchlist", "exchange"] = "watchlist",
    ticker_overrides: list[str] | None = None,
    exchange: str | None = None,
) -> str:
    """Create a persisted enabled signal row.

    `name`: human-readable label (not necessarily unique).
    `signal_type`: must match a catalog key from `list_signal_catalog`.
    `params`: overrides merged on top of that type's catalog defaults (JSON object; use numbers not strings).
    `ticker_scope`: `watchlist` (scan the global watchlist; omit `ticker_overrides` and `exchange`),
    `tickers` (set non-empty `ticker_overrides`; omit `exchange`), or `exchange` (set `exchange`; omit overrides).
    `ticker_overrides`: required when scope is `tickers` — list of ticker strings.
    `exchange`: when scope is `exchange`, exactly `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (omit otherwise).
    """
    if signal_type not in CATALOG:
        return json.dumps({"error": f"invalid signal_type: {signal_type}"})
    scope = ticker_scope
    pdct = dict(params) if params else {}
    merge_params(signal_type, pdct)  # validate

    ex_norm: str | None = exchange.strip().upper() if exchange and exchange.strip() else None
    ov: list[str] | None = None

    if scope == "tickers":
        if not ticker_overrides:
            return json.dumps({"error": "ticker_scope=tickers requires non-empty ticker_overrides list"})
        if ex_norm:
            return json.dumps({"error": "exchange must be omitted when ticker_scope is tickers"})
        ov = [str(x).upper() for x in ticker_overrides if str(x).strip()]
        if not ov:
            return json.dumps({"error": "ticker_overrides list is empty"})

    elif scope == "watchlist":
        if ticker_overrides:
            return json.dumps({"error": "omit ticker_overrides when ticker_scope is watchlist"})
        if ex_norm:
            return json.dumps({"error": "omit exchange when ticker_scope is watchlist"})

    else:  # exchange
        if ticker_overrides:
            return json.dumps({"error": "omit ticker_overrides when ticker_scope is exchange"})
        if not ex_norm:
            return json.dumps({"error": "ticker_scope=exchange requires exchange (NYSE, NASDAQ, AMEX, or CRYPTO)"})
        if ex_norm not in _EXCHANGES:
            return json.dumps({"error": f"invalid exchange: {exchange}; use NYSE, NASDAQ, AMEX, or CRYPTO"})

    i = _get_store().signal_create(name, signal_type, pdct, ov, ticker_scope=scope, exchange=ex_norm)
    return json.dumps(
        {
            "id": i,
            "name": name,
            "signal_type": signal_type,
            "ticker_scope": scope,
            "exchange": ex_norm,
        }
    )


@mcp.tool()
def list_signals() -> str:
    """List configured signals (includes `id` for `delete_signal` / `run_scan`)."""
    rows = _get_store().signal_list()
    out = [
        {
            "id": r.id,
            "name": r.name,
            "signal_type": r.signal_type,
            "params": r.params,
            "ticker_scope": r.ticker_scope,
            "ticker_overrides": r.ticker_overrides,
            "exchange": r.exchange,
            "enabled": r.enabled,
        }
        for r in rows
    ]
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def delete_signal(signal_id: int) -> str:
    """Delete a signal and its alert history. `signal_id` is the integer `id` from `list_signals`."""
    ok = _get_store().signal_delete(signal_id)
    return json.dumps({"ok": ok})


@mcp.tool()
def run_scan(
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Run an on-demand scan and return only triggered rows.

    Universe — pass **at most one** of:
    - omit `symbol`, `tickers`, and `exchange`: each enabled signal uses its saved scope / overrides / exchange;
    - `symbol`: single ticker string (e.g. `NVDA`);
    - `tickers`: list of ticker strings;
    - `exchange`: `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (full Yahoo screener list for equities; fixed crypto list for `CRYPTO`).

    `signal_id`: optional; when set, only that **enabled** signal runs (ignored when `all_signal_types` is true).
    `all_signal_types`: when true together with exactly one universe selector above, run **every** catalog
    signal type against that universe (`signal_id` must be omitted); uses 1 year of daily bars per symbol.
    """
    pr = _get_provider()
    tick_list: list[str] | None = None
    has_sym = bool(symbol and symbol.strip())
    has_tix = tickers is not None
    has_ex = bool(exchange and exchange.strip())
    if sum(1 for x in (has_sym, has_tix, has_ex) if x) > 1:
        return json.dumps({"error": "pass at most one of symbol, tickers, or exchange"})
    if symbol:
        tick_list = [symbol.strip().upper()]
    if tickers is not None:
        tick_list = [str(x).upper() for x in tickers if str(x).strip()]
    if exchange:
        exu = exchange.strip().upper()
        if exu not in _EXCHANGES:
            return json.dumps({"error": f"invalid exchange: {exchange}; use NYSE, NASDAQ, AMEX, or CRYPTO"})
        try:
            tick_list = fetch_exchange_tickers(exu)
        except ValueError as e:
            return json.dumps({"error": str(e)})
        if not tick_list:
            return json.dumps({"error": "no symbols returned for exchange", "exchange": exu})

    if all_signal_types:
        if signal_id is not None:
            return json.dumps({"error": "signal_id cannot be used with all_signal_types"})
        if not tick_list:
            return json.dumps({"error": "all_signal_types requires symbol, tickers, or exchange"})

        triggered_results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        checked = 0
        for sym in tick_list:
            try:
                df = pr.get_history(sym, period="1y", interval="1d")
            except Exception as e:  # noqa: BLE001
                errors.append({"symbol": sym, "error": str(e)})
                continue
            if df is None or df.empty:
                errors.append({"symbol": sym, "error": "no_history"})
                continue

            for signal_type in CATALOG:
                checked += 1
                params = merge_params(signal_type, {})
                asig = ActiveSignal(
                    id=0,
                    name=signal_type,
                    signal_type=signal_type,
                    params=params,
                    ticker_overrides=[sym],
                )
                trig, det = evaluate(asig, df)
                if trig:
                    triggered_results.append(
                        {
                            "signal_type": signal_type,
                            "name": signal_type,
                            "symbol": sym,
                            "params": params,
                            "triggered": True,
                            "details": det,
                        }
                    )
        return json.dumps(
            {
                "symbols": tick_list,
                "exchange": exchange.strip().upper() if exchange and exchange.strip() else None,
                "mode": "all_signal_types",
                "results": triggered_results,
                "count": len(triggered_results),
                "triggered_count": len(triggered_results),
                "checked_count": checked,
                "errors": errors,
            },
            indent=2,
            default=str,
        )

    st = _get_store()
    results: list[dict[str, Any]] = []
    watch_syms = [w.symbol for w in st.watchlist_get()]
    srows = st.signal_list()
    if signal_id is not None:
        srows = [r for r in srows if r.id == signal_id and r.enabled]
    else:
        srows = [r for r in srows if r.enabled]
    for srow in srows:
        tix = tick_list if tick_list is not None else _resolve_signal_universe(srow, watch_syms)
        if not tix:
            continue
        asig = ActiveSignal(
            id=srow.id,
            name=srow.name,
            signal_type=srow.signal_type,
            params=srow.params,
            ticker_overrides=srow.ticker_overrides,
        )
        for sym in tix:
            try:
                df = pr.get_history(sym, period="1y", interval="1d")
            except Exception:  # noqa: BLE001
                continue
            if df is None or df.empty:
                continue
            try:
                trig, det = evaluate(asig, df)
            except Exception as e:  # noqa: BLE001
                results.append({"symbol": sym, "error": str(e), "signal_id": srow.id})
                continue
            if trig:
                results.append(
                    {
                        "signal_id": srow.id,
                        "name": srow.name,
                        "symbol": sym,
                        "triggered": True,
                        "details": det,
                    }
                )
    return json.dumps({"results": results, "count": len(results)}, indent=2, default=str)


@mcp.tool()
def add_to_watchlist(symbols: list[str]) -> str:
    """`add_to_watchlist`: append ticker strings supplied in `symbols` to the global watchlist.

    `symbols` is a native Python list of tickers (`list[str]`); under MCP structured tool calls this is encoded as a JSON array.
    Returns JSON text: `{"added": [...]}` where `added` lists tickers inserted in this invocation (symbols already stored are omitted).
    """
    added = _get_store().watchlist_add([str(x) for x in symbols])
    return json.dumps({"added": added})


@mcp.tool()
def remove_from_watchlist(symbols: list[str]) -> str:
    """Remove symbols from the watchlist (`symbols`: array of tickers)."""
    n = _get_store().watchlist_remove([str(x) for x in symbols])
    return json.dumps({"removed": n})


@mcp.tool()
def get_watchlist() -> str:
    """Return the current watchlist as a JSON array of ticker strings."""
    w = [x.symbol for x in _get_store().watchlist_get()]
    return json.dumps(w, indent=2)


@mcp.tool()
def chart_price_history(
    symbol: str = "SPY",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    interval: Annotated[str, Field(description="yfinance bar size (e.g. 1d, 1h); intraday couples to allowed period ranges")] = "1d",
    show_ma: Annotated[bool, Field(description="Overlay a simple moving average.")] = False,
    ma_period: Annotated[int, Field(description="SMA period used when show_ma is true.")] = 50,
    show_bollinger_bands: Annotated[bool, Field(description="Overlay Bollinger Bands.")] = False,
    bb_period: Annotated[int, Field(description="Bollinger Band moving-average period.")] = 20,
    bb_std: Annotated[float, Field(description="Bollinger Band standard-deviation multiplier.")] = 2.0,
    show_ema: Annotated[bool, Field(description="Overlay an exponential moving average.")] = False,
    ema_period: Annotated[int, Field(description="EMA period used when show_ema is true.")] = 21,
    show_ma_cloud: Annotated[bool, Field(description="Overlay a filled SMA cloud between fast and slow averages.")] = False,
    ma_cloud_fast: Annotated[int, Field(description="Fast SMA period used for the MA cloud.")] = 50,
    ma_cloud_slow: Annotated[int, Field(description="Slow SMA period used for the MA cloud.")] = 200,
    show_fib_retracement: Annotated[bool, Field(description="Overlay Fibonacci retracement levels from the visible high/low swing.")] = False,
    show_avwap: Annotated[bool, Field(description="Overlay anchored VWAP from the first visible bar, or avwap_anchor when provided.")] = False,
    avwap_anchor: Annotated[str | None, Field(description="Optional aVWAP anchor date/time parseable by pandas, for example 2025-01-02.")] = None,
    pe_subchart: Annotated[
        bool,
        Field(
            description=(
                "When true, add a lower panel for P/E vs EPS: quarterly TTM where Yahoo provides "
                "enough quarters, otherwise fiscal-year Diluted EPS (Yahoo caps statement history; "
                "ETFs often have no EPS)."
            )
        ),
    ] = False,
) -> Image | str:
    """Candlestick price history chart. Returns PNG image; on failure JSON text with `error`."""
    return _chart_tool_result(
        "price_history",
        {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "show_ma": show_ma,
            "ma_period": ma_period,
            "show_bollinger_bands": show_bollinger_bands,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "show_ema": show_ema,
            "ema_period": ema_period,
            "show_ma_cloud": show_ma_cloud,
            "ma_cloud_fast": ma_cloud_fast,
            "ma_cloud_slow": ma_cloud_slow,
            "show_fib_retracement": show_fib_retracement,
            "show_avwap": show_avwap,
            "avwap_anchor": avwap_anchor,
            "pe_subchart": pe_subchart,
        },
    )


@mcp.tool()
def chart_price_overlay(
    symbols: list[str] | None = None,
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    normalize: bool = True,
) -> Image | str:
    """Multi-symbol line overlay (default symbols SPY and QQQ if `symbols` omitted). Normalized to 100 when `normalize`."""
    p: dict[str, Any] = {"period": period, "normalize": normalize}
    if symbols is not None:
        p["symbols"] = symbols
    return _chart_tool_result("price_overlay", p)


@mcp.tool()
def chart_forward_returns(
    symbol: str = "SPY",
    event_type: Literal[
        "golden_cross",
        "macd_bullish_crossover",
        "pct_from_ma",
        "rsi_oversold",
        "rsi_overbought",
    ] = "rsi_oversold",
    windows: list[int] | None = None,
    event_params: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional event detector parameters. For pct_from_ma, use "
                '{"ma_type":"ema","ma_period":200,"pct":3}.'
            )
        ),
    ] = None,
) -> Image | str:
    """Price chart with signal markers plus a forward-return table after historical signal events.

    Uses ~10y daily history; `period` is not configurable.
    `windows`: forward horizons in **trading bars** along the daily close series
    (default 5, 10, 21, 42, 63, 126, 252; about 1-6 and 12 months).
    Supported `event_type`: rsi_oversold, rsi_overbought, golden_cross,
    macd_bullish_crossover, pct_from_ma.
    `event_params`: optional event-specific detector parameters. For `pct_from_ma`,
    supported keys are `ma_type` ("sma" or "ema"), `ma_period`, and `pct`.
    """
    p: dict[str, Any] = {"symbol": symbol, "event_type": event_type}
    if windows is not None:
        p["windows"] = windows
    if event_params is not None:
        p["event_params"] = event_params
    return _chart_tool_result("forward_returns", p)


@mcp.tool()
def chart_drawdown_comparison(
    symbols: list[str] | None = None,
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "5y",
) -> Image | str:
    """Underwater (drawdown %) chart vs running high. Default symbols: ^GSPC and QQQ."""
    p: dict[str, Any] = {"period": period}
    if symbols is not None:
        p["symbols"] = symbols
    return _chart_tool_result("drawdown_comparison", p)


@mcp.tool()
def chart_log_cycle(
    symbol: str = "BTC-USD",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "max",
) -> Image | str:
    """Weekly log10(close) line chart for long-horizon inspection."""
    return _chart_tool_result("log_cycle", {"symbol": symbol, "period": period})


@mcp.resource("signals://triggered", mime_type="application/json")
def resource_triggered() -> str:
    """Last 50 fired alerts (JSON)."""
    rows = _get_store().alerts_recent(50)
    o = [
        {
            "id": r.id,
            "signal_id": r.signal_id,
            "symbol": r.symbol,
            "triggered_at": r.triggered_at,
            "details": r.details,
        }
        for r in rows
    ]
    return json.dumps(o, default=str, indent=2)


@mcp.resource("signals://watchlist", mime_type="application/json")
def resource_watchlist() -> str:
    """Active watchlist tickers (JSON)."""
    return json.dumps([w.symbol for w in _get_store().watchlist_get()])


@mcp.resource("research://forward-returns/{symbol}/{event_type}", mime_type="text/markdown")
def resource_forward_returns(symbol: str, event_type: str) -> str:
    """Markdown summary of mean/median forward returns (7/30/90 trading days) after RSI events.

    URI path `symbol`: yfinance ticker. `event_type`: `rsi_oversold` or `rsi_overbought` (same as `chart_forward_returns`).
    """
    return forward_returns_markdown(_get_provider(), symbol, event_type)


def main() -> None:
    """Configure logging and run the FastMCP stdio server."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("SCANNER_MCP_LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    log.info("Starting ScannerMCP from %s", __file__)

    def _handle_stop(signum: int, _: Any) -> None:
        log.info("Received signal %s, shutting down", signum)
        _shutdown_scheduler()
        logging.shutdown()
        os._exit(128 + signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    try:
        mcp.run()
    finally:
        _shutdown_scheduler()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


if __name__ == "__main__":
    main()

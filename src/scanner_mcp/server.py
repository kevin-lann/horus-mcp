"""FastMCP entry point: market tools, signals, charts, resources."""

from __future__ import annotations

import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from io import StringIO
from typing import Any, AsyncIterator

from apscheduler.schedulers.base import BaseScheduler
from fastmcp import FastMCP

from scanner_mcp.charts import generator as chartgen
from scanner_mcp.data.movers import screen_movers
from scanner_mcp.data.provider import YFinanceProvider
from scanner_mcp.db.store import Store
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


@mcp.tool()
def debug_quote(symbol: str) -> str:
    """Debug yfinance quote fields and history fallback for one symbol."""
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
    """Current / last price, day change, volume, and market cap from yfinance fast_info."""
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
    """Compute indicators with buy/hold/sell rating each plus consensus. Names: rsi, rsi:14, macd, bbands, sma:50, ema:20, ath_distance, beta."""
    r = _compute_indicators(symbol, indicators, period)
    return json.dumps(r, indent=2, default=str)


@mcp.tool()
def get_ath_distance(symbol: str) -> str:
    """Percent distance of last close from all-time high (over visible history)."""
    pr = _get_provider()
    df = pr.get_history(symbol, period="max", interval="1d")
    if df.empty:
        return json.dumps({"error": "no data"})
    ind = Indicators(df)
    d = ind.ath_distance()
    return json.dumps({"symbol": symbol.upper(), "pct_from_ath": d})


@mcp.tool()
def get_option_chain(symbol: str, expiry: str | None = None) -> str:
    """Options chain: calls and puts table as text (use expiry YYYY-MM-DD or first listed)."""
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
    """Major US indices, ETFs, crypto, VIX: price and day % change."""
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
    """Top daily gainers for exchange: NYSE, NASDAQ, AMEX, or CRYPTO."""
    try:
        rows = screen_movers("gainers", exchange, limit=limit)
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


@mcp.tool()
def top_losers(exchange: str, limit: int = 20) -> str:
    """Top daily losers for exchange: NYSE, NASDAQ, AMEX, or CRYPTO."""
    try:
        rows = screen_movers("losers", exchange, limit=limit)
        return json.dumps(rows, default=str, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


@mcp.tool()
def list_signal_catalog() -> str:
    """List predefined signal types and default parameters."""
    return json.dumps(list_catalog_entries(), indent=2, default=str)


@mcp.tool()
def create_signal(
    name: str,
    signal_type: str,
    params: str = "{}",
    ticker_overrides: str | None = None,
) -> str:
    """Create a persisted enabled signal.

    `params` must be a JSON object merged with catalog defaults.
    `ticker_overrides` may be a JSON list; when omitted, scans use the global
    watchlist for this signal.
    """
    if signal_type not in CATALOG:
        return json.dumps({"error": f"invalid signal_type: {signal_type}"})
    try:
        pdct = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params JSON: {e}"})
    merge_params(signal_type, pdct)  # validate
    ov: list[str] | None = None
    if ticker_overrides:
        try:
            o = json.loads(ticker_overrides)
            if isinstance(o, list):
                ov = [str(x).upper() for x in o]
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"ticker_overrides: {e}"})
    i = _get_store().signal_create(name, signal_type, pdct, ov)
    return json.dumps({"id": i, "name": name, "signal_type": signal_type})


@mcp.tool()
def list_signals() -> str:
    """List configured signals."""
    rows = _get_store().signal_list()
    out = [
        {
            "id": r.id,
            "name": r.name,
            "signal_type": r.signal_type,
            "params": r.params,
            "ticker_overrides": r.ticker_overrides,
            "enabled": r.enabled,
        }
        for r in rows
    ]
    return json.dumps(out, indent=2, default=str)


@mcp.tool()
def delete_signal(signal_id: int) -> str:
    """Delete a signal and its alert history."""
    ok = _get_store().signal_delete(signal_id)
    return json.dumps({"ok": ok})


@mcp.tool()
def run_scan(
    signal_id: int | None = None,
    tickers: str | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
) -> str:
    """Run an on-demand scan and return only triggered results.

    When `tickers` is omitted, each signal uses its own ticker overrides or the
    global watchlist. When `signal_id` is set, only that enabled signal is
    evaluated. Set `all_signal_types` with `symbol` or `tickers` to evaluate
    every catalog signal type and return the triggered matches.
    """
    pr = _get_provider()
    tick_list: list[str] | None = None
    if symbol and tickers:
        return json.dumps({"error": "pass either symbol or tickers, not both"})
    if symbol:
        tick_list = [symbol.upper()]
    if tickers:
        try:
            t = json.loads(tickers)
            if isinstance(t, list):
                tick_list = [str(x).upper() for x in t if str(x).strip()]
            else:
                return json.dumps({"error": "tickers must be a JSON array of strings"})
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "error": f"tickers JSON: {e}",
                    "hint": "Use valid JSON with double quotes, e.g. [\"AAPL\", \"MSFT\"].",
                }
            )

    if all_signal_types:
        if signal_id is not None:
            return json.dumps({"error": "signal_id cannot be used with all_signal_types"})
        if not tick_list:
            return json.dumps({"error": "all_signal_types requires symbol or tickers"})

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
    srows = st.signal_list()
    if signal_id is not None:
        srows = [r for r in srows if r.id == signal_id and r.enabled]
    else:
        srows = [r for r in srows if r.enabled]
    for srow in srows:
        tix = tick_list or srow.ticker_overrides or [w.symbol for w in st.watchlist_get()]
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
def add_to_watchlist(symbols: str) -> str:
    """Add symbols. Pass JSON list e.g. [\"AAPL\",\"MSFT\"]."""
    try:
        arr = json.loads(symbols)
    except json.JSONDecodeError as e:
        return json.dumps({"error": str(e)})
    if not isinstance(arr, list):
        return json.dumps({"error": "expected JSON array of strings"})
    a = [str(x) for x in arr]
    added = _get_store().watchlist_add(a)
    return json.dumps({"added": added})


@mcp.tool()
def remove_from_watchlist(symbols: str) -> str:
    """Remove tickers. JSON list."""
    try:
        arr = json.loads(symbols)
    except json.JSONDecodeError as e:
        return json.dumps({"error": str(e)})
    n = _get_store().watchlist_remove([str(x) for x in arr])
    return json.dumps({"removed": n})


@mcp.tool()
def get_watchlist() -> str:
    """Current watchlist tickers."""
    w = [x.symbol for x in _get_store().watchlist_get()]
    return json.dumps(w, indent=2)


@mcp.tool()
def generate_chart(
    chart_type: str,
    params: str = "{}",
) -> str:
    """Build a chart and return JSON containing image/png base64 data.

    `params` is a JSON object string. Supported types are `price_history`,
    `price_overlay`, `forward_returns`, `drawdown_comparison`, and `log_cycle`.
    """
    pr = _get_provider()
    try:
        p = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"params: {e}"})
    if not isinstance(p, dict):
        return json.dumps({"error": "params must be a JSON object"})
    try:
        r = chartgen.generate_chart(pr, chart_type, p)
        return json.dumps(r)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)})


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
    """Markdown table of mean/median forward returns (7/30/90d) after RSI events."""
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

"""Signal validation and scan execution helpers."""

from __future__ import annotations

import json
from typing import Any

from scanner_mcp.data.exchange_universe import fetch_exchange_tickers
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.signals.catalog import CATALOG, merge_params
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal

EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "CRYPTO"})


def normalize_exchange(exchange: str | None) -> str | None:
    """Normalize an optional exchange string."""
    if exchange and exchange.strip():
        return exchange.strip().upper()
    return None


def resolve_signal_universe(signal_row: SignalRow, watchlist: list[str]) -> list[str]:
    """Tickers to scan for one persisted signal: overrides, watchlist, or exchange list."""
    scope = signal_row.ticker_scope
    if scope == "tickers":
        return list(signal_row.ticker_overrides or [])
    if scope == "exchange":
        if not signal_row.exchange:
            return []
        try:
            return fetch_exchange_tickers(signal_row.exchange)
        except ValueError:
            return []
    return list(watchlist)


def validate_signal_creation(
    name: str,
    signal_type: str,
    params: dict[str, Any] | None,
    ticker_scope: str,
    ticker_overrides: list[str] | None,
    exchange: str | None,
) -> dict[str, Any]:
    """Validate signal creation inputs and return normalized values or an error payload."""
    if signal_type not in CATALOG:
        return {"error": f"invalid signal_type: {signal_type}"}

    normalized_params = dict(params) if params else {}
    merge_params(signal_type, normalized_params)

    exchange_value = normalize_exchange(exchange)
    overrides: list[str] | None = None

    if ticker_scope == "tickers":
        if not ticker_overrides:
            return {"error": "ticker_scope=tickers requires non-empty ticker_overrides list"}
        if exchange_value:
            return {"error": "exchange must be omitted when ticker_scope is tickers"}
        overrides = [str(value).upper() for value in ticker_overrides if str(value).strip()]
        if not overrides:
            return {"error": "ticker_overrides list is empty"}
    elif ticker_scope == "watchlist":
        if ticker_overrides:
            return {"error": "omit ticker_overrides when ticker_scope is watchlist"}
        if exchange_value:
            return {"error": "omit exchange when ticker_scope is watchlist"}
    else:
        if ticker_overrides:
            return {"error": "omit ticker_overrides when ticker_scope is exchange"}
        if not exchange_value:
            return {"error": "ticker_scope=exchange requires exchange (NYSE, NASDAQ, AMEX, or CRYPTO)"}
        if exchange_value not in EXCHANGES:
            return {"error": f"invalid exchange: {exchange}; use NYSE, NASDAQ, AMEX, or CRYPTO"}

    return {
        "name": name,
        "signal_type": signal_type,
        "params": normalized_params,
        "ticker_scope": ticker_scope,
        "ticker_overrides": overrides,
        "exchange": exchange_value,
    }


def create_signal_payload(
    store: Store,
    name: str,
    signal_type: str,
    params: dict[str, Any] | None,
    ticker_scope: str,
    ticker_overrides: list[str] | None,
    exchange: str | None,
) -> str:
    """Create a persisted signal or return a JSON error payload."""
    validated = validate_signal_creation(name, signal_type, params, ticker_scope, ticker_overrides, exchange)
    if "error" in validated:
        return json.dumps(validated)
    signal_id = store.signal_create(
        validated["name"],
        validated["signal_type"],
        validated["params"],
        validated["ticker_overrides"],
        ticker_scope=validated["ticker_scope"],
        exchange=validated["exchange"],
    )
    return json.dumps(
        {
            "id": signal_id,
            "name": validated["name"],
            "signal_type": validated["signal_type"],
            "ticker_scope": validated["ticker_scope"],
            "exchange": validated["exchange"],
        }
    )


def run_scan_payload(
    store: Store,
    provider: DataProvider,
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Run an on-demand scan and return the JSON payload."""
    tick_list: list[str] | None = None
    has_symbol = bool(symbol and symbol.strip())
    has_tickers = tickers is not None
    has_exchange = bool(exchange and exchange.strip())
    if sum(1 for value in (has_symbol, has_tickers, has_exchange) if value) > 1:
        return json.dumps({"error": "pass at most one of symbol, tickers, or exchange"})
    if symbol:
        tick_list = [symbol.strip().upper()]
    if tickers is not None:
        tick_list = [str(value).upper() for value in tickers if str(value).strip()]
    if exchange:
        exchange_value = normalize_exchange(exchange)
        if exchange_value not in EXCHANGES:
            return json.dumps({"error": f"invalid exchange: {exchange}; use NYSE, NASDAQ, AMEX, or CRYPTO"})
        try:
            tick_list = fetch_exchange_tickers(exchange_value)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        if not tick_list:
            return json.dumps({"error": "no symbols returned for exchange", "exchange": exchange_value})

    if all_signal_types:
        if signal_id is not None:
            return json.dumps({"error": "signal_id cannot be used with all_signal_types"})
        if not tick_list:
            return json.dumps({"error": "all_signal_types requires symbol, tickers, or exchange"})

        triggered_results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        checked = 0
        for ticker in tick_list:
            try:
                df = provider.get_history(ticker, period="1y", interval="1d")
            except Exception as exc:  # noqa: BLE001
                errors.append({"symbol": ticker, "error": str(exc)})
                continue
            if df is None or df.empty:
                errors.append({"symbol": ticker, "error": "no_history"})
                continue

            for signal_type in CATALOG:
                checked += 1
                params = merge_params(signal_type, {})
                signal = ActiveSignal(
                    id=0,
                    name=signal_type,
                    signal_type=signal_type,
                    params=params,
                    ticker_overrides=[ticker],
                )
                triggered, details = evaluate(signal, df)
                if triggered:
                    triggered_results.append(
                        {
                            "signal_type": signal_type,
                            "name": signal_type,
                            "symbol": ticker,
                            "params": params,
                            "triggered": True,
                            "details": details,
                        }
                    )
        return json.dumps(
            {
                "symbols": tick_list,
                "exchange": normalize_exchange(exchange),
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

    results: list[dict[str, Any]] = []
    watch_symbols = [row.symbol for row in store.watchlist_get()]
    signal_rows = store.signal_list()
    if signal_id is not None:
        signal_rows = [row for row in signal_rows if row.id == signal_id and row.enabled]
    else:
        signal_rows = [row for row in signal_rows if row.enabled]
    for signal_row in signal_rows:
        universe = tick_list if tick_list is not None else resolve_signal_universe(signal_row, watch_symbols)
        if not universe:
            continue
        signal = ActiveSignal(
            id=signal_row.id,
            name=signal_row.name,
            signal_type=signal_row.signal_type,
            params=signal_row.params,
            ticker_overrides=signal_row.ticker_overrides,
        )
        for ticker in universe:
            try:
                df = provider.get_history(ticker, period="1y", interval="1d")
            except Exception:  # noqa: BLE001
                continue
            if df is None or df.empty:
                continue
            try:
                triggered, details = evaluate(signal, df)
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": ticker, "error": str(exc), "signal_id": signal_row.id})
                continue
            if triggered:
                results.append(
                    {
                        "signal_id": signal_row.id,
                        "name": signal_row.name,
                        "symbol": ticker,
                        "triggered": True,
                        "details": details,
                    }
                )
    return json.dumps({"results": results, "count": len(results)}, indent=2, default=str)

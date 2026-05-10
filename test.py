#!/usr/bin/env python3
"""Local smoke-test runner for scanner-mcp tools.

Examples:
  python3 test.py --tool price --symbol SPY
  python3 test.py --tool indicators --symbol AAPL
  python3 test.py --tool chart --chart-type price_history --symbol SPY
  python3 test.py --tool all --symbol SPY
  python3 test.py --tool create_signal --mutate --signal-type rsi_oversold
  python3 test.py --tool delete_signal --mutate --signal-id 1
  python3 test.py --tool watchlist --mutate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastmcp.utilities.types import Image  # noqa: E402
from scanner_mcp import server  # noqa: E402


ToolFn = Callable[[], Any]
DEFAULT_SYMBOLS = ["SPY", "QQQ"]


def _parse_jsonish(value: str) -> Any:
    """Parse JSON output when possible, otherwise keep raw text."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _compact_chart_payload(payload: Any) -> Any:
    """Replace large base64 chart data with a readable terminal summary."""
    if not isinstance(payload, dict):
        return payload
    data = payload.get("data")
    if isinstance(data, str):
        payload = dict(payload)
        payload["data"] = f"{data[:80]}... ({len(data)} base64 chars)"
    return payload


def _print_result(name: str, raw: Any) -> None:
    """Pretty-print a tool response in a consistent local-test format."""
    print(f"\n=== {name} ===")
    if isinstance(raw, Image):
        sz = len(raw.data) if raw.data is not None else 0
        print(json.dumps({"type": "image/png", "bytes": sz}, indent=2))
        return
    if not isinstance(raw, str):
        print(str(raw)[:4000])
        return
    parsed = _parse_jsonish(raw)
    parsed = _compact_chart_payload(parsed)
    if isinstance(parsed, (dict, list)):
        print(json.dumps(parsed, indent=2, default=str))
    else:
        print(str(parsed)[:4000])


def _symbols(args: argparse.Namespace) -> list[str]:
    """Return CLI symbols or the script's default multi-symbol sample."""
    return args.symbols or DEFAULT_SYMBOLS


def _run(name: str, fn: ToolFn) -> None:
    """Run one tool wrapper and print either its result or raised exception."""
    try:
        _print_result(name, fn())
    except Exception as exc:  # noqa: BLE001
        print(f"\n=== {name} ===")
        print(f"ERROR: {exc}")


def _chart_tool_call(args: argparse.Namespace) -> Image | str:
    """Dispatch to typed chart tools; optional --chart-params merges JSON overrides into defaults."""
    ct = args.chart_type
    if ct == "price_history":
        kw: dict[str, Any] = {
            "symbol": args.symbol,
            "period": args.period,
            "interval": args.interval,
        }
        if args.chart_params:
            kw.update(json.loads(args.chart_params))
        return server.chart_price_history(**kw)
    if ct == "price_overlay":
        kw = {"symbols": _symbols(args), "period": args.period, "normalize": True}
        if args.chart_params:
            kw.update(json.loads(args.chart_params))
        return server.chart_price_overlay(**kw)
    if ct == "forward_returns":
        kw = {
            "symbol": args.symbol,
            "event_type": "rsi_oversold",
            "windows": [7, 30, 90],
        }
        if args.chart_params:
            kw.update(json.loads(args.chart_params))
        return server.chart_forward_returns(**kw)
    if ct == "drawdown_comparison":
        kw = {"symbols": _symbols(args), "period": args.period}
        if args.chart_params:
            kw.update(json.loads(args.chart_params))
        return server.chart_drawdown_comparison(**kw)
    if ct == "log_cycle":
        kw = {"symbol": "BTC-USD", "period": "max"}
        if args.chart_params:
            kw.update(json.loads(args.chart_params))
        return server.chart_log_cycle(**kw)
    raise ValueError(f"unknown chart_type: {ct}")


def run_price(args: argparse.Namespace) -> None:
    _run("get_price", lambda: server.get_price(args.symbol))


def run_debug_quote(args: argparse.Namespace) -> None:
    _run("debug_quote", lambda: server.debug_quote(args.symbol))


def run_indicators(args: argparse.Namespace) -> None:
    _run(
        "get_indicators",
        lambda: server.get_indicators(
            args.symbol,
            ["rsi", "macd", "bbands", "sma:50", "ema:20", "ath_distance", "beta"],
            args.period,
        ),
    )


def run_ath(args: argparse.Namespace) -> None:
    _run("get_ath_distance", lambda: server.get_ath_distance(args.symbol))


def run_options(args: argparse.Namespace) -> None:
    _run("get_option_chain", lambda: server.get_option_chain(args.symbol, args.expiry))


def run_snapshot(_: argparse.Namespace) -> None:
    _run("market_snapshot", server.market_snapshot)


def run_movers(args: argparse.Namespace) -> None:
    _run("top_gainers", lambda: server.top_gainers(args.exchange, args.limit))
    _run("top_losers", lambda: server.top_losers(args.exchange, args.limit))


def run_catalog(_: argparse.Namespace) -> None:
    _run("list_signal_catalog", server.list_signal_catalog)


def run_signals(_: argparse.Namespace) -> None:
    _run("list_signals", server.list_signals)


def run_create_signal(args: argparse.Namespace) -> None:
    if not args.mutate:
        print("\nPass --mutate to create a persisted signal in the local SQLite DB.")
        return
    if args.signal_tickers and args.signal_exchange:
        print("\nUse only one of --signal-tickers or --signal-exchange.")
        return
    if args.signal_tickers:
        ticker_scope = "tickers"
        ticker_overrides = _symbols(args)
        sig_exchange = None
    elif args.signal_exchange:
        ticker_scope = "exchange"
        ticker_overrides = None
        sig_exchange = args.signal_exchange
    else:
        ticker_scope = "watchlist"
        ticker_overrides = None
        sig_exchange = None
    raw_params = args.signal_params.strip()
    sig_params_obj: dict[str, Any] | None
    try:
        sig_params_obj = json.loads(raw_params) if raw_params else {}
    except json.JSONDecodeError as exc:
        print(f"\nInvalid --signal-params JSON: {exc}")
        return
    _run(
        "create_signal",
        lambda: server.create_signal(
            args.signal_name,
            args.signal_type,
            sig_params_obj,
            ticker_scope=ticker_scope,
            ticker_overrides=ticker_overrides,
            exchange=sig_exchange,
        ),
    )
    _run("list_signals", server.list_signals)


def run_delete_signal(args: argparse.Namespace) -> None:
    if not args.mutate:
        print("\nPass --mutate to delete a persisted signal from the local SQLite DB.")
        return
    if args.signal_id is None:
        print("\nPass --signal-id ID to choose which signal to delete.")
        _run("list_signals", server.list_signals)
        return
    _run("delete_signal", lambda: server.delete_signal(args.signal_id))
    _run("list_signals", server.list_signals)


def run_watchlist(args: argparse.Namespace) -> None:
    _run("get_watchlist", server.get_watchlist)
    if not args.mutate:
        print("\nPass --mutate to also test add_to_watchlist/remove_from_watchlist.")
        return
    syms = _symbols(args)
    _run("add_to_watchlist", lambda: server.add_to_watchlist(syms))
    _run("get_watchlist", server.get_watchlist)
    _run("remove_from_watchlist", lambda: server.remove_from_watchlist(syms))


def run_scan(args: argparse.Namespace) -> None:
    """Run scan without ticker overrides unless --symbols or --scan-exchange was provided."""
    ex = args.scan_exchange
    _run(
        "run_scan",
        lambda: server.run_scan(
            tickers=args.symbols,
            exchange=ex,
        ),
    )


def run_chart(args: argparse.Namespace) -> None:
    _run(args.chart_type, lambda: _chart_tool_call(args))
    print(f"\nDebug PNGs are written to: {ROOT / 'output'}")


def run_all(args: argparse.Namespace) -> None:
    run_price(args)
    run_indicators(args)
    run_ath(args)
    run_catalog(args)
    run_chart(args)


def build_parser() -> argparse.ArgumentParser:
    """Create the local smoke-test CLI parser."""
    parser = argparse.ArgumentParser(description="Local scanner-mcp tool smoke tests")
    parser.add_argument(
        "--tool",
        choices=[
            "all",
            "price",
            "debug_quote",
            "indicators",
            "ath",
            "options",
            "snapshot",
            "movers",
            "catalog",
            "signals",
            "create_signal",
            "delete_signal",
            "watchlist",
            "scan",
            "chart",
        ],
        default="all",
        help="Tool or tool group to test",
    )
    parser.add_argument("--symbol", default="SPY", help="Primary ticker symbol")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Ticker list for multi-symbol tools. For scan, omitted means use each persisted signal's tickers/watchlist.",
    )
    parser.add_argument("--period", default="1mo", help="History period for applicable tools")
    parser.add_argument("--interval", default="1d", help="Chart/history interval")
    parser.add_argument("--expiry", default=None, help="Option expiry YYYY-MM-DD")
    parser.add_argument("--exchange", default="NASDAQ", help="Mover exchange: NYSE, NASDAQ, AMEX, or CRYPTO")
    parser.add_argument(
        "--scan-exchange",
        default=None,
        metavar="NYSE|NASDAQ|AMEX|CRYPTO",
        help="With scan: run on full Yahoo screener universe for that venue",
    )
    parser.add_argument("--signal-name", default="local test signal", help="Name to use with --tool create_signal")
    parser.add_argument("--signal-id", type=int, default=None, help="Signal ID for --tool delete_signal")
    parser.add_argument(
        "--signal-type",
        default="rsi_oversold",
        help="Signal type to create, e.g. rsi_oversold, pct_from_ath, macd_bullish_crossover",
    )
    parser.add_argument("--signal-params", default="{}", help="JSON params for --tool create_signal")
    parser.add_argument(
        "--signal-tickers",
        action="store_true",
        help="Use --symbols as ticker_overrides (ticker_scope=tickers)",
    )
    parser.add_argument(
        "--signal-exchange",
        default=None,
        metavar="NYSE|NASDAQ|AMEX|CRYPTO",
        help="With create_signal: scan full exchange instead of watchlist (cannot combine with --signal-tickers)",
    )
    parser.add_argument(
        "--chart-type",
        choices=[
            "price_history",
            "price_overlay",
            "forward_returns",
            "drawdown_comparison",
            "log_cycle",
        ],
        default="price_history",
    )
    parser.add_argument(
        "--chart-params",
        default=None,
        help="Optional JSON object merged into defaults for the selected --chart-type",
    )
    parser.add_argument("--mutate", action="store_true", help="Allow tests that modify the local SQLite DB")
    return parser


def main() -> int:
    """Dispatch the selected local smoke-test command."""
    args = build_parser().parse_args()
    runners: dict[str, Callable[[argparse.Namespace], None]] = {
        "all": run_all,
        "price": run_price,
        "debug_quote": run_debug_quote,
        "indicators": run_indicators,
        "ath": run_ath,
        "options": run_options,
        "snapshot": run_snapshot,
        "movers": run_movers,
        "catalog": run_catalog,
        "signals": run_signals,
        "create_signal": run_create_signal,
        "delete_signal": run_delete_signal,
        "watchlist": run_watchlist,
        "scan": run_scan,
        "chart": run_chart,
    }
    runners[args.tool](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

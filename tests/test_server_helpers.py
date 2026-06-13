from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import patch

import pandas as pd

from scanner_mcp import server
from scanner_mcp.charts import service as chart_service
from scanner_mcp.market import service as market_service
from scanner_mcp.signals import service as signals_service


class FakeProvider:
    def __init__(self, fast_info: dict | None = None, history: pd.DataFrame | None = None) -> None:
        self.fast_info = fast_info or {}
        self.history = history if history is not None else pd.DataFrame()

    def get_fast_info(self, _symbol: str) -> dict:
        return self.fast_info

    def get_history(self, _symbol: str, *, period: str, interval: str) -> pd.DataFrame:
        return self.history


class ServerHelpersTest(unittest.TestCase):
    def test_parse_ind_and_float_helpers(self) -> None:
        self.assertEqual(market_service.parse_indicator("rsi:10"), ("rsi", {"period": 10}, None))
        self.assertEqual(market_service.parse_indicator("ema:21"), ("ema", {"period": 21}, None))
        self.assertEqual(market_service.parse_indicator("macd"), ("macd", {}, None))
        self.assertEqual(market_service.parse_indicator("rsi:bad"), ("rsi", {}, "invalid indicator period for rsi:bad"))
        self.assertEqual(market_service.as_float("12.5"), 12.5)
        self.assertIsNone(market_service.as_float(float("nan")))
        self.assertIsNone(market_service.as_float("bad"))

    def test_quote_snapshot_uses_fast_info_then_history_fallback(self) -> None:
        provider = FakeProvider({"last_price": 110.0, "previous_close": 100.0})
        self.assertEqual(market_service.quote_snapshot(provider, "SPY")["day_change_pct"], 10.0)  # type: ignore[arg-type]

        hist = pd.DataFrame({"Close": [100.0, 105.0]})
        provider = FakeProvider({}, hist)
        out = market_service.quote_snapshot(provider, "SPY")  # type: ignore[arg-type]

        self.assertEqual(out["last_price"], 105.0)
        self.assertEqual(out["previous_close"], 100.0)
        self.assertEqual(out["day_change_pct"], 5.0)

    def test_compute_indicators_handles_unknowns_and_consensus(self) -> None:
        df = pd.DataFrame(
            {
                "High": [float(x) for x in range(1, 35)],
                "Close": [float(x) for x in range(1, 35)],
            }
        )
        provider = FakeProvider(history=df)

        out = market_service.compute_indicators(provider, "spy", ["rsi:3", "sma:3", "unknown"], "6mo")

        self.assertEqual(out["symbol"], "SPY")
        self.assertIn("rsi_3", out)
        self.assertIn("sma_3", out)
        self.assertEqual(out["unknown"]["error"], "unknown indicator: unknown")
        self.assertIn(out["consensus"], {"buy", "hold", "sell"})

    def test_compute_indicators_reports_invalid_period_per_indicator(self) -> None:
        df = pd.DataFrame({"High": [1.0] * 20, "Close": [1.0] * 20})
        provider = FakeProvider(history=df)

        out = market_service.compute_indicators(provider, "spy", ["rsi:0", "ema:bad"], "6mo")

        self.assertEqual(out["rsi:0"]["error"], "invalid indicator period for rsi:0")
        self.assertEqual(out["ema:bad"]["error"], "invalid indicator period for ema:bad")
        self.assertEqual(out["consensus"], "hold")

    def test_chart_tool_result_returns_json_error_for_bad_chart_response(self) -> None:
        with (
            patch("scanner_mcp.charts.service.generate_chart", return_value={"mime": "image/png"}),
        ):
            self.assertEqual(
                json.loads(chart_service.chart_tool_result(object(), "x", {}))["error"],
                "chart response missing image data",
            )

        with (
            patch("scanner_mcp.charts.service.generate_chart", side_effect=RuntimeError("boom")),
        ):
            self.assertEqual(json.loads(chart_service.chart_tool_result(object(), "x", {}))["error"], "boom")

    def test_chart_tool_result_rejects_invalid_or_empty_base64(self) -> None:
        with patch("scanner_mcp.charts.service.generate_chart", return_value={"mime": "image/png", "data": "%%%"}):
            self.assertEqual(
                json.loads(chart_service.chart_tool_result(object(), "x", {}))["error"],
                "chart response missing image data",
            )

        with patch(
            "scanner_mcp.charts.service.generate_chart",
            return_value={"mime": "image/png", "data": base64.b64encode(b"").decode("ascii")},
        ):
            self.assertEqual(
                json.loads(chart_service.chart_tool_result(object(), "x", {}))["error"],
                "chart response missing image data",
            )

    def test_chart_fundamental_overlay_passes_params(self) -> None:
        with patch("scanner_mcp.server.chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(server.chart_fundamental_overlay("msft", "earnings", "annual", "10y", "1d", "line"), "ok")

        chart_tool.assert_called_once()
        self.assertEqual(chart_tool.call_args.args[1:], (
            "fundamental_overlay",
            {
                "symbol": "msft",
                "metric": "earnings",
                "frequency": "annual",
                "period": "10y",
                "interval": "1d",
                "price_style": "line",
            },
        ))

    def test_new_chart_tools_pass_new_params(self) -> None:
        with patch("scanner_mcp.server.chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(server.chart_fundamental_momentum("aapl", "annual", "10y", "1d", "earnings_growth", "line"), "ok")
            self.assertEqual(server.chart_basket_breadth(["AAPL", "MSFT"], "QQQ", "2y", 20, 42), "ok")
            self.assertEqual(server.chart_pairs_spread("ko", "pep", "3y", "price_spread", 30), "ok")

        self.assertEqual(
            chart_tool.call_args_list[0].args[1:],
            (
                "fundamental_momentum",
                {
                    "symbol": "aapl",
                    "frequency": "annual",
                    "period": "10y",
                    "interval": "1d",
                    "profitability_metric": "earnings_growth",
                    "price_style": "line",
                },
            ),
        )
        self.assertEqual(
            chart_tool.call_args_list[1].args[1:],
            (
                "basket_breadth",
                {"symbols": ["AAPL", "MSFT"], "benchmark": "QQQ", "period": "2y", "sma_period": 20, "corr_window": 42},
            ),
        )
        self.assertEqual(
            chart_tool.call_args_list[2].args[1:],
            (
                "pairs_spread",
                {"symbol": "ko", "benchmark": "pep", "period": "3y", "spread_mode": "price_spread", "z_window": 30},
            ),
        )

    def test_new_chart_tools_pass_params(self) -> None:
        with patch("scanner_mcp.server.chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(server.chart_ratio("spy", "xlp", "6mo"), "ok")
            self.assertEqual(server.chart_relative_strength("aapl", "spy", "3y", 30), "ok")
            self.assertEqual(server.chart_sector_rotation(["XLK", "XLF"], "5y", 42), "ok")

        self.assertEqual(chart_tool.call_args_list[0].args[1:], ("ratio_chart", {"symbol": "spy", "benchmark": "xlp", "period": "6mo"}))
        self.assertEqual(
            chart_tool.call_args_list[1].args[1:],
            ("relative_strength", {"symbol": "aapl", "benchmark": "spy", "period": "3y", "ma_period": 30}),
        )
        self.assertEqual(
            chart_tool.call_args_list[2].args[1:],
            ("sector_rotation", {"symbols": ["XLK", "XLF"], "period": "5y", "return_window": 42}),
        )

    def test_chart_forward_returns_passes_signal_dates(self) -> None:
        with patch("scanner_mcp.server.chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(
                server.chart_forward_returns(
                    "spy",
                    "rsi_oversold",
                    [5, 21],
                    {"period": 10, "threshold": 35},
                    ["2024-01-13", "2024-02-10"],
                ),
                "ok",
            )

        self.assertEqual(
            chart_tool.call_args.args[1:],
            (
                "forward_returns",
                {
                    "symbol": "spy",
                    "event_type": "rsi_oversold",
                    "windows": [5, 21],
                    "event_params": {"period": 10, "threshold": 35},
                    "signal_dates": ["2024-01-13", "2024-02-10"],
                },
            ),
        )

    def test_create_signal_validates_scope_arguments_and_persists(self) -> None:
        class FakeStore:
            def signal_create(self, user_id, name, signal_type, params, ticker_overrides, ticker_scope, exchange):  # noqa: ANN001
                self.args = (user_id, name, signal_type, params, ticker_overrides, ticker_scope, exchange)
                return 42

        fake_store = FakeStore()
        ok = json.loads(signals_service.create_signal_payload(fake_store, "user-a", "Name", "rsi_oversold", {"threshold": 35}, "tickers", ["spy"], None))
        bad = json.loads(signals_service.create_signal_payload(fake_store, "user-a", "Name", "rsi_oversold", None, "watchlist", ["spy"], None))
        ex_bad = json.loads(signals_service.create_signal_payload(fake_store, "user-a", "Name", "rsi_oversold", None, "exchange", None, "BAD"))

        self.assertEqual(ok["id"], 42)
        self.assertEqual(fake_store.args, ("user-a", "Name", "rsi_oversold", {"threshold": 35}, ["SPY"], "tickers", None))
        self.assertIn("omit ticker_overrides", bad["error"])
        self.assertEqual(ex_bad["error"], "invalid exchange: BAD; use NYSE, NASDAQ, AMEX, or CRYPTO")

    def test_create_signal_rejects_unsupported_intraday_history_combos(self) -> None:
        class FakeStore:
            def signal_create(self, *args, **kwargs):  # noqa: ANN002, ANN003
                raise AssertionError("signal_create should not be called for invalid input")

        payload = json.loads(
            signals_service.create_signal_payload(
                FakeStore(),
                "user-a",
                "Intraday RSI",
                "rsi_oversold",
                None,
                "tickers",
                ["spy"],
                None,
                history_period="3mo",
                interval="1h",
            )
        )

        self.assertEqual(
            payload["error"],
            "unsupported history_period/interval combination: 3mo with 1h; intraday intervals require history_period of 1d, 5d, or 1mo",
        )

    def test_get_ath_distance_handles_none_history(self) -> None:
        provider = FakeProvider(history=None)

        with patch("scanner_mcp.server.get_provider", return_value=provider):
            result = json.loads(server.get_ath_distance("spy"))

        self.assertEqual(result, {"error": "no data"})

    def test_run_scan_payload_records_persisted_history_fetch_errors(self) -> None:
        class FakeStore:
            def watchlist_get(self, _user_id):  # noqa: ANN001
                return []

            def signal_list(self, _user_id):  # noqa: ANN001
                return [
                    type(
                        "SignalRow",
                        (),
                        {
                            "id": 7,
                            "name": "RSI Oversold",
                            "signal_type": "rsi_oversold",
                            "params": {},
                            "ticker_overrides": ["SPY"],
                            "ticker_scope": "tickers",
                            "exchange": None,
                            "enabled": True,
                        },
                    )()
                ]

        class ExplodingProvider:
            def get_history(self, _ticker: str, *, period: str, interval: str):  # noqa: ANN001
                raise RuntimeError("fetch failed")

        payload = json.loads(signals_service.run_scan_payload(FakeStore(), ExplodingProvider(), "user-a"))

        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["errors"], [{"symbol": "SPY", "signal_id": 7, "error": "fetch failed"}])

    def test_run_scan_payload_records_persisted_empty_history_errors(self) -> None:
        class FakeStore:
            def watchlist_get(self, _user_id):  # noqa: ANN001
                return []

            def signal_list(self, _user_id):  # noqa: ANN001
                return [
                    type(
                        "SignalRow",
                        (),
                        {
                            "id": 7,
                            "name": "RSI Oversold",
                            "signal_type": "rsi_oversold",
                            "params": {},
                            "ticker_overrides": ["SPY"],
                            "ticker_scope": "tickers",
                            "exchange": None,
                            "enabled": True,
                        },
                    )()
                ]

        class EmptyProvider:
            def get_history(self, _ticker: str, *, period: str, interval: str):  # noqa: ANN001
                return pd.DataFrame()

        payload = json.loads(signals_service.run_scan_payload(FakeStore(), EmptyProvider(), "user-a"))

        self.assertEqual(payload["results"], [])
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["errors"], [{"symbol": "SPY", "signal_id": 7, "error": "no_history"}])


if __name__ == "__main__":
    unittest.main()

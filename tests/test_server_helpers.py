from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import pandas as pd

from scanner_mcp import server


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
        self.assertEqual(server._parse_ind("rsi:10"), ("rsi", {"period": 10}))
        self.assertEqual(server._parse_ind("ema:21"), ("ema", {"period": 21}))
        self.assertEqual(server._parse_ind("macd"), ("macd", {}))
        self.assertEqual(server._as_float("12.5"), 12.5)
        self.assertIsNone(server._as_float(float("nan")))
        self.assertIsNone(server._as_float("bad"))

    def test_quote_snapshot_uses_fast_info_then_history_fallback(self) -> None:
        provider = FakeProvider({"last_price": 110.0, "previous_close": 100.0})
        self.assertEqual(server._quote_snapshot(provider, "SPY")["day_change_pct"], 10.0)  # type: ignore[arg-type]

        hist = pd.DataFrame({"Close": [100.0, 105.0]})
        provider = FakeProvider({}, hist)
        out = server._quote_snapshot(provider, "SPY")  # type: ignore[arg-type]

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

        with patch("scanner_mcp.server._get_provider", return_value=provider):
            out = server._compute_indicators("spy", ["rsi:3", "sma:3", "unknown"], "6mo")

        self.assertEqual(out["symbol"], "SPY")
        self.assertIn("rsi_3", out)
        self.assertIn("sma_3", out)
        self.assertEqual(out["unknown"]["error"], "unknown indicator: unknown")
        self.assertIn(out["consensus"], {"buy", "hold", "sell"})

    def test_chart_tool_result_returns_json_error_for_bad_chart_response(self) -> None:
        with (
            patch("scanner_mcp.server._get_provider", return_value=object()),
            patch("scanner_mcp.server.chartgen.generate_chart", return_value={"mime": "image/png"}),
        ):
            self.assertEqual(json.loads(server._chart_tool_result("x", {}))["error"], "chart response missing image data")

        with (
            patch("scanner_mcp.server._get_provider", return_value=object()),
            patch("scanner_mcp.server.chartgen.generate_chart", side_effect=RuntimeError("boom")),
        ):
            self.assertEqual(json.loads(server._chart_tool_result("x", {}))["error"], "boom")

    def test_chart_fundamental_overlay_passes_params(self) -> None:
        with patch("scanner_mcp.server._chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(server.chart_fundamental_overlay("msft", "earnings", "annual", "10y", "1d", "line"), "ok")

        chart_tool.assert_called_once_with(
            "fundamental_overlay",
            {
                "symbol": "msft",
                "metric": "earnings",
                "frequency": "annual",
                "period": "10y",
                "interval": "1d",
                "price_style": "line",
            },
        )

    def test_new_chart_tools_pass_params(self) -> None:
        with patch("scanner_mcp.server._chart_tool_result", return_value="ok") as chart_tool:
            self.assertEqual(server.chart_ratio("spy", "xlp", "6mo"), "ok")
            self.assertEqual(server.chart_relative_strength("aapl", "spy", "3y", 30), "ok")
            self.assertEqual(server.chart_sector_rotation(["XLK", "XLF"], "5y", 42), "ok")

        self.assertEqual(chart_tool.call_args_list[0].args, ("ratio_chart", {"symbol": "spy", "benchmark": "xlp", "period": "6mo"}))
        self.assertEqual(
            chart_tool.call_args_list[1].args,
            ("relative_strength", {"symbol": "aapl", "benchmark": "spy", "period": "3y", "ma_period": 30}),
        )
        self.assertEqual(
            chart_tool.call_args_list[2].args,
            ("sector_rotation", {"symbols": ["XLK", "XLF"], "period": "5y", "return_window": 42}),
        )

    def test_create_signal_validates_scope_arguments_and_persists(self) -> None:
        class FakeStore:
            def signal_create(self, name, signal_type, params, ticker_overrides, ticker_scope, exchange):  # noqa: ANN001
                self.args = (name, signal_type, params, ticker_overrides, ticker_scope, exchange)
                return 42

        fake_store = FakeStore()
        with patch("scanner_mcp.server._get_store", return_value=fake_store):
            ok = json.loads(server.create_signal("Name", "rsi_oversold", {"threshold": 35}, "tickers", ["spy"], None))
            bad = json.loads(server.create_signal("Name", "rsi_oversold", None, "watchlist", ["spy"], None))
            ex_bad = json.loads(server.create_signal("Name", "rsi_oversold", None, "exchange", None, "BAD"))

        self.assertEqual(ok["id"], 42)
        self.assertEqual(fake_store.args, ("Name", "rsi_oversold", {"threshold": 35}, ["SPY"], "tickers", None))
        self.assertIn("omit ticker_overrides", bad["error"])
        self.assertIn("invalid exchange", ex_bad["error"])


if __name__ == "__main__":
    unittest.main()

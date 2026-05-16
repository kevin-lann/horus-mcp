from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from scanner_mcp.signals.catalog import list_catalog_entries, merge_params
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal


def _signal(signal_type: str, params: dict | None = None) -> ActiveSignal:
    return ActiveSignal(1, signal_type, signal_type, params or {}, None)


class SignalCatalogTest(unittest.TestCase):
    def test_catalog_lists_public_entries_and_merges_params(self) -> None:
        entries = list_catalog_entries()

        self.assertTrue(any(x["signal_type"] == "golden_cross" for x in entries))
        self.assertEqual(merge_params("rsi_oversold", {"threshold": 35})["period"], 14)
        self.assertEqual(merge_params("rsi_oversold", {"threshold": 35})["threshold"], 35)
        with self.assertRaises(ValueError):
            merge_params("missing", {})


class SignalEvaluatorTest(unittest.TestCase):
    def test_rejects_missing_data_and_unknown_signal(self) -> None:
        self.assertEqual(evaluate(_signal("rsi_oversold"), pd.DataFrame()), (False, {"error": "no_data"}))

        with self.assertRaises(ValueError):
            evaluate(_signal("does_not_exist"), pd.DataFrame({"Close": [1.0, 2.0]}))

    def test_latest_golden_and_death_crosses(self) -> None:
        df = pd.DataFrame({"Close": [10.0, 9.0, 8.0, 9.0, 12.0]})

        ok, details = evaluate(_signal("golden_cross", {"fast": 2, "slow": 3}), df)

        self.assertTrue(ok)
        self.assertIn("sma_2", details)
        self.assertIn("sma_3", details)

        df2 = pd.DataFrame({"Close": [8.0, 9.0, 10.0, 9.0, 6.0]})
        ok, _ = evaluate(_signal("death_cross", {"fast": 2, "slow": 3}), df2)
        self.assertTrue(ok)

    def test_rsi_thresholds_use_latest_value(self) -> None:
        df = pd.DataFrame({"Close": [float(x) for x in range(20)]})

        with patch("scanner_mcp.signals.evaluator.Indicators.rsi", return_value=25.0):
            ok, details = evaluate(_signal("rsi_oversold"), df)
        self.assertTrue(ok)
        self.assertEqual(details["threshold"], 30)

        with patch("scanner_mcp.signals.evaluator.Indicators.rsi", return_value=75.0):
            ok, _ = evaluate(_signal("rsi_overbought"), df)
        self.assertTrue(ok)

    def test_macd_crossovers_handle_both_directions(self) -> None:
        df = pd.DataFrame({"Close": [100.0, 99.0, 101.0]})
        macd = pd.DataFrame(
            {
                "MACD_12_26_9": [-1.0, -0.5, 1.0],
                "MACDs_12_26_9": [0.0, 0.0, 0.0],
                "MACDh_12_26_9": [-1.0, -0.5, 1.0],
            }
        )

        with patch("scanner_mcp.signals.calculations.ta.macd", return_value=macd):
            ok, details = evaluate(_signal("macd_bullish_crossover"), df)

        self.assertTrue(ok)
        self.assertEqual(details, {"macd": 1.0, "signal": 0.0})

        bearish = pd.DataFrame(
            {
                "MACD_12_26_9": [1.0, 0.5, -1.0],
                "MACDs_12_26_9": [0.0, 0.0, 0.0],
            }
        )
        with patch("scanner_mcp.signals.calculations.ta.macd", return_value=bearish):
            ok, _ = evaluate(_signal("macd_bearish_crossover"), df)
        self.assertTrue(ok)

    def test_pct_from_ma_pct_from_ath_bbands_and_bull_flag(self) -> None:
        df = pd.DataFrame(
            {
                "High": [100.0, 110.0, 120.0, 100.0],
                "Close": [100.0, 105.0, 120.0, 90.0],
            }
        )

        ok, details = evaluate(_signal("pct_from_ath", {"min_pct_below_ath": 20.0}), df)
        self.assertTrue(ok)
        self.assertAlmostEqual(details["pct_from_ath"], -25.0)

        close = pd.Series([10.0, 10.0, 10.0, 11.0])
        ok, details = evaluate(_signal("pct_from_ma", {"ma_period": 3, "pct": 10.0}), pd.DataFrame({"Close": close}))
        self.assertTrue(ok)
        self.assertLessEqual(details["diff_pct"], 10.0)

        bands = pd.DataFrame({"BBL_3_2.0": [9.0], "BBU_3_2.0": [11.0]})
        with patch("scanner_mcp.signals.evaluator.ta.bbands", return_value=bands):
            ok, details = evaluate(
                _signal("bbands_breakout", {"length": 3, "std": 2.0, "side": "upper"}),
                pd.DataFrame({"Close": [12.0]}),
            )
        self.assertTrue(ok)
        self.assertEqual(details["broke"], "upper")

        flag_df = pd.DataFrame({"Close": [100.0, 110.0, 120.0, 121.0, 122.0, 123.0]})
        ok, details = evaluate(
            _signal("bull_flag", {"prior_lookback": 2, "prior_move_pct": 9.0, "consol_days": 3, "max_range_pct": 2.0}),
            flag_df,
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(details["prior_move_pct"], 9.0)

    def test_evaluator_returns_exception_details(self) -> None:
        with patch("scanner_mcp.signals.evaluator.calc.moving_average", side_effect=RuntimeError("boom")):
            ok, details = evaluate(_signal("golden_cross", {"fast": 2, "slow": 3}), pd.DataFrame({"Close": [1, 2, 3, 4, 5]}))

        self.assertFalse(ok)
        self.assertEqual(details, {"error": "boom"})


if __name__ == "__main__":
    unittest.main()

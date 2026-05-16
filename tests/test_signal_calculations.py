from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scanner_mcp.signals import calculations as calc  # noqa: E402


class SignalCalculationsTest(unittest.TestCase):
    def test_crossed_detects_bullish_and_bearish_crosses(self) -> None:
        self.assertTrue(calc.crossed(1.0, 3.0, 2.0, 2.0, "bullish"))
        self.assertFalse(calc.crossed(3.0, 2.0, 2.0, 2.0, "bullish"))
        self.assertTrue(calc.crossed(3.0, 1.0, 2.0, 2.0, "bearish"))
        self.assertFalse(calc.crossed(1.0, 2.0, 2.0, 2.0, "bearish"))

    def test_latest_cross_values_returns_last_two_valid_points(self) -> None:
        lhs = pd.Series([10.0, 11.0, 12.0])
        rhs = pd.Series([9.0, 10.0, 13.0])

        vals = calc.latest_cross_values(lhs, rhs)

        self.assertIsNotNone(vals)
        assert vals is not None
        self.assertEqual(vals.lhs_prev, 11.0)
        self.assertEqual(vals.lhs_cur, 12.0)
        self.assertEqual(vals.rhs_prev, 10.0)
        self.assertEqual(vals.rhs_cur, 13.0)

    def test_latest_cross_values_rejects_short_or_nan_series(self) -> None:
        self.assertIsNone(calc.latest_cross_values(pd.Series([1.0]), pd.Series([1.0, 2.0])))
        self.assertIsNone(calc.latest_cross_values(pd.Series([1.0, float("nan")]), pd.Series([1.0, 2.0])))

    def test_cross_indexes_skips_nan_and_returns_matching_direction(self) -> None:
        lhs = pd.Series([1.0, 3.0, float("nan"), 4.0, 1.0])
        rhs = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])

        self.assertEqual(calc.cross_indexes(lhs, rhs, "bullish"), [1])
        self.assertEqual(calc.cross_indexes(lhs, rhs, "bearish"), [4])

    def test_moving_average_returns_normalized_type_and_series(self) -> None:
        close = pd.Series([10.0, 11.0, 12.0, 13.0])

        sma_type, sma = calc.moving_average(close, 2, "unknown")
        ema_type, ema = calc.moving_average(close, 2, "EMA")

        self.assertEqual(sma_type, "sma")
        self.assertAlmostEqual(float(sma.iloc[-1]), 12.5)
        self.assertEqual(ema_type, "ema")
        self.assertFalse(pd.isna(ema.iloc[-1]))

    def test_pct_distance_from_ma_returns_absolute_percent_distance(self) -> None:
        close = pd.Series([100.0, 110.0, 90.0])
        ma = pd.Series([100.0, 100.0, 100.0])

        out = calc.pct_distance_from_ma(close, ma)

        self.assertEqual(out.tolist(), [0.0, 10.0, 10.0])
    
    def test_pct_distance_from_ma_handles_zero_ma(self) -> None:
        close = pd.Series([100.0, 110.0])
        ma = pd.Series([100.0, 0.0])
        
        out = calc.pct_distance_from_ma(close, ma)
        
        self.assertEqual(out.iloc[0], 0.0)
        self.assertTrue(pd.isna(out.iloc[1]))  # or expect inf, depending on fix

    def test_rsi_threshold_cross_indexes_detects_entry_only(self) -> None:
        close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0])
        rsi = pd.Series([50.0, 31.0, 29.0, 28.0, 40.0])

        with patch("scanner_mcp.signals.calculations.ta.rsi", return_value=rsi):
            self.assertEqual(
                calc.rsi_threshold_cross_indexes(close, period=14, threshold=30.0, below=True),
                [2],
            )

    def test_rsi_threshold_cross_indexes_detects_overbought_entry(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0, 103.0])
        rsi = pd.Series([50.0, 69.0, 71.0, 72.0])

        with patch("scanner_mcp.signals.calculations.ta.rsi", return_value=rsi):
            self.assertEqual(
                calc.rsi_threshold_cross_indexes(close, period=14, threshold=70.0, below=False),
                [2],
            )

    def test_macd_columns_resolves_line_signal_and_histogram(self) -> None:
        out = pd.DataFrame(
            {
                "MACD_8_21_5": [1.0],
                "MACDs_8_21_5": [0.5],
                "MACDh_8_21_5": [0.5],
            }
        )

        cols = calc.macd_columns(out)

        self.assertIsNotNone(cols)
        assert cols is not None
        self.assertEqual(cols.macd, "MACD_8_21_5")
        self.assertEqual(cols.signal, "MACDs_8_21_5")
        self.assertEqual(cols.hist, "MACDh_8_21_5")

    def test_macd_columns_returns_none_without_required_columns(self) -> None:
        self.assertIsNone(calc.macd_columns(None))
        self.assertIsNone(calc.macd_columns(pd.DataFrame({"MACD_1_2_3": [1.0]})))

    def test_macd_wrapper_returns_output_and_columns(self) -> None:
        out = pd.DataFrame(
            {
                "MACD_12_26_9": [1.0],
                "MACDs_12_26_9": [0.5],
            }
        )

        with patch("scanner_mcp.signals.calculations.ta.macd", return_value=out):
            result = calc.macd(pd.Series([1.0, 2.0]), fast=12, slow=26, signal=9)

        self.assertIsNotNone(result)
        assert result is not None
        returned, cols = result
        self.assertTrue(returned.equals(out))
        self.assertIsNotNone(cols)
        assert cols is not None
        self.assertEqual(cols.macd, "MACD_12_26_9")
        self.assertEqual(cols.signal, "MACDs_12_26_9")


if __name__ == "__main__":
    unittest.main()

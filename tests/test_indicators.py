from __future__ import annotations

import unittest

import pandas as pd

from scanner_mcp.indicators import ratings, ta
from scanner_mcp.indicators.core import Indicators, beta_from_returns


class TechnicalAnalysisHelpersTest(unittest.TestCase):
    def test_sma_ema_macd_and_bbands_shapes(self) -> None:
        close = pd.Series([1, 2, 3, 4, 5], dtype=float)

        sma = ta.sma(close, 3)
        ema = ta.ema(close, 3)
        macd = ta.macd(close, fast=2, slow=3, signal=2)
        bands = ta.bbands(close, length=3, std=2.0)

        self.assertTrue(pd.isna(sma.iloc[1]))
        self.assertEqual(float(sma.iloc[-1]), 4.0)
        self.assertFalse(pd.isna(ema.iloc[-1]))
        self.assertEqual(list(macd.columns), ["MACD_2_3_2", "MACDs_2_3_2", "MACDh_2_3_2"])
        self.assertEqual(list(bands.columns), ["BBL_3_2.0", "BBM_3_2.0", "BBU_3_2.0"])
        self.assertAlmostEqual(float(bands["BBM_3_2.0"].iloc[-1]), 4.0)

    def test_rsi_handles_one_way_series(self) -> None:
        rising = pd.Series(range(1, 20), dtype=float)
        falling = pd.Series(range(20, 1, -1), dtype=float)

        self.assertEqual(float(ta.rsi(rising, length=3).iloc[-1]), 100.0)
        self.assertEqual(float(ta.rsi(falling, length=3).iloc[-1]), 0.0)


class IndicatorsFacadeTest(unittest.TestCase):
    def test_latest_indicator_values_and_ath_distance(self) -> None:
        df = pd.DataFrame(
            {
                "Open": range(1, 31),
                "High": [float(x) for x in range(2, 32)],
                "Low": range(0, 30),
                "Close": [float(x) for x in range(1, 31)],
                "Volume": [100] * 30,
            }
        )
        ind = Indicators(df)

        self.assertFalse(ind.empty)
        self.assertIsNotNone(ind.rsi(14))
        self.assertIsNotNone(ind.macd(fast=3, slow=6, signal=3))
        self.assertIsNotNone(ind.bbands(period=5))
        self.assertEqual(ind.sma(3), 29.0)
        self.assertIsNotNone(ind.ema(3))
        self.assertAlmostEqual(ind.ath_distance() or 0.0, (30.0 - 31.0) / 31.0 * 100.0)
        self.assertEqual(ind.last_close(), 30.0)

    def test_indicators_return_none_without_enough_data(self) -> None:
        ind = Indicators(pd.DataFrame({"Close": [1.0]}))

        self.assertTrue(ind.empty)
        self.assertIsNone(ind.rsi())
        self.assertIsNone(ind.macd())
        self.assertIsNone(ind.bbands())
        self.assertIsNone(ind.sma(2))
        self.assertIsNone(ind.ema(2))
        self.assertIsNone(ind.ath_distance())
        self.assertIsNone(ind.last_close())

    def test_bbands_pct_b_uses_midpoint_when_band_width_is_zero(self) -> None:
        ind = Indicators(pd.DataFrame({"Close": [10.0, 10.0, 10.0]}))

        out = ind.bbands(period=3)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["pct_b"], 0.5)

    def test_beta_requires_enough_aligned_non_constant_benchmark_returns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=25)
        bench = pd.Series([x / 100.0 for x in range(25)], index=idx)
        sym = bench * 2.0

        self.assertAlmostEqual(beta_from_returns(sym, bench) or 0.0, 2.0)
        self.assertIsNone(beta_from_returns(sym.iloc[:10], bench.iloc[:10]))
        self.assertIsNone(beta_from_returns(sym, pd.Series([0.01] * 25, index=idx)))


class RatingsTest(unittest.TestCase):
    def test_rating_thresholds_and_dispatch(self) -> None:
        self.assertEqual(ratings.consensus(["buy", "hold", "sell"]), "hold")
        self.assertEqual(ratings.consensus(["buy", "buy", "sell"]), "buy")
        self.assertEqual(ratings.rate_rsi(20.0), "buy")
        self.assertEqual(ratings.rate_rsi(80.0), "sell")
        self.assertEqual(ratings.rate_macd({"hist": 2.0, "hist_prev": 1.0}), "buy")
        self.assertEqual(ratings.rate_macd({"hist": -2.0, "hist_prev": -1.0}), "sell")
        self.assertEqual(ratings.rate_bbands({"pct_b": 0.1}), "buy")
        self.assertEqual(ratings.rate_bbands({"pct_b": 0.9}), "sell")
        self.assertEqual(ratings.rate_price_vs_ma(102.0, 100.0), "buy")
        self.assertEqual(ratings.rate_price_vs_ma(98.0, 100.0), "sell")
        self.assertEqual(ratings.rate_ath_distance(-40.0), "buy")
        self.assertEqual(ratings.rate_ath_distance(-1.0), "sell")
        self.assertEqual(ratings.rate_single("sma:50", {"price": 102.0, "value": 100.0}), "buy")
        self.assertEqual(ratings.rate_single("unknown", {}), "hold")


if __name__ == "__main__":
    unittest.main()

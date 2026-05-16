from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scanner_mcp.charts import generator as chartgen  # noqa: E402


class PriceHistoryChartTest(unittest.TestCase):
    def test_optional_overlays_add_expected_traces(self) -> None:
        dates = pd.date_range("2025-01-01", periods=240, freq="D")
        close = pd.Series(range(100, 340), index=dates, dtype=float)
        df = pd.DataFrame(
            {
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Volume": 1000,
            },
            index=dates,
        )
        fig = go.Figure()

        chartgen._add_price_history_main_traces(
            fig,
            df,
            "XYZ",
            {
                "show_ma": True,
                "ma_period": 20,
                "show_bollinger_bands": True,
                "show_ema": True,
                "show_ma_cloud": True,
                "show_fib_retracement": True,
                "show_avwap": True,
            },
        )

        names = [trace.name for trace in fig.data]
        self.assertIn("XYZ", names)
        self.assertIn("SMA 20", names)
        self.assertIn("EMA 21", names)
        self.assertIn("SMA 50 cloud", names)
        self.assertIn("aVWAP", names)
        self.assertTrue(any(str(name).startswith("BB upper") for name in names))
        fib_trace = next(trace for trace in fig.data if trace.name == "Fib 0.618")
        self.assertFalse(fib_trace.showlegend)
        self.assertGreater(fib_trace.x[1], dates[-1])
        fib_range, fib_label_x = chartgen._fib_x_padding(dates)
        self.assertEqual(fib_trace.x[1], fib_label_x)
        self.assertLess(fib_trace.x[1], fib_range[1])
        fib_label = next(annotation for annotation in fig.layout.annotations if annotation.text == "0.618 190.83")
        self.assertEqual(fib_label.xref, "paper")
        self.assertEqual(fib_label.xanchor, "right")
        self.assertEqual(fib_label.font.size, 8)

    def test_anchored_vwap_starts_at_anchor(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "High": [12.0, 24.0, 36.0],
                "Low": [9.0, 18.0, 27.0],
                "Close": [9.0, 18.0, 27.0],
                "Volume": [100.0, 100.0, 300.0],
            },
            index=dates,
        )

        out = chartgen._anchored_vwap(df, "2025-01-02")

        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertAlmostEqual(float(out.iloc[1]), 20.0)
        self.assertAlmostEqual(float(out.iloc[2]), 27.5)


if __name__ == "__main__":
    unittest.main()

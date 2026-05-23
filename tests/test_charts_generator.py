from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scanner_mcp.charts import generator
from scanner_mcp.charts.comparison import (
    price_overlay,
    drawdown_comparison,
    log_cycle,
    ratio_chart,
    relative_strength_chart,
    sector_rotation_chart,
)
from scanner_mcp.charts.data import contiguous_true_spans, fib_x_padding, price_history_frames
from scanner_mcp.charts.forward_returns import (
    format_number,
    forward_return_cell_color,
    mix_rgb,
    window_label,
)
from scanner_mcp.charts.fundamentals import fundamental_overlay
from scanner_mcp.charts.params import as_bool, positive_int
from scanner_mcp.charts.price_history import (
    add_price_history_main_traces,
    anchored_vwap,
    price_history,
)
from scanner_mcp.data.fundamentals_utils import merge_asof_price_over_eps, statement_metric_series


class FakeProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[dict[str, object]] = []
        self.fundamentals = pd.Series(dtype=float)
        self.pe = pd.Series(dtype=float)

    def get_history(
        self,
        symbol: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        start: object | None = None,
        end: object | None = None,
    ) -> pd.DataFrame:
        self.calls.append({"symbol": symbol, "period": period, "interval": interval, "start": start, "end": end})
        return self.frames.get(symbol, pd.DataFrame())

    def get_fundamental_series(self, _symbol: str, _metric: str, _frequency: str) -> pd.Series:
        return self.fundamentals

    def get_historical_pe_series(self, _symbol: str, _close: pd.Series) -> pd.Series:
        return self.pe


class ChartGeneratorTest(unittest.TestCase):
    def test_dispatch_and_small_parsers(self) -> None:
        self.assertTrue(as_bool("yes"))
        self.assertFalse(as_bool("no"))
        self.assertEqual(positive_int("-5", 10), 10)
        self.assertEqual(positive_int("7", 10), 7)

        with self.assertRaises(ValueError):
            generator.generate_chart(FakeProvider({}), "missing", {})

    def test_asof_pe_merge_and_trailing_pe_fallbacks(self) -> None:
        close = pd.Series(
            [10.0, 20.0, 30.0],
            index=pd.to_datetime(["2024-01-01", "2024-03-01", "2024-05-01"]),
        )
        pe = merge_asof_price_over_eps(
            close,
            pd.to_datetime(["2024-02-01", "2024-04-01"]),
            np.array([2.0, -1.0]),
        )

        self.assertTrue(pd.isna(pe.iloc[0]))
        self.assertEqual(pe.iloc[1], 10.0)
        self.assertTrue(pd.isna(pe.iloc[2]))

    def test_statement_metric_series_extracts_revenue_and_earnings(self) -> None:
        stmt = pd.DataFrame(
            {
                pd.Timestamp("2024-12-31"): [100.0, 30.0],
                pd.Timestamp("2023-12-31"): [90.0, 25.0],
            },
            index=["Total Revenue", "Net Income"],
        )

        revenue = statement_metric_series(stmt, "revenue")
        earnings = statement_metric_series(stmt, "earnings")

        self.assertEqual(list(revenue), [90.0, 100.0])
        self.assertEqual(list(earnings), [25.0, 30.0])
        with self.assertRaises(ValueError):
            statement_metric_series(stmt, "cashflow")

    def test_anchored_vwap_respects_anchor_and_zero_volume(self) -> None:
        df = pd.DataFrame(
            {
                "High": [12.0, 14.0, 16.0],
                "Low": [8.0, 10.0, 12.0],
                "Close": [10.0, 13.0, 15.0],
                "Volume": [100.0, 0.0, 100.0],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        out = anchored_vwap(df, "2024-01-02")

        self.assertTrue(pd.isna(out.iloc[0]))
        self.assertTrue(pd.isna(out.iloc[1]))
        self.assertAlmostEqual(float(out.iloc[2]), (16.0 + 12.0 + 15.0) / 3.0)

    def test_price_overlay_drawdown_and_log_cycle_render_without_real_kaleido(self) -> None:
        df = pd.DataFrame({"Close": [100.0, 110.0, 105.0]}, index=pd.date_range("2024-01-01", periods=3))
        provider = FakeProvider({"SPY": df, "QQQ": df, "BTC-USD": df})

        with (
            patch("scanner_mcp.charts.comparison.fig_to_b64", return_value="pngdata") as render,
        ):
            self.assertEqual(price_overlay(provider, {"symbols": ["SPY", "QQQ"], "period": "1mo"})["data"], "pngdata")
            self.assertEqual(drawdown_comparison(provider, {"symbols": ["SPY"], "period": "1mo"})["data"], "pngdata")
            self.assertEqual(log_cycle(provider, {"symbol": "BTC-USD", "period": "max"})["data"], "pngdata")

        self.assertEqual(render.call_count, 3)

    def test_price_overlay_and_drawdown_reject_empty_symbol_data(self) -> None:
        provider = FakeProvider({})

        with self.assertRaisesRegex(ValueError, "no data available for requested symbols"):
            price_overlay(provider, {"symbols": "SPY,QQQ", "period": "1mo"})

        with self.assertRaisesRegex(ValueError, "no data available for requested symbols"):
            drawdown_comparison(provider, {"symbols": "SPY", "period": "1mo"})

    def test_ratio_and_relative_strength_and_sector_rotation_render(self) -> None:
        index = pd.date_range("2024-01-01", periods=90, freq="D")
        spy = pd.DataFrame({"Close": np.linspace(100.0, 130.0, len(index))}, index=index)
        xlp = pd.DataFrame({"Close": np.linspace(50.0, 55.0, len(index))}, index=index)
        aapl = pd.DataFrame({"Close": np.linspace(150.0, 210.0, len(index))}, index=index)
        xlk = pd.DataFrame({"Close": np.linspace(100.0, 135.0, len(index))}, index=index)
        xlf = pd.DataFrame({"Close": np.linspace(100.0, 108.0, len(index))}, index=index)
        xle = pd.DataFrame({"Close": np.linspace(100.0, 94.0, len(index))}, index=index)
        provider = FakeProvider({"SPY": spy, "XLP": xlp, "AAPL": aapl, "XLK": xlk, "XLF": xlf, "XLE": xle})

        def fake_render(fig: go.Figure, chart_type: str) -> str:
            if chart_type == "ratio_chart":
                ratio_trace = fig.data[0]
                self.assertEqual(ratio_trace.name, "SPY/XLP")
                self.assertAlmostEqual(float(ratio_trace.y[0]), 2.0)
            elif chart_type == "relative_strength":
                self.assertEqual(fig.data[0].name, "AAPL/SPY")
                self.assertEqual(fig.data[1].name, "10D SMA")
                self.assertGreaterEqual(len(fig.layout.shapes), 1)
            elif chart_type == "sector_rotation":
                names = [trace.name for trace in fig.data]
                self.assertIn("XLK", names)
                self.assertIn("XLF 21D", names)
                self.assertEqual(fig.layout.yaxis.title.text, "Normalized (base=100)")
                self.assertEqual(fig.layout.yaxis2.title.text, "21D return %")
                self.assertGreaterEqual(pd.Timestamp(fig.layout.xaxis.range[1]) - index[-1], pd.Timedelta(days=15))
            else:
                self.fail(f"unexpected chart type {chart_type}")
            return "pngdata"

        with patch("scanner_mcp.charts.comparison.fig_to_b64", side_effect=fake_render):
            self.assertEqual(
                ratio_chart(provider, {"symbol": "SPY", "benchmark": "XLP", "period": "6mo"})["data"],
                "pngdata",
            )
            self.assertEqual(
                relative_strength_chart(
                    provider,
                    {"symbol": "AAPL", "benchmark": "SPY", "period": "6mo", "ma_period": 10},
                )["data"],
                "pngdata",
            )
            self.assertEqual(
                sector_rotation_chart(
                    provider,
                    {"symbols": ["XLK", "XLF", "XLE"], "period": "6mo", "return_window": 21},
                )["data"],
                "pngdata",
            )

    def test_fundamental_overlay_fetches_statement_bars(self) -> None:
        price = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0, 13.0],
                "High": [11.0, 12.0, 13.0, 14.0],
                "Low": [9.0, 10.0, 11.0, 12.0],
                "Close": [10.5, 11.5, 12.5, 13.5],
            },
            index=pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]),
        )
        fundamentals = pd.Series(
            [250_000_000_000.0, 300_000_000_000.0],
            index=pd.to_datetime(["2024-06-30", "2024-09-30"]),
            dtype=float,
        )
        fundamentals.attrs["source"] = "Alpha Vantage"

        def fake_render(fig, chart_type: str) -> str:
            self.assertEqual(chart_type, "fundamental_overlay")
            self.assertEqual(fig.data[0].name, "Quarterly Revenue (Alpha Vantage)")
            self.assertEqual(fig.data[1].name, "XYZ")
            self.assertIn("300B", fig.layout.yaxis2.ticktext)
            return "pngdata"

        provider = FakeProvider({"XYZ": price})
        provider.fundamentals = fundamentals
        with patch("scanner_mcp.charts.fundamentals.fig_to_b64", side_effect=fake_render):
            result = fundamental_overlay(provider, {"symbol": "XYZ", "metric": "revenue", "period": "1y"})

        self.assertEqual(result["data"], "pngdata")

    def test_price_history_adds_requested_overlays(self) -> None:
        df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0, 13.0],
                "High": [11.0, 12.0, 15.0, 14.0],
                "Low": [9.0, 10.0, 11.0, 12.0],
                "Close": [10.5, 11.5, 13.0, 13.5],
                "Volume": [100.0, 100.0, 100.0, 100.0],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        fig = go.Figure()

        add_price_history_main_traces(
            fig,
            df,
            "XYZ",
            {
                "show_ma": True,
                "ma_period": 2,
                "show_ema": True,
                "ema_period": 2,
                "show_bollinger_bands": True,
                "bb_period": 2,
                "show_ma_cloud": True,
                "ma_cloud_fast": 2,
                "ma_cloud_slow": 3,
                "show_avwap": True,
                "show_fib_retracement": True,
            },
        )

        names = {trace.name for trace in fig.data}
        self.assertIn("XYZ", names)
        self.assertIn("SMA 2", names)
        self.assertIn("EMA 2", names)
        self.assertIn("aVWAP", names)
        fib_trace = next(trace for trace in fig.data if trace.name == "Fib 0.618")
        self.assertFalse(fib_trace.showlegend)
        self.assertGreater(fib_trace.x[1], df.index[-1])
        fib_range, fib_label_x = fib_x_padding(df.index)
        self.assertEqual(fib_trace.x[1], fib_label_x)
        self.assertLess(fib_trace.x[1], fib_range[1])
        fib_label = next(annotation for annotation in fig.layout.annotations if annotation.text == "0.618 11.29")
        self.assertEqual(fib_label.xref, "paper")
        self.assertEqual(fib_label.xanchor, "right")
        self.assertEqual(fib_label.font.size, 8)

    def test_price_history_fetches_preroll_for_indicator_warmup(self) -> None:
        visible_index = pd.date_range("2024-01-10", periods=3, freq="D")
        full_index = pd.date_range("2024-01-01", periods=12, freq="D")
        visible_df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.0, 11.0, 12.0],
                "Volume": [100.0, 100.0, 100.0],
            },
            index=visible_index,
        )
        full_df = pd.DataFrame(
            {
                "Open": np.arange(1.0, 13.0),
                "High": np.arange(2.0, 14.0),
                "Low": np.arange(0.0, 12.0),
                "Close": np.arange(1.0, 13.0),
                "Volume": 100.0,
            },
            index=full_index,
        )

        class WarmupProvider(FakeProvider):
            def get_history(
                self,
                symbol: str,
                *,
                period: str = "6mo",
                interval: str = "1d",
                start: object | None = None,
                end: object | None = None,
            ) -> pd.DataFrame:
                self.calls.append({"symbol": symbol, "period": period, "interval": interval, "start": start, "end": end})
                return full_df if start is not None or end is not None else visible_df

        provider = WarmupProvider({"XYZ": visible_df})

        def fake_render(fig: go.Figure, _chart_type: str) -> str:
            candle = fig.data[0]
            sma = next(trace for trace in fig.data if trace.name == "SMA 5")
            self.assertEqual(list(candle.x), list(visible_index))
            self.assertEqual(list(sma.x), list(visible_index))
            self.assertFalse(pd.isna(pd.Series(sma.y)).any())
            return "pngdata"

        with patch("scanner_mcp.charts.price_history.fig_to_b64", side_effect=fake_render):
            result = price_history(provider, {"symbol": "XYZ", "period": "1mo", "show_ma": True, "ma_period": 5})

        self.assertEqual(result["data"], "pngdata")
        self.assertEqual(len(provider.calls), 2)
        self.assertIsNone(provider.calls[0]["start"])
        self.assertIsNotNone(provider.calls[1]["start"])
        self.assertIsNotNone(provider.calls[1]["end"])

    def test_avwap_defaults_to_first_visible_bar_when_preroll_exists(self) -> None:
        visible_index = pd.date_range("2024-01-10", periods=3, freq="D")
        full_index = pd.date_range("2024-01-01", periods=12, freq="D")
        visible_df = pd.DataFrame(
            {
                "Open": [10.0, 11.0, 12.0],
                "High": [11.0, 12.0, 13.0],
                "Low": [9.0, 10.0, 11.0],
                "Close": [10.0, 11.0, 12.0],
                "Volume": [100.0, 100.0, 100.0],
            },
            index=visible_index,
        )
        full_df = pd.DataFrame(
            {
                "Open": np.arange(1.0, 13.0),
                "High": np.arange(2.0, 14.0),
                "Low": np.arange(0.0, 12.0),
                "Close": np.arange(1.0, 13.0),
                "Volume": 100.0,
            },
            index=full_index,
        )
        fig = go.Figure()

        add_price_history_main_traces(
            fig,
            visible_df,
            "XYZ",
            {"show_avwap": True},
            indicator_df=full_df,
        )

        avwap = next(trace for trace in fig.data if trace.name == "aVWAP")
        expected = anchored_vwap(full_df, visible_index[0]).reindex(visible_index)
        np.testing.assert_allclose(np.asarray(avwap.y, dtype=float), expected.to_numpy(dtype=float))

    def test_forward_return_color_helpers(self) -> None:
        self.assertEqual(window_label(21), "1 Month")
        self.assertEqual(window_label(7), "7d")
        self.assertEqual(format_number(3.0), "3")
        self.assertEqual(mix_rgb((255, 255, 255), (0, 0, 0), 0.5), "rgb(128,128,128)")
        self.assertEqual(forward_return_cell_color("Mean", None, []), "#ffffff")
        self.assertNotEqual(forward_return_cell_color("Mean", 5.0, [5.0]), "#ffffff")
        self.assertNotEqual(forward_return_cell_color("% Positive", 25.0, [25.0]), "#ffffff")

    def test_contiguous_true_spans_groups_runs(self) -> None:
        mask = pd.Series(
            [False, True, True, False, True],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )

        spans = contiguous_true_spans(mask)

        self.assertEqual(spans[0], (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")))
        self.assertEqual(spans[1], (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05")))


if __name__ == "__main__":
    unittest.main()

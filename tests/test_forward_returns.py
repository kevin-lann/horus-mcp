from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scanner_mcp.research.forward_returns import (  # noqa: E402
    ForwardEvent,
    ForwardStudy,
    ForwardWindowResult,
    SignalEvent,
    compute_custom_date_forward_study_from_history,
    compute_event_forward_study_from_history,
    summarize_forward_study,
)
from scanner_mcp.charts.forward_returns import forward_event_title, forward_returns_chart  # noqa: E402


class ForwardReturnsTest(unittest.TestCase):
    def test_custom_detector_computes_forward_path_stats(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 110.0, 104.5, 121.0, 115.5]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "custom_signal",
            [2, 4],
            detectors={"custom_signal": lambda _df, _params: [SignalEvent(0, "Custom")]},
        )

        self.assertEqual(len(study.events), 1)
        event = study.events[0]
        self.assertAlmostEqual(event.windows[2].final_return, 4.5)
        self.assertAlmostEqual(event.windows[2].max_loss, 0.0)
        self.assertAlmostEqual(event.windows[2].max_gain, 10.0)
        self.assertAlmostEqual(event.windows[4].final_return, 15.5)
        self.assertAlmostEqual(event.windows[4].max_gain, 21.0)

    def test_empty_detector_registry_is_respected(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 110.0, 120.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "rsi_oversold",
            [1],
            detectors={},
        )

        self.assertTrue(study.price.empty)
        self.assertEqual(study.events, [])

    def test_incomplete_horizons_are_excluded(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 95.0, 98.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "custom_signal",
            [1, 3],
            detectors={"custom_signal": lambda _df, _params: [SignalEvent(1, "Custom")]},
        )

        self.assertEqual(list(study.events[0].windows), [1])
        self.assertEqual(summarize_forward_study(study)[3]["n"], 0)

    def test_custom_signal_dates_map_to_next_trading_session_and_dedupe(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 103.0, 107.0, 111.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"]),
        )

        study = compute_custom_date_forward_study_from_history(
            df,
            "XYZ",
            ["2024-01-01", "2024-01-06", "2024-01-08"],
            [1],
        )

        self.assertEqual([ev.index for ev in study.events], [0, 2])
        self.assertEqual([str(ev.date.date()) for ev in study.events], ["2024-01-02", "2024-01-08"])
        self.assertEqual([ev.label for ev in study.events], ["Custom Signal", "Custom Signal"])
        self.assertAlmostEqual(study.events[0].windows[1].final_return, 3.0)
        self.assertAlmostEqual(study.events[1].windows[1].final_return, (111.0 - 107.0) / 107.0 * 100.0)

    def test_rsi_crossing_detector_finds_threshold_events(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 99.0, 98.0, 97.0, 96.0]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        rsi = pd.Series([50.0, 35.0, 29.0, 28.0, 40.0])

        with patch("scanner_mcp.signals.calculations.ta.rsi", return_value=rsi):
            study = compute_event_forward_study_from_history(df, "XYZ", "rsi_oversold", [1])

        self.assertEqual([ev.index for ev in study.events], [2])
        self.assertEqual(study.events[0].label, "RSI Oversold")
        self.assertAlmostEqual(study.events[0].windows[1].final_return, -1.0204081632653061)

    def test_golden_cross_detector_finds_sma_cross(self) -> None:
        df = pd.DataFrame(
            {"Close": [10.0, 9.0, 8.0, 9.0, 12.0, 15.0]},
            index=pd.date_range("2024-01-01", periods=6),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "golden_cross",
            [1],
            params={"fast": 2, "slow": 3},
        )

        self.assertEqual([ev.index for ev in study.events], [4])
        self.assertEqual(study.events[0].label, "Golden Cross (2/3)")
        self.assertAlmostEqual(study.events[0].windows[1].final_return, 25.0)

    def test_macd_bullish_detector_finds_signal_cross(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 99.0, 98.0, 101.0, 103.0]},
            index=pd.date_range("2024-01-01", periods=5),
        )
        macd = pd.DataFrame(
            {
                "MACD_12_26_9": [float("nan"), 0.0, -1.0, 1.0, 2.0],
                "MACDs_12_26_9": [float("nan"), 0.0, 0.0, 0.0, 1.0],
                "MACDh_12_26_9": [float("nan"), 0.0, -1.0, 1.0, 1.0],
            }
        )

        with patch("scanner_mcp.signals.calculations.ta.macd", return_value=macd):
            study = compute_event_forward_study_from_history(df, "XYZ", "macd_bullish_crossover", [1])

        self.assertEqual([ev.index for ev in study.events], [3])
        self.assertEqual(study.events[0].label, "MACD Bullish Cross (12/26/9)")
        self.assertAlmostEqual(study.events[0].windows[1].final_return, 1.9801980198019802)

    def test_pct_from_ma_detector_finds_entry_into_band(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 100.0, 100.0, 110.0, 104.0, 108.0]},
            index=pd.date_range("2024-01-01", periods=6),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "pct_from_ma",
            [1],
            params={"ma_period": 3, "ma_type": "sma", "pct": 2.0},
        )

        self.assertEqual([ev.index for ev in study.events], [4])
        self.assertEqual(study.events[0].label, "Within 2% of SMA 3")
        self.assertAlmostEqual(study.events[0].windows[1].final_return, 3.8461538461538463)

    def test_forward_returns_chart_passes_event_params(self) -> None:
        price = pd.Series(
            [100.0, 103.0],
            index=pd.date_range("2024-01-01", periods=2),
        )
        study = ForwardStudy(
            symbol="XYZ",
            event_type="pct_from_ma",
            windows=[1],
            price=price,
            events=[
                ForwardEvent(
                    index=0,
                    date=price.index[0],
                    price=100.0,
                    label="Within 3% of EMA 200",
                    event_type="pct_from_ma",
                    windows={1: ForwardWindowResult(final_return=3.0, max_loss=0.0, max_gain=3.0)},
                )
            ],
        )
        event_params = {"ma_type": "ema", "ma_period": 200, "pct": 3}
        provider = object()
        captured_title = None

        def fake_fig_to_b64(fig, _chart_type: str) -> str:
            nonlocal captured_title
            captured_title = fig.layout.title.text
            return "pngdata"

        with (
            patch("scanner_mcp.charts.forward_returns.compute_event_forward_study", return_value=study) as compute,
            patch("scanner_mcp.charts.forward_returns.fig_to_b64", side_effect=fake_fig_to_b64),
        ):
            result = forward_returns_chart(
                provider=provider,
                params={
                    "symbol": "XYZ",
                    "event_type": "pct_from_ma",
                    "windows": [1],
                    "event_params": event_params,
                },
            )

        self.assertEqual(result, {"mime": "image/png", "data": "pngdata"})
        self.assertEqual(captured_title, "XYZ forward returns after price moves within 3% of 200-day EMA")
        compute.assert_called_once_with(
            provider,
            "XYZ",
            "pct_from_ma",
            [1],
            period="10y",
            params=event_params,
        )

    def test_forward_event_titles_include_resolved_params(self) -> None:
        self.assertEqual(
            forward_event_title("pct_from_ma", {"ma_type": "ema", "ma_period": 200, "pct": 3}),
            "price moves within 3% of 200-day EMA",
        )
        self.assertEqual(
            forward_event_title("rsi_oversold", {"period": 10, "threshold": 35}),
            "RSI Oversold (10-day RSI crosses below 35)",
        )
        self.assertEqual(
            forward_event_title("rsi_overbought", None),
            "RSI Overbought (14-day RSI crosses above 70)",
        )
        self.assertEqual(
            forward_event_title("golden_cross", {"fast": 20, "slow": 100}),
            "Golden Cross (20-day SMA crosses above 100-day SMA)",
        )
        self.assertEqual(
            forward_event_title("macd_bullish_crossover", {"fast": 8, "slow": 21, "signal": 5}),
            "MACD Bullish Crossover (8/21 MACD crosses above 5-day signal)",
        )

    def test_forward_returns_chart_uses_largest_populated_marker_window(self) -> None:
        price = pd.Series(
            [100.0, 104.0],
            index=pd.date_range("2024-01-01", periods=2),
        )
        study = ForwardStudy(
            symbol="XYZ",
            event_type="rsi_oversold",
            windows=[1, 3],
            price=price,
            events=[
                ForwardEvent(
                    index=0,
                    date=price.index[0],
                    price=100.0,
                    label="RSI Oversold",
                    event_type="rsi_oversold",
                    windows={1: ForwardWindowResult(final_return=4.0, max_loss=0.0, max_gain=4.0)},
                )
            ],
        )
        provider = object()
        marker_names: list[str | None] = []
        marker_text: list[str] = []

        def fake_fig_to_b64(fig, _chart_type: str) -> str:
            marker_names.extend(trace.name for trace in fig.data if getattr(trace, "mode", None) == "markers")
            for trace in fig.data:
                if getattr(trace, "mode", None) == "markers":
                    marker_text.extend(trace.text)
            return "pngdata"

        with (
            patch("scanner_mcp.charts.forward_returns.compute_event_forward_study", return_value=study),
            patch("scanner_mcp.charts.forward_returns.fig_to_b64", side_effect=fake_fig_to_b64),
        ):
            forward_returns_chart(
                provider=provider,
                params={"symbol": "XYZ", "event_type": "rsi_oversold", "windows": [1, 3]},
            )

        self.assertIn("Signal Positive After 1d", marker_names)
        self.assertEqual(marker_text, ["RSI Oversold<br>1d return: 4.0%"])

    def test_forward_returns_chart_treats_zero_and_nonfinite_returns_as_neutral(self) -> None:
        price = pd.Series([100.0, 101.0, 102.0], index=pd.date_range("2024-01-01", periods=3))
        study = ForwardStudy(
            symbol="XYZ",
            event_type="rsi_oversold",
            windows=[1],
            price=price,
            events=[
                ForwardEvent(
                    index=0,
                    date=price.index[0],
                    price=100.0,
                    label="Zero",
                    event_type="rsi_oversold",
                    windows={1: ForwardWindowResult(final_return=0.0, max_loss=0.0, max_gain=0.0)},
                ),
                ForwardEvent(
                    index=1,
                    date=price.index[1],
                    price=101.0,
                    label="NaN",
                    event_type="rsi_oversold",
                    windows={1: ForwardWindowResult(final_return=float("nan"), max_loss=0.0, max_gain=0.0)},
                ),
            ],
        )
        marker_names: list[str | None] = []
        marker_text: list[str] = []

        def fake_fig_to_b64(fig, _chart_type: str) -> str:
            marker_names.extend(trace.name for trace in fig.data if getattr(trace, "mode", None) == "markers")
            for trace in fig.data:
                if getattr(trace, "mode", None) == "markers":
                    marker_text.extend(trace.text)
            return "pngdata"

        with (
            patch("scanner_mcp.charts.forward_returns.compute_event_forward_study", return_value=study),
            patch("scanner_mcp.charts.forward_returns.fig_to_b64", side_effect=fake_fig_to_b64),
        ):
            forward_returns_chart(
                provider=object(),
                params={"symbol": "XYZ", "event_type": "rsi_oversold", "windows": [1]},
            )

        self.assertEqual(marker_names, ["Signal After 1d"])
        self.assertEqual(marker_text, ["Zero<br>1d return: 0.0%", "NaN<br>1d return: n/a"])

    def test_summary_aggregates_by_window(self) -> None:
        df = pd.DataFrame(
            {"Close": [100.0, 110.0, 90.0, 99.0]},
            index=pd.date_range("2024-01-01", periods=4),
        )

        study = compute_event_forward_study_from_history(
            df,
            "XYZ",
            "custom_signal",
            [1],
            detectors={
                "custom_signal": lambda _df, _params: [
                    SignalEvent(0, "Custom"),
                    SignalEvent(2, "Custom"),
                ]
            },
        )
        summary = summarize_forward_study(study)[1]

        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(float(summary["mean"]), 10.0)
        self.assertAlmostEqual(float(summary["median"]), 10.0)
        self.assertAlmostEqual(float(summary["positive_pct"]), 100.0)


if __name__ == "__main__":
    unittest.main()

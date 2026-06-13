from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.scanner import scheduler


class StoreIntegrationTest(unittest.TestCase):
    def test_watchlist_signal_alert_lifecycle_is_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            self.assertEqual(store.watchlist_add("user-a", [" aapl ", "AAPL", "", "msft"]), ["AAPL", "MSFT"])
            self.assertEqual(store.watchlist_add("user-b", ["msft", "tsla"]), ["MSFT", "TSLA"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["AAPL", "MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-b")], ["MSFT", "TSLA"])
            self.assertEqual(store.watchlist_remove("user-a", ["aapl", "missing"]), 1)
            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-b")], ["MSFT", "TSLA"])

            sid = store.signal_create(
                "user-a",
                "Dip",
                "pct_from_ath",
                {"min_pct_below_ath": 10},
                ["spy"],
                ticker_scope="tickers",
                exchange=None,
            )
            other_sid = store.signal_create("user-b", "Other", "rsi_oversold", {}, ["qqq"], ticker_scope="tickers", exchange=None)
            sig = store.signal_get("user-a", sid)
            self.assertIsNotNone(sig)
            assert sig is not None
            self.assertEqual(sig.ticker_overrides, ["spy"])
            self.assertEqual(sig.ticker_scope, "tickers")
            self.assertTrue(sig.enabled)
            self.assertIsNone(store.signal_get("user-a", other_sid))

            self.assertTrue(store.signal_set_enabled("user-a", sid, False))
            self.assertFalse(store.signal_get("user-a", sid).enabled)  # type: ignore[union-attr]
            aid = store.alert_insert("user-a", sid, "spy", {"x": 1})
            alerts = store.alerts_recent("user-a")
            self.assertEqual(alerts[0].id, aid)
            self.assertEqual(alerts[0].symbol, "SPY")
            self.assertEqual(alerts[0].details, {"x": 1})
            self.assertEqual(store.alerts_recent("user-b"), [])
            self.assertFalse(store.signal_delete("user-b", sid))
            self.assertTrue(store.signal_delete("user-a", sid))
            self.assertIsNone(store.signal_get("user-a", sid))
            self.assertEqual(store.alerts_recent("user-a"), [])

    def test_scan_jobs_are_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            job_id = store.scan_job_create("user-a", "run_scan", {"symbol": "SPY"})

            self.assertIsNotNone(store.scan_job_get("user-a", job_id))
            self.assertIsNone(store.scan_job_get("user-b", job_id))
            self.assertFalse(store.scan_job_request_cancel("user-b", job_id))
            self.assertTrue(store.scan_job_request_cancel("user-a", job_id))


class SchedulerTest(unittest.TestCase):
    def test_tickers_for_signal_resolves_scope(self) -> None:
        tickers = SignalRow(1, "T", "rsi_oversold", {}, ["AAPL"], "tickers", None, True, "now")
        watch = SignalRow(2, "W", "rsi_oversold", {}, None, "watchlist", None, True, "now")
        exchange = SignalRow(3, "E", "rsi_oversold", {}, None, "exchange", "NASDAQ", True, "now")

        self.assertEqual(scheduler._tickers_for_signal(tickers, ["MSFT"]), ["AAPL"])
        self.assertEqual(scheduler._tickers_for_signal(watch, ["MSFT"]), ["MSFT"])
        with patch("scanner_mcp.scanner.scheduler.fetch_exchange_tickers", return_value=["QQQ"]):
            self.assertEqual(scheduler._tickers_for_signal(exchange, ["MSFT"]), ["QQQ"])

    def test_run_full_scan_persists_and_notifies_triggered_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            store.watchlist_add("user-a", ["SPY"])
            sid_a = store.signal_create("user-a", "RSI", "rsi_oversold", {}, None)
            store.watchlist_add("user-b", ["QQQ"])
            sid_b = store.signal_create("user-b", "RSI-B", "rsi_oversold", {}, None)

            class Provider:
                def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
                    self.symbol = symbol
                    return pd.DataFrame({"Close": [1.0, 2.0]})

            with (
                patch("scanner_mcp.scanner.scheduler.evaluate", return_value=(True, {"rsi": 20.0})) as eval_mock,
                patch("scanner_mcp.scanner.scheduler.notify_desktop") as notify_mock,
            ):
                result = scheduler.run_full_scan(store, Provider(), notify=True)  # type: ignore[arg-type]

            self.assertEqual(result["checked"], 2)
            self.assertEqual(result["fired"], 2)
            user_ids_in_alerts = {a["user_id"] for a in result["alerts"]}
            self.assertIn("user-a", user_ids_in_alerts)
            self.assertIn("user-b", user_ids_in_alerts)
            alert_a = next(a for a in result["alerts"] if a["user_id"] == "user-a")
            alert_b = next(a for a in result["alerts"] if a["user_id"] == "user-b")
            self.assertEqual(alert_a["signal_id"], sid_a)
            self.assertEqual(alert_b["signal_id"], sid_b)
            self.assertEqual(store.alerts_recent("user-a")[0].details, {"rsi": 20.0})
            self.assertEqual(store.alerts_recent("user-b")[0].details, {"rsi": 20.0})
            self.assertEqual(eval_mock.call_count, 2)
            self.assertEqual(notify_mock.call_count, 2)

    def test_scan_job_swallows_exceptions(self) -> None:
        with patch("scanner_mcp.scanner.scheduler.run_full_scan", side_effect=RuntimeError("boom")):
            scheduler._scan_job(object(), object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

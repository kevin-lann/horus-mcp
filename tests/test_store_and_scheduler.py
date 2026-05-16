from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.scanner import scheduler


class StoreIntegrationTest(unittest.TestCase):
    def test_watchlist_signal_alert_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            self.assertEqual(store.watchlist_add([" aapl ", "AAPL", "", "msft"]), ["AAPL", "MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get()], ["AAPL", "MSFT"])
            self.assertEqual(store.watchlist_remove(["aapl", "missing"]), 1)
            self.assertEqual([w.symbol for w in store.watchlist_get()], ["MSFT"])

            sid = store.signal_create(
                "Dip",
                "pct_from_ath",
                {"min_pct_below_ath": 10},
                ["spy"],
                ticker_scope="tickers",
                exchange=None,
            )
            sig = store.signal_get(sid)
            self.assertIsNotNone(sig)
            assert sig is not None
            self.assertEqual(sig.ticker_overrides, ["spy"])
            self.assertEqual(sig.ticker_scope, "tickers")
            self.assertTrue(sig.enabled)

            self.assertTrue(store.signal_set_enabled(sid, False))
            self.assertFalse(store.signal_get(sid).enabled)  # type: ignore[union-attr]
            aid = store.alert_insert(sid, "spy", {"x": 1})
            alerts = store.alerts_recent()
            self.assertEqual(alerts[0].id, aid)
            self.assertEqual(alerts[0].symbol, "SPY")
            self.assertEqual(alerts[0].details, {"x": 1})
            self.assertTrue(store.signal_delete(sid))
            self.assertIsNone(store.signal_get(sid))
            self.assertEqual(store.alerts_recent(), [])


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
            store.watchlist_add(["SPY"])
            sid = store.signal_create("RSI", "rsi_oversold", {}, None)

            class Provider:
                def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
                    self.symbol = symbol
                    return pd.DataFrame({"Close": [1.0, 2.0]})

            with (
                patch("scanner_mcp.scanner.scheduler.evaluate", return_value=(True, {"rsi": 20.0})) as eval_mock,
                patch("scanner_mcp.scanner.scheduler.notify_desktop") as notify_mock,
            ):
                result = scheduler.run_full_scan(store, Provider(), notify=True)  # type: ignore[arg-type]

            self.assertEqual(result["checked"], 1)
            self.assertEqual(result["fired"], 1)
            self.assertEqual(result["alerts"][0]["signal_id"], sid)
            self.assertEqual(store.alerts_recent()[0].details, {"rsi": 20.0})
            eval_mock.assert_called_once()
            notify_mock.assert_called_once()

    def test_scan_job_swallows_exceptions(self) -> None:
        with patch("scanner_mcp.scanner.scheduler.run_full_scan", side_effect=RuntimeError("boom")):
            scheduler._scan_job(object(), object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

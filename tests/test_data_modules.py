from __future__ import annotations

import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from scanner_mcp.data.cache import TTLCache
from scanner_mcp.data import exchange_universe, movers
from scanner_mcp.data.provider import YFinanceProvider, _merge_asof_price_over_eps


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


class TTLCacheTest(unittest.TestCase):
    def test_get_set_expire_and_clear(self) -> None:
        cache: TTLCache[str] = TTLCache(default_ttl=10.0)
        with patch("scanner_mcp.data.cache.time.monotonic", side_effect=[100.0, 105.0, 111.0]):
            cache.set("a", "value")
            self.assertEqual(cache.get("a"), "value")
            self.assertIsNone(cache.get("a"))
        cache.set("b", "value")
        cache.clear()
        self.assertIsNone(cache.get("b"))


class YFinanceProviderTest(unittest.TestCase):
    def test_history_is_cached_and_returned_as_copy(self) -> None:
        calls: list[tuple[str, str]] = []
        df = pd.DataFrame({"Close": [1.0, 2.0]})

        class FakeTicker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, *, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
                calls.append((period, interval))
                return df

        with patch("scanner_mcp.data.provider.yf.Ticker", FakeTicker):
            provider = YFinanceProvider()
            first = provider.get_history(" spy ", period="1mo", interval="1d")
            first.iloc[0, 0] = 999.0
            second = provider.get_history("SPY", period="1mo", interval="1d")

        self.assertEqual(calls, [("1mo", "1d")])
        self.assertEqual(float(second.iloc[0]["Close"]), 1.0)

    def test_history_failure_returns_empty_dataframe(self) -> None:
        class FakeTicker:
            def __init__(self, _symbol: str) -> None:
                pass

            def history(self, **_kwargs: object) -> pd.DataFrame:
                raise RuntimeError("network")

        with patch("scanner_mcp.data.provider.yf.Ticker", FakeTicker):
            self.assertTrue(YFinanceProvider().get_history("SPY").empty)

    def test_fast_info_cache_and_option_chain_paths(self) -> None:
        class FakeTicker:
            options = ["2024-01-19", "2024-02-16"]

            def __init__(self, _symbol: str) -> None:
                self.fast_info = {"last_price": 123.0}

            def option_chain(self, expiry: str) -> types.SimpleNamespace:
                return types.SimpleNamespace(
                    calls=pd.DataFrame({"strike": [100]}),
                    puts=pd.DataFrame({"strike": [90]}),
                    expiry=expiry,
                )

        with patch("scanner_mcp.data.provider.yf.Ticker", FakeTicker):
            provider = YFinanceProvider()
            self.assertEqual(provider.get_fast_info("spy"), {"last_price": 123.0})
            self.assertEqual(provider.get_fast_info("SPY"), {"last_price": 123.0})
            chain = provider.get_option_chain("SPY", None)

        self.assertEqual(chain["expiries"], ["2024-01-19", "2024-02-16"])
        self.assertFalse(chain["calls"].empty)

    def test_alpha_vantage_income_statement_series_parses_quarterly_and_annual_rows(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_get(_url: str, *, params: dict[str, object], timeout: float) -> FakeHTTPResponse:
            calls.append(params)
            self.assertEqual(params["function"], "INCOME_STATEMENT")
            return FakeHTTPResponse(
                {
                    "quarterlyReports": [
                        {"fiscalDateEnding": "2024-03-31", "totalRevenue": "100", "netIncome": "10"},
                        {"fiscalDateEnding": "2023-12-31", "totalRevenue": "90", "netIncome": "9"},
                    ],
                    "annualReports": [
                        {"fiscalDateEnding": "2024-12-31", "totalRevenue": "500", "netIncome": "50"},
                        {"fiscalDateEnding": "2023-12-31", "totalRevenue": "450", "netIncome": "45"},
                    ]
                }
            )

        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}),
            patch("scanner_mcp.data.provider.httpx.get", side_effect=fake_get),
        ):
            provider = YFinanceProvider()
            quarterly = provider.get_fundamental_series("aapl", "revenue", "quarterly")
            annual = provider.get_fundamental_series("aapl", "earnings", "annual")

        self.assertEqual(list(quarterly), [90.0, 100.0])
        self.assertEqual(quarterly.attrs["source"], "Alpha Vantage")
        self.assertEqual(list(annual), [45.0, 50.0])
        self.assertEqual([call["symbol"] for call in calls], ["AAPL"])

    def test_alpha_vantage_missing_api_key_returns_empty_without_http_request(self) -> None:
        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}, clear=True),
            patch("scanner_mcp.data.provider._ENV_FILE_PATH", Path("/tmp/missing-scanner-mcp-env")),
            patch("scanner_mcp.data.provider.httpx.get") as http_get,
        ):
            provider = YFinanceProvider()
            series = provider._alpha_vantage_income_statement_series("AAPL", "revenue", "quarterly")

        self.assertTrue(series.empty)
        http_get.assert_not_called()

    def test_alpha_vantage_api_key_can_be_read_from_root_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("ALPHA_VANTAGE_API_KEY=file-key\n", encoding="utf-8")
            with (
                patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}, clear=True),
                patch("scanner_mcp.data.provider._ENV_FILE_PATH", env_path),
            ):
                self.assertEqual(YFinanceProvider()._alpha_vantage_api_key(), "file-key")

    def test_historical_pe_prefers_alpha_vantage_earnings_before_yahoo_fallback(self) -> None:
        close = pd.Series(
            [100.0, 120.0, 150.0],
            index=pd.to_datetime(["2024-01-02", "2024-04-02", "2024-07-02"]),
            dtype=float,
        )

        def fake_get(_url: str, *, params: dict[str, object], timeout: float) -> FakeHTTPResponse:
            self.assertEqual(timeout, 15.0)
            self.assertEqual(params["function"], "EARNINGS")
            return FakeHTTPResponse(
                {
                    "quarterlyEarnings": [
                        {"fiscalDateEnding": "2024-06-30", "reportedEPS": "1.50"},
                        {"fiscalDateEnding": "2024-03-31", "reportedEPS": "1.00"},
                        {"fiscalDateEnding": "2023-12-31", "reportedEPS": "1.00"},
                        {"fiscalDateEnding": "2023-09-30", "reportedEPS": "1.00"},
                        {"fiscalDateEnding": "2023-06-30", "reportedEPS": "1.00"},
                    ]
                }
            )

        with (
            patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}),
            patch("scanner_mcp.data.provider.httpx.get", side_effect=fake_get),
            patch("scanner_mcp.data.provider.yf.Ticker", side_effect=AssertionError("Yahoo fallback should not be used")),
        ):
            pe = YFinanceProvider().get_historical_pe_series("AAPL", close)

        self.assertTrue(pd.isna(pe.iloc[0]))
        self.assertEqual(float(pe.iloc[1]), 30.0)
        self.assertEqual(float(pe.iloc[2]), 150.0 / 4.5)
        self.assertEqual(pe.attrs["source"], "Alpha Vantage EPS")

    def test_pe_merge_normalizes_mixed_datetime_resolutions(self) -> None:
        close = pd.Series(
            [100.0, 120.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-04-02"]).astype("datetime64[s]"),
            dtype=float,
        )
        pe = _merge_asof_price_over_eps(
            close,
            pd.DatetimeIndex(["2023-12-31", "2024-03-31"]).astype("datetime64[us]"),
            pd.Series([4.0, 5.0]).to_numpy(dtype=float),
        )

        self.assertEqual(list(pe), [25.0, 24.0])


class MoversTest(unittest.TestCase):
    def test_quote_normalization_and_exchange_filtering(self) -> None:
        quote = {
            "symbol": "ABC",
            "shortName": "ABC Inc",
            "regularMarketPrice": 10,
            "regularMarketChangePercent": 5,
            "regularMarketVolume": 1000,
            "exchange": "NMS",
        }

        self.assertEqual(movers._quote_to_row(quote)["symbol"], "ABC")  # type: ignore[index]
        self.assertEqual([r["symbol"] for r in movers._filter_quotes([quote], "NASDAQ")], ["ABC"])
        self.assertEqual(movers._filter_quotes([quote], "NYSE"), [])
        self.assertIsNone(movers._quote_to_row({"symbol": "MISS"}))

    def test_screen_movers_filters_sorts_and_limits(self) -> None:
        quotes = [
            {"symbol": "A", "regularMarketPrice": 1, "regularMarketChangePercent": 1, "exchange": "NMS"},
            {"symbol": "B", "regularMarketPrice": 1, "regularMarketChangePercent": 5, "exchange": "NMS"},
            {"symbol": "C", "regularMarketPrice": 1, "regularMarketChangePercent": -7, "exchange": "NMS"},
        ]

        with patch("scanner_mcp.data.movers.yf.screen", return_value={"quotes": quotes}):
            gainers = movers.screen_movers("gainers", "nasdaq", limit=2)
            losers = movers.screen_movers("losers", "nasdaq", limit=1)

        self.assertEqual([x["symbol"] for x in gainers], ["B", "A"])
        self.assertEqual([x["symbol"] for x in losers], ["C"])
        with self.assertRaises(ValueError):
            movers.screen_movers("gainers", "bad")

    def test_crypto_movers_uses_fast_info(self) -> None:
        class FakeTickers:
            def __init__(self, _symbols: str) -> None:
                self.tickers = {
                    "BTC-USD": types.SimpleNamespace(fast_info={"last_price": 110.0, "previous_close": 100.0, "last_volume": 1}),
                    "ETH-USD": types.SimpleNamespace(fast_info={"last_price": 90.0, "previous_close": 100.0, "last_volume": 2}),
                }

        with (
            patch.object(movers, "_CRYPTO_TICKERS", ["BTC-USD", "ETH-USD"]),
            patch("scanner_mcp.data.movers.yf.Tickers", FakeTickers),
        ):
            self.assertEqual([x["symbol"] for x in movers._crypto_movers("gainers", 2)], ["BTC-USD", "ETH-USD"])
            self.assertEqual([x["symbol"] for x in movers._crypto_movers("losers", 1)], ["ETH-USD"])


class ExchangeUniverseTest(unittest.TestCase):
    def setUp(self) -> None:
        exchange_universe._CACHE.clear()

    def test_crypto_universe_caps_and_caches(self) -> None:
        out = exchange_universe.fetch_exchange_tickers("crypto", max_symbols=2)

        self.assertEqual(out, ["BTC-USD", "ETH-USD"])
        self.assertEqual(exchange_universe.fetch_exchange_tickers("CRYPTO", max_symbols=1), ["BTC-USD"])

    def test_equity_universe_paginates_dedupes_and_honors_cap(self) -> None:
        first_page = [{"symbol": f"SYM{x:03d}"} for x in range(249)]
        first_page.append({"symbol": "BBB"})
        responses = [
            {"quotes": first_page},
            {"quotes": [{"symbol": "BBB"}, {"symbol": "ccc"}]},
        ]

        def fake_screen(_query: object, *, offset: int, size: int, sortField: str, sortAsc: bool) -> dict:
            self.assertEqual(size, 250)
            return responses[0 if offset == 0 else 1]

        with patch("scanner_mcp.data.exchange_universe.yf.screen", side_effect=fake_screen):
            out = exchange_universe.fetch_exchange_tickers("NYSE", max_symbols=251, use_cache=False)

        self.assertEqual(out[-2:], ["BBB", "CCC"])
        self.assertEqual(len(out), 251)
        with self.assertRaises(ValueError):
            exchange_universe.fetch_exchange_tickers("BAD")


if __name__ == "__main__":
    unittest.main()

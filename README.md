# scanner-mcp

MCP server for market research, technical indicators (with buy/hold/sell ratings), signal scanning, charts (base64 PNG), and watchlists. Data via **yfinance**; persistence in **SQLite** at `~/.scanner_mcp/data.db` (override with `SCANNER_MCP_DB`).

## Setup

Use a virtual environment so dependencies and the `scanner-mcp` entry point stay isolated from the system Python.

```bash
cd /path/to/scanner-mcp
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Run

With the venv activated:

```bash
scanner-mcp
```

The default transport is **stdio**. Other MCP clients should run the same `scanner-mcp` command (full path to the venv’s binary if the client does not load your shell `PATH`).

**Env**

- `SCANNER_MCP_DB` — SQLite file path (overrides default)
- `SCAN_TIME` — daily scan time `HH:MM` Eastern (default `16:30`)
- `LOG_LEVEL` — default `INFO`

## Claude Desktop

1. Complete [Setup](#setup) so `.venv` exists and `pip install -e .` has been run.
2. Open the Desktop config file:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
3. Add a `scanner-mcp` entry under `mcpServers` (use the real absolute path to your venv; merge with any servers you already have):

```json
{
  "mcpServers": {
    "scanner-mcp": {
      "command": "~/Projects/scanner-mcp/.venv/bin/scanner-mcp",
      "args": []
    }
  }
}
```

4. Optional `env` block for the same variables as in **Env** above, e.g. `SCANNER_MCP_DB`, `SCAN_TIME`, `LOG_LEVEL`.
5. Restart Claude Desktop.

**Windows:** set `"command"` to the `.exe`, e.g. `C:\\path\\to\\scanner-mcp\\.venv\\Scripts\\scanner-mcp.exe`.

If the shim is not on `PATH` for your client, you can use the venv’s Python: `"command": "/path/to/.venv/bin/python"`, `"args": ["-m", "scanner_mcp.server"]`.

## Tools

- Market: `get_price`, `get_indicators`, `get_ath_distance`, `get_option_chain`, `market_snapshot`, `top_gainers`, `top_losers`
- Signals: `list_signal_catalog`, `create_signal` (`params`: JSON object; `ticker_overrides`: string array when scope=tickers), `list_signals`, `delete_signal`, `run_scan` (`tickers`: string array)
- Watchlist: `add_to_watchlist`, `remove_from_watchlist`, `get_watchlist` (`symbols`: array of tickers)
- Charts (typed args): `chart_price_history`, `chart_price_overlay`, `chart_forward_returns`, `chart_drawdown_comparison`, `chart_log_cycle` — each uses `_chart_tool_result`, which on success returns a FastMCP `Image` (`Image(data=base64.b64decode(...))`: raw PNG bytes wrapped for an MCP image block). On failure it returns JSON text with an `error` field.

## Unit tests
Run unit tests for the entire repo using:
```bash
python3 -m pytest
```

## Local tool testing

Use `test.py` to smoke-test tools directly from the repo without starting an MCP client. Run it from the project root after setup:

```bash
source .venv/bin/activate
python3 test.py --help
```

Examples:

```bash
python3 test.py --tool catalog
python3 test.py --tool price --symbol SPY
python3 test.py --tool indicators --symbol AAPL --period 6mo
python3 test.py --tool options --symbol AAPL
python3 test.py --tool snapshot
python3 test.py --tool movers --exchange NASDAQ --limit 5
python3 test.py --tool signals
python3 test.py --tool chart --chart-type price_history --symbol SPY
python3 test.py --tool chart --chart-type price_overlay --symbols SPY QQQ --period 6mo
python3 test.py --tool all --symbol SPY
```

Chart tests print a compact base64 summary and save debug PNGs to `output/`.

Tests that modify the local SQLite database are opt-in:

```bash
python3 test.py --tool watchlist --mutate --symbols AAPL MSFT
python3 test.py --tool create_signal --mutate --signal-name "debug rsi" --signal-type rsi_oversold --signal-params '{"threshold":100}' --signal-tickers --symbols AAPL MSFT AMZN META VST BTC HOOD BE LITE SOFI COIN MSTR PLTR TSM MU NVDA AMD PLTR NFLX PATH ZETA NOW CEG UUUU MA RDDT SNDK CLS IREN APLD NBIS CRWV ASTS RKLB ORCL
python3 test.py --tool scan
python3 test.py --tool scan --symbols AAPL MSFT
python3 test.py --tool delete_signal --mutate --signal-id 1
```

By default, `scan` does not pass a ticker override. It lets each persisted signal use its own `ticker_overrides`, or the global watchlist when a signal has no overrides. Add `--symbols` only when you want to override the scan tickers for the test run.

## Testing MCP server locally

Use the official MCP inspector tool for visual debugging:

```bash
npx @modelcontextprotocol/inspector scanner-mcp
```

## Resources

- `signals://triggered` — recent alerts (JSON)
- `signals://watchlist` — tickers (JSON)
- `research://forward-returns/{symbol}/{event_type}` — markdown table (`event_type`: `rsi_oversold` | `rsi_overbought` | `golden_cross` | `macd_bullish_crossover` | `pct_from_ma`)

Forward-return charts show a price panel with historical signal markers plus a summary table. Default horizons are 21, 42, 63, 84, 105, 126, and 252 daily trading bars. `chart_forward_returns` accepts optional `event_params`; for `pct_from_ma`, pass values such as `{"ma_type":"ema","ma_period":200,"pct":3}`.

## Connecting to the SQLite DB

Using CLI:
```bash
sqlite3 ~/.scanner_mcp/data.db
```
Useful once inside:

- `.tables` — list tables (watchlist, signals, alerts).
- `.schema alerts` — column layout.
- `SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT 20;`
- `SELECT * FROM signals;`
- `SELECT * FROM watchlist;`
- `.quit`

One-shot from the shell:
```bash
sqlite3 ~/.scanner_mcp/data.db "SELECT id, signal_id, symbol, triggered_at FROM alerts ORDER BY triggered_at DESC LIMIT 10;"
```

## Requirements

- Python 3.11+
- [Kaleido](https://github.com/plotly/Kaleido) for static PNG export; first chart run may install deps.

## Disclaimer

Not financial advice. Heuristic buy/hold/sell tags are for tooling only.

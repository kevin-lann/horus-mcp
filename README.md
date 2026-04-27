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

- `SCANNER_MCP_DB` — SQLite file path
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
- Signals: `list_signal_catalog`, `create_signal`, `list_signals`, `delete_signal`, `run_scan`
- Watchlist: `add_to_watchlist`, `remove_from_watchlist`, `get_watchlist` (symbols as JSON list strings for portability)
- Charts: `generate_chart` with `params` JSON — types: `price_history`, `price_overlay`, `forward_returns`, `drawdown_comparison`, `log_cycle`

## Resources

- `signals://triggered` — recent alerts (JSON)
- `signals://watchlist` — tickers (JSON)
- `research://forward-returns/{symbol}/{event_type}` — markdown table (`event_type`: `rsi_oversold` | `rsi_overbought`)

## Requirements

- Python 3.11+
- [Kaleido](https://github.com/plotly/Kaleido) for static PNG export; first chart run may install deps.

## Disclaimer

Not financial advice. Heuristic buy/hold/sell tags are for tooling only.

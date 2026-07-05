# Contributing to Horus MCP

This document explains how to set up a development environment, propose changes, and get them merged.

## Ways to contribute

- Report bugs or unexpected behavior via [GitHub Issues](https://github.com/kevin-lann/horus-mcp/issues).
- Suggest new signals, chart types, or tools.
- Improve documentation.
- Submit pull requests for bug fixes or features.

For anything larger than a small fix, please open an issue first so we can agree on the approach before you invest time.

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/kevin-lann/horus-mcp.git
cd horus-mcp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

See the [README](README.md#quick-setup) for Windows instructions and more detail.

## Running the server

```bash
horus-mcp
```

## Local Testing

### MCP Inspector

For interactive MCP debugging, first ensure the venv is activate with the above steps. Then run:

```bash
npx @modelcontextprotocol/inspector horus-mcp
```

### Unit Tests

Run the test suite:

```bash
python3 -m pytest
```

### Smoke Testing Individual Tools

The repo includes `test.py` for direct local testing without an MCP client:

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

Examples that modify the local SQLite database:

```bash
python3 test.py --tool watchlist --mutate --symbols AAPL MSFT
python3 test.py --tool create_signal --mutate --signal-name "debug rsi" --signal-type rsi_oversold --signal-params '{"threshold":100}' --signal-tickers --symbols AAPL MSFT
python3 test.py --tool scan
python3 test.py --tool scan --symbols AAPL MSFT
python3 test.py --tool delete_signal --mutate --signal-id 1
```

## SQLite Database

Open the default database from the command line:

```bash
sqlite3 ~/.scanner_mcp/data.db
```

Useful commands:

- `.tables`
- `.schema alerts`
- `SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT 20;`
- `SELECT * FROM signals;`
- `SELECT * FROM watchlist;`
- `.quit`

One-shot query:

```bash
sqlite3 ~/.scanner_mcp/data.db "SELECT id, signal_id, symbol, triggered_at FROM alerts ORDER BY triggered_at DESC LIMIT 10;"
```

Examples that write to the local SQLite database require the `--mutate` flag. See the [Local Testing](README.md#local-testing) section of the README for more examples.

## Pull request process

1. Fork the repository and create a branch off `master` (e.g. `fix/rsi-rounding` or `feat/pairs-chart`).
2. Make your change, keeping the style consistent with the surrounding code.
3. Add or update tests where it makes sense. New tools and signals should include coverage.
4. Ensure `python3 -m pytest` passes.
5. Update the README or other docs if your change affects setup, tools, or configuration.
6. Open a pull request with a clear description of what changed and why. Link any related issue.

## Guidelines

- Keep pull requests focused. Unrelated changes are easier to review as separate PRs.
- Do not commit secrets, API keys, or a populated `.env` file.
- The database lives at `~/.scanner_mcp/data.db` by default and should never be committed.
- Be respectful in issues and reviews.

## Reporting security issues

Do not open a public issue for security vulnerabilities. Follow the process in [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers this project.

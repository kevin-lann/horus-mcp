# Security Policy

## Supported versions

Horus MCP is pre-1.0 and under active development. Security fixes are applied to the latest release on the `master` branch only. Please upgrade to the latest version before reporting an issue.

| Version | Supported |
| --- | --- |
| latest `master` | Yes |
| older | No |

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/kevin-lann/horus-mcp/security) of the repository.
2. Click **Report a vulnerability** to open a private advisory.

Include as much of the following as you can:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof of concept.
- Affected version or commit.
- Any suggested remediation.

## Scope and hardening notes

Horus MCP runs locally over MCP `stdio` transport and stores data in a local SQLite database (`~/.scanner_mcp/data.db` by default). Keep the following in mind when deploying:

- Treat API keys such as `ALPHA_VANTAGE_API_KEY` as secrets. Configure them via environment variables and never commit them.
- Market data is retrieved from third-party providers and is not guaranteed to be accurate. This project is for research and tooling only and is not financial advice.
- Only connect the server to MCP clients you trust, since connected clients can invoke any exposed tool, including those that modify the local database.

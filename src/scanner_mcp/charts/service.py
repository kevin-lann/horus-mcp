"""Chart-to-MCP response helpers."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from fastmcp.utilities.types import Image

from scanner_mcp.charts import generate_chart
from scanner_mcp.data.provider import DataProvider


def chart_tool_result(provider: DataProvider, chart_type: str, params: dict[str, Any]) -> Image | str:
    """Run chart generation: MCP image block on success, JSON text on failure."""
    try:
        result = generate_chart(provider, chart_type, params)
        if not isinstance(result, dict):
            return json.dumps({"error": "unexpected chart response"})
        b64 = result.get("data")
        if not isinstance(b64, str):
            return json.dumps({"error": "chart response missing image data"})
        try:
            decoded = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return json.dumps({"error": "chart response missing image data"})
        if not decoded:
            return json.dumps({"error": "chart response missing image data"})
        return Image(data=decoded)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})

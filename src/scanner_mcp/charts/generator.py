"""Chart dispatcher for supported chart types."""

from __future__ import annotations

from typing import Any, Callable

from scanner_mcp.charts.comparison import (
    drawdown_comparison,
    log_cycle,
    price_overlay,
    ratio_chart,
    relative_strength_chart,
    sector_rotation_chart,
)
from scanner_mcp.charts.forward_returns import forward_returns_chart
from scanner_mcp.charts.fundamentals import fundamental_overlay
from scanner_mcp.charts.price_history import price_history
from scanner_mcp.data.provider import DataProvider

ChartBuilder = Callable[[DataProvider, dict[str, Any]], dict[str, str]]

_CHART_BUILDERS: dict[str, ChartBuilder] = {
    "price_history": price_history,
    "price_overlay": price_overlay,
    "ratio_chart": ratio_chart,
    "relative_strength": relative_strength_chart,
    "sector_rotation": sector_rotation_chart,
    "fundamental_overlay": fundamental_overlay,
    "forward_returns": forward_returns_chart,
    "drawdown_comparison": drawdown_comparison,
    "log_cycle": log_cycle,
}


def generate_chart(
    provider: DataProvider,
    chart_type: str,
    params: dict[str, Any],
) -> dict[str, str]:
    """Dispatch a supported chart type and return an MCP-safe image payload."""
    chart_key = chart_type.lower().strip()
    try:
        builder = _CHART_BUILDERS[chart_key]
    except KeyError as exc:
        raise ValueError(f"Unknown chart_type: {chart_type}") from exc
    return builder(provider, params)

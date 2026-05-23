"""Shared chart layout helpers."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def clean_layout(fig: go.Figure, *, title: str, yaxis_title: str, height: int = 680) -> None:
    """Apply a consistent clean chart style used by comparison-style presets."""
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        height=height,
        margin={"l": 60, "r": 80, "t": 78, "b": 48},
        legend={"orientation": "h", "x": 0.01, "xanchor": "left", "y": 1.08, "yanchor": "top"},
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,0.08)", zeroline=False)


def price_history_legend_layout() -> dict[str, Any]:
    """Place price-history legends above the plot area, not over the data."""
    return {
        "orientation": "h",
        "x": 0.02,
        "xanchor": "left",
        "y": 1.07,
        "yanchor": "top",
        "bgcolor": "rgba(255,255,255,0.85)",
    }

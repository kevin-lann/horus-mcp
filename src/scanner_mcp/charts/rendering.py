"""Plot rendering and optional debug-image persistence."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
import io
import logging
from pathlib import Path
from uuid import uuid4

import plotly.graph_objects as go

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_DIR = _PROJECT_ROOT / "output"


def save_debug_png(chart_type: str, png_bytes: bytes) -> None:
    """Persist a generated PNG under project-root/output for local debugging."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{chart_type}_{timestamp}_{uuid4().hex[:8]}.png"
    (_OUTPUT_DIR / filename).write_bytes(png_bytes)


def fig_to_b64(fig: go.Figure, chart_type: str) -> str:
    """Render a Plotly figure to PNG bytes, save a debug copy, and return base64."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", engine="kaleido", scale=1.5)
    buf.seek(0)
    png_bytes = buf.getvalue()
    try:
        save_debug_png(chart_type, png_bytes)
    except OSError:
        log.exception("Failed to save debug chart image")
    return base64.b64encode(png_bytes).decode("ascii")

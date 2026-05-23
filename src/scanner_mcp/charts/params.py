"""Shared chart parameter parsing helpers."""

from __future__ import annotations

from typing import Any


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like MCP/JSON values without treating all strings as true."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def positive_int(value: Any, default: int) -> int:
    """Parse a positive integer or fall back to the supplied default."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out > 0 else default

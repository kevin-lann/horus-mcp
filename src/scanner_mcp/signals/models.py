"""Dataclasses for in-memory signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActiveSignal:
    id: int
    name: str
    signal_type: str
    params: dict[str, Any]
    ticker_overrides: list[str] | None
    history_period: str = "1y"
    interval: str = "1d"

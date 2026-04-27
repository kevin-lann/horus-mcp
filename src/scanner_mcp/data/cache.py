"""Simple in-memory TTL cache for yfinance fetches."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires: float


class TTLCache(Generic[T]):
    def __init__(self, default_ttl: float) -> None:
        self._default_ttl = default_ttl
        self._data: dict[Hashable, _Entry[T]] = {}

    def get(self, key: Hashable) -> T | None:
        ent = self._data.get(key)
        if not ent:
            return None
        if time.monotonic() > ent.expires:
            del self._data[key]
            return None
        return ent.value

    def set(self, key: Hashable, value: T, ttl: float | None = None) -> None:
        t = ttl if ttl is not None else self._default_ttl
        self._data[key] = _Entry(value, time.monotonic() + t)

    def clear(self) -> None:
        self._data.clear()

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
    """Tiny monotonic-clock TTL cache with lazy expiration."""

    def __init__(self, default_ttl: float) -> None:
        """Create a cache whose entries live for `default_ttl` seconds by default."""
        self._default_ttl = default_ttl
        self._data: dict[Hashable, _Entry[T]] = {}

    def get(self, key: Hashable) -> T | None:
        """Return a cached value, or None when missing or expired."""
        ent = self._data.get(key)
        if not ent:
            return None
        if time.monotonic() > ent.expires:
            del self._data[key]
            return None
        return ent.value

    def set(self, key: Hashable, value: T, ttl: float | None = None) -> None:
        """Store a value using either an explicit TTL or the default TTL."""
        t = ttl if ttl is not None else self._default_ttl
        self._data[key] = _Entry(value, time.monotonic() + t)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._data.clear()

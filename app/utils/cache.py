"""Tiny in-memory TTL cache (insertion-ordered eviction)."""

from __future__ import annotations

import time
from typing import Callable, Generic, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """Stores values for ``ttl_seconds``; ``ttl_seconds <= 0`` disables caching."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int = 128,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._items: dict[str, tuple[float, V]] = {}

    def get(self, key: str) -> V | None:
        if self._ttl <= 0:
            return None
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at >= self._ttl:
            del self._items[key]
            return None
        return value

    def set(self, key: str, value: V) -> None:
        if self._ttl <= 0:
            return
        self._items.pop(key, None)
        self._items[key] = (self._clock(), value)
        while len(self._items) > self._max:
            self._items.pop(next(iter(self._items)))

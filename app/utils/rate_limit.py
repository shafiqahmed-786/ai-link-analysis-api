"""In-memory sliding-window rate limiter (per client key).

Protects the LLM budget of a public deployment.  State is per process: with
several workers each enforces its own limit, which is acceptable for this
service; use a shared store (e.g. Redis) if strict global limits are needed.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable


class SlidingWindowRateLimiter:
    """Allow at most ``limit`` hits per ``window_seconds`` for each key."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def check(self, key: str) -> float | None:
        """Record a hit. Return ``None`` if allowed, else seconds until a slot frees up."""
        if not self.enabled:
            return None
        now = self._clock()
        if len(self._hits) >= self._max_keys:
            self._purge(now)
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return max(0.0, self._window - (now - hits[0]))
        hits.append(now)
        return None

    def _purge(self, now: float) -> None:
        for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] >= self._window]:
            del self._hits[key]

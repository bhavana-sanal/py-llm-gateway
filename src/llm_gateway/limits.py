"""Per-key request rate limiting and monthly spend tracking."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime


class TokenBucket:
    """Allows bursts up to ``requests_per_minute`` and refills continuously at that rate."""

    def __init__(
        self, requests_per_minute: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._capacity = float(requests_per_minute)
        self._refill_per_second = requests_per_minute / 60.0
        self._clock = clock
        self._tokens = self._capacity
        self._updated = clock()

    def try_acquire(self) -> float:
        """Take one token. Returns 0.0 on success, otherwise the seconds until one is free."""
        now = self._clock()
        elapsed = now - self._updated
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
        self._updated = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        return (1.0 - self._tokens) / self._refill_per_second


class SpendLedger:
    """Month-to-date spend per key (calendar months in UTC), held in memory."""

    def __init__(self, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._now = now
        self._spend: defaultdict[tuple[str, str], float] = defaultdict(float)

    def _month(self) -> str:
        return self._now().strftime("%Y-%m")

    def add(self, key: str, usd: float) -> None:
        self._spend[key, self._month()] += usd

    def spent(self, key: str) -> float:
        return self._spend.get((key, self._month()), 0.0)

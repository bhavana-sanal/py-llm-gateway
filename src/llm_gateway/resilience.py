"""Retry backoff and a per-provider circuit breaker."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from enum import StrEnum

from llm_gateway.config import ResilienceConfig


def backoff_delay(
    config: ResilienceConfig, attempt: int, retry_after: float | None, rng: random.Random
) -> float | None:
    """Seconds to wait before retry number ``attempt + 1``, or None to stop retrying.

    An upstream ``Retry-After`` wins over our own schedule, but one longer than
    ``max_retry_after_seconds`` means falling through to the next provider is faster.
    Otherwise use exponential backoff with full jitter, which spreads out clients that
    failed at the same moment instead of having them retry in lockstep.
    """
    if retry_after is not None:
        return retry_after if retry_after <= config.max_retry_after_seconds else None
    ceiling = min(config.backoff_max_seconds, config.backoff_base_seconds * 2**attempt)
    return rng.uniform(0.0, ceiling)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stops sending traffic to a provider after consecutive failures.

    CLOSED counts consecutive failures and trips to OPEN at the threshold. OPEN rejects
    calls until ``reset_timeout`` has passed, then lets a single probe through as
    HALF_OPEN: success closes the circuit, failure re-opens it.
    """

    def __init__(
        self,
        failure_threshold: int,
        reset_timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow(self) -> bool:
        if self._state is CircuitState.CLOSED:
            return True
        if self._clock() - self._opened_at < self._reset_timeout:
            return False
        # Restarting the timer for the probe means a probe that never reports back
        # (say, the client disconnected) only blocks the provider for one more period.
        self._state = CircuitState.HALF_OPEN
        self._opened_at = self._clock()
        return True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._state is CircuitState.HALF_OPEN or self._failures >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()

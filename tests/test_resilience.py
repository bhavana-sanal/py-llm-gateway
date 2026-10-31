import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from llm_gateway.config import ResilienceConfig
from llm_gateway.providers.base import parse_retry_after
from llm_gateway.resilience import CircuitBreaker, CircuitState, backoff_delay


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10, clock=FakeClock())
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_success()  # a success resets the count
    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow()


def test_breaker_half_open_probe_closes_on_success() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    breaker.record_failure()
    clock.now = 9.9
    assert not breaker.allow()

    clock.now = 10.0
    assert breaker.allow()
    assert breaker.state is CircuitState.HALF_OPEN
    assert not breaker.allow(), "only one probe at a time"

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow()


def test_breaker_half_open_probe_failure_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout=10, clock=clock)
    for _ in range(5):
        breaker.record_failure()
    clock.now = 10.0
    assert breaker.allow()
    breaker.record_failure()  # one failure is enough while half-open
    assert breaker.state is CircuitState.OPEN
    clock.now = 19.9
    assert not breaker.allow()


def test_breaker_reissues_probe_that_never_reported() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=10, clock=clock)
    breaker.record_failure()
    clock.now = 10.0
    assert breaker.allow()
    clock.now = 20.0
    assert breaker.allow()


def test_backoff_uses_full_jitter_within_exponential_ceiling() -> None:
    config = ResilienceConfig(backoff_base_seconds=0.5, backoff_max_seconds=4.0)
    rng = random.Random(42)
    for attempt, ceiling in [(0, 0.5), (1, 1.0), (2, 2.0), (3, 4.0), (8, 4.0)]:
        delays = [backoff_delay(config, attempt, None, rng) for _ in range(200)]
        assert all(d is not None and 0 <= d <= ceiling for d in delays)
        assert max(d for d in delays if d is not None) > ceiling * 0.8


def test_backoff_prefers_retry_after_up_to_limit() -> None:
    config = ResilienceConfig(max_retry_after_seconds=30)
    rng = random.Random(0)
    assert backoff_delay(config, 0, 7.0, rng) == 7.0
    assert backoff_delay(config, 0, 31.0, rng) is None


def test_parse_retry_after() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    assert parse_retry_after(future) == pytest.approx(60, abs=2)
    past = format_datetime(datetime.now(UTC) - timedelta(seconds=60), usegmt=True)
    assert parse_retry_after(past) == 0.0

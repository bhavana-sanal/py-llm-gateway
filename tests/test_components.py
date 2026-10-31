"""Unit tests for the small building blocks: cache, limits, metrics and logging."""

import json
import logging
from datetime import UTC, datetime

import pytest

from llm_gateway.cache import TTLCache, cache_key
from llm_gateway.limits import SpendLedger, TokenBucket
from llm_gateway.logs import JsonFormatter, request_id
from llm_gateway.metrics import Counter, Histogram


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_key_ignores_transport_fields_and_order() -> None:
    a = {"model": "m", "messages": [], "temperature": 0, "stream": False, "user": "u1"}
    b = {"temperature": 0, "messages": [], "model": "m"}
    assert cache_key("alice", a) == cache_key("alice", b)
    assert cache_key("alice", a) != cache_key("bob", a)
    assert cache_key("alice", a) != cache_key("alice", {**b, "temperature": 0.1})


def test_cache_expires_entries() -> None:
    clock = FakeClock()
    cache: TTLCache[str] = TTLCache(max_entries=10, ttl_seconds=5, clock=clock)
    cache.put("k", "v")
    clock.now = 4.9
    assert cache.get("k") == "v"
    clock.now = 5.0
    assert cache.get("k") is None
    assert len(cache) == 0


def test_cache_evicts_least_recently_used() -> None:
    cache: TTLCache[int] = TTLCache(max_entries=2, ttl_seconds=60, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # "b" is now the least recently used
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


def test_token_bucket_bursts_then_refills() -> None:
    clock = FakeClock()
    bucket = TokenBucket(requests_per_minute=3, clock=clock)
    assert [bucket.try_acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.try_acquire() == pytest.approx(20.0)  # one token every 20 s
    clock.now = 20.0
    assert bucket.try_acquire() == 0.0
    clock.now = 1000.0
    assert [bucket.try_acquire() for _ in range(4)][-1] > 0, "capacity caps the burst"


def test_spend_ledger_resets_each_month() -> None:
    now = datetime(2026, 1, 31, 23, 59, tzinfo=UTC)
    ledger = SpendLedger(now=lambda: now)
    ledger.add("alice", 1.5)
    ledger.add("alice", 0.5)
    assert ledger.spent("alice") == 2.0
    assert ledger.spent("bob") == 0.0
    now = datetime(2026, 2, 1, tzinfo=UTC)
    assert ledger.spent("alice") == 0.0


def test_counter_rendering_escapes_labels() -> None:
    counter = Counter("demo_total", "Demo.", ("name",))
    counter.inc(name='a "quoted"\nvalue\\')
    assert list(counter.render()) == [
        "# HELP demo_total Demo.",
        "# TYPE demo_total counter",
        r'demo_total{name="a \"quoted\"\nvalue\\"} 1.0',
    ]
    with pytest.raises(ValueError, match="labels"):
        counter.inc(other="x")
    with pytest.raises(ValueError, match="increase"):
        counter.inc(-1, name="x")


def test_histogram_buckets_are_cumulative() -> None:
    histogram = Histogram("latency_seconds", "Latency.", buckets=(0.1, 1.0))
    for value in (0.05, 0.5, 5.0):
        histogram.observe(value)
    assert list(histogram.render())[2:] == [
        'latency_seconds_bucket{le="0.1"} 1',
        'latency_seconds_bucket{le="1.0"} 2',
        'latency_seconds_bucket{le="+Inf"} 3',
        "latency_seconds_sum 5.55",
        "latency_seconds_count 3",
    ]


def test_json_log_lines_carry_request_id_and_extras() -> None:
    token = request_id.set("req-42")
    try:
        record = logging.makeLogRecord(
            {"name": "gw", "levelno": logging.INFO, "levelname": "INFO", "msg": "usage %s"}
        )
        record.args = ("ok",)
        record.cost_usd = 0.5
        entry = json.loads(JsonFormatter().format(record))
    finally:
        request_id.reset(token)
    assert entry["msg"] == "usage ok"
    assert entry["request_id"] == "req-42"
    assert entry["cost_usd"] == 0.5
    assert entry["level"] == "info"

import json
from typing import Any

import httpx
import pytest

from llm_gateway.config import Target
from llm_gateway.metrics import Metrics
from llm_gateway.providers import ProviderError, build_provider
from llm_gateway.router import AllProvidersFailed, Router

from helpers import SleepRecorder, Upstream, completion_body, make_config, sse_response

PAYLOAD = {"model": "chain", "messages": [{"role": "user", "content": "hi"}]}
PRIMARY = Target(provider="primary", model="gpt-5-mini")
BACKUP = Target(provider="backup", model="llama3.3")


def make_router(
    upstream: Upstream, sleep: SleepRecorder, **resilience: Any
) -> tuple[Router, Metrics]:
    config = make_config(resilience={"max_retries": 2, **resilience})
    providers = {
        name: build_provider(name, provider, upstream.transport)
        for name, provider in config.providers.items()
    }
    metrics = Metrics()
    return Router(providers, config.models, config.resilience, metrics, sleep=sleep), metrics


def ok(content: str = "hi") -> httpx.Response:
    return httpx.Response(200, json=completion_body(content))


def status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, json={"error": {"message": f"status {code}"}}, headers=headers)


async def test_falls_back_after_retrying_503() -> None:
    upstream = Upstream({"primary.test": [status(503)] * 3, "backup.test": [ok("from backup")]})
    sleep = SleepRecorder()
    router, metrics = make_router(upstream, sleep)

    target, body = await router.complete("chain", PAYLOAD)

    assert target == BACKUP
    assert body["choices"][0]["message"]["content"] == "from backup"
    assert upstream.calls("primary.test") == 3
    assert len(sleep.delays) == 2
    assert metrics.provider_errors.value(provider="primary", reason="503") == 3
    assert json.loads(upstream.requests[-1].content)["model"] == "llama3.3"


async def test_retry_honors_retry_after() -> None:
    upstream = Upstream({"primary.test": [status(429, **{"retry-after": "3"}), ok()]})
    sleep = SleepRecorder()
    router, _ = make_router(upstream, sleep)

    target, _ = await router.complete("chain", PAYLOAD)

    assert target == PRIMARY
    assert sleep.delays == [3.0]


async def test_long_retry_after_falls_through_without_waiting() -> None:
    upstream = Upstream(
        {"primary.test": [status(429, **{"retry-after": "120"})], "backup.test": [ok()]}
    )
    sleep = SleepRecorder()
    router, _ = make_router(upstream, sleep, max_retry_after_seconds=30)

    target, _ = await router.complete("chain", PAYLOAD)

    assert target == BACKUP
    assert sleep.delays == []


async def test_timeouts_are_retried() -> None:
    upstream = Upstream({"primary.test": [httpx.ReadTimeout("slow"), ok()]})
    router, metrics = make_router(upstream, SleepRecorder())

    target, _ = await router.complete("chain", PAYLOAD)

    assert target == PRIMARY
    assert metrics.provider_errors.value(provider="primary", reason="timeout") == 1


async def test_client_errors_are_not_retried() -> None:
    upstream = Upstream({"primary.test": [status(400)], "backup.test": [status(400)]})
    sleep = SleepRecorder()
    router, _ = make_router(upstream, sleep)

    with pytest.raises(AllProvidersFailed) as info:
        await router.complete("chain", PAYLOAD)

    assert [e.status for e in info.value.errors] == [400, 400]
    assert sleep.delays == []
    assert router.breakers["primary"].allow(), "a 400 says nothing about provider health"


async def test_open_circuit_skips_provider() -> None:
    upstream = Upstream(
        {"primary.test": [status(500), status(500)], "backup.test": [ok(), ok(), ok()]}
    )
    router, metrics = make_router(
        upstream, SleepRecorder(), max_retries=0, circuit_failure_threshold=2
    )

    for _ in range(3):
        target, _ = await router.complete("chain", PAYLOAD)
        assert target == BACKUP

    assert upstream.calls("primary.test") == 2
    assert metrics.provider_errors.value(provider="primary", reason="circuit_open") == 1


async def test_stream_falls_back_before_first_chunk() -> None:
    chunk = {"id": "c", "choices": [{"index": 0, "delta": {"content": "ok"}}]}
    upstream = Upstream(
        {"primary.test": [status(502)], "backup.test": [sse_response(chunk, "[DONE]")]}
    )
    router, _ = make_router(upstream, SleepRecorder(), max_retries=0)

    target, chunks = await router.open_stream("chain", PAYLOAD)

    assert target == BACKUP
    assert [c async for c in chunks] == [chunk]


async def test_stream_failure_after_first_chunk_is_not_retried() -> None:
    first = {"id": "c", "choices": [{"index": 0, "delta": {"content": "par"}}]}

    async def broken_body() -> Any:
        yield f"data: {json.dumps(first)}\n\n".encode()
        raise httpx.ReadError("connection reset")

    reply = httpx.Response(
        200, content=broken_body(), headers={"content-type": "text/event-stream"}
    )
    upstream = Upstream({"primary.test": [reply], "backup.test": [ok()]})
    router, metrics = make_router(upstream, SleepRecorder())

    target, chunks = await router.open_stream("chain", PAYLOAD)
    received = [await anext(chunks)]
    with pytest.raises(ProviderError, match="connection"):
        await anext(chunks)

    assert target == PRIMARY
    assert received == [first]
    assert upstream.calls("backup.test") == 0
    assert metrics.provider_errors.value(provider="primary", reason="connection") == 1

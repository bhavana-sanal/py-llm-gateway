import json
from typing import Any

import httpx
import pytest

from helpers import API_KEY, Upstream, completion_body, gateway_client, make_config, sse_response

HELLO = [{"role": "user", "content": "Hello there"}]


def chat(model: str = "mock", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": HELLO, **extra}


def sse_events(response: httpx.Response) -> list[str]:
    frames = response.text.split("\n\n")
    assert frames[-1] == "", "stream must end with a blank line"
    assert all(f.startswith("data: ") for f in frames[:-1])
    return [f.removeprefix("data: ") for f in frames[:-1]]


# --- auth and validation -------------------------------------------------------------


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", f"Basic {API_KEY}", "Bearer"])
async def test_rejects_missing_or_invalid_key(authorization: str | None) -> None:
    async with gateway_client() as client:
        headers = {"authorization": authorization} if authorization else {}
        client.headers.pop("authorization")
        response = await client.post("/v1/chat/completions", json=chat(), headers=headers)

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "message": "Missing or invalid API key. Send it as 'Authorization: Bearer <key>'.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }


async def test_auth_is_checked_before_body_validation() -> None:
    async with gateway_client() as client:
        client.headers.pop("authorization")
        response = await client.post("/v1/chat/completions", json={"nonsense": True})
    assert response.status_code == 401


async def test_invalid_body_gets_openai_style_400() -> None:
    async with gateway_client() as client:
        response = await client.post("/v1/chat/completions", json={"model": "mock", "messages": []})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"


async def test_unknown_model_is_404() -> None:
    async with gateway_client() as client:
        response = await client.post("/v1/chat/completions", json=chat("gpt-99"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_lists_model_aliases() -> None:
    async with gateway_client() as client:
        response = await client.get("/v1/models")
    assert [m["id"] for m in response.json()["data"]] == ["mock", "chain", "claude"]


# --- completions ---------------------------------------------------------------------


async def test_non_streaming_completion_with_mock() -> None:
    async with gateway_client() as client:
        response = await client.post(
            "/v1/chat/completions", json=chat(), headers={"x-request-id": "req-123"}
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-gateway-target"] == "mock/mock-1"
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Mock response to: Hello there"


async def test_streaming_uses_sse_framing_and_hides_usage_by_default() -> None:
    async with gateway_client() as client:
        response = await client.post("/v1/chat/completions", json=chat(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["choices"] and "usage" not in c for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Mock response to: Hello there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_includes_usage_when_requested() -> None:
    async with gateway_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json=chat(stream=True, stream_options={"include_usage": True}),
        )
    events = sse_events(response)
    usage_chunk = json.loads(events[-2])
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["total_tokens"] > 0


async def test_anthropic_stream_is_translated_end_to_end() -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-5-5"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    upstream = Upstream({"claude.test": [sse_response(*events)]})
    async with gateway_client(upstream=upstream) as client:
        response = await client.post("/v1/chat/completions", json=chat("claude", stream=True))

    chunks = [json.loads(e) for e in sse_events(response)[:-1]]
    assert [c["choices"][0]["delta"] for c in chunks] == [
        {"role": "assistant", "content": ""},
        {"content": "Hi"},
        {},
    ]


async def test_mid_stream_failure_is_reported_in_band() -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "m"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    upstream = Upstream({"claude.test": [sse_response(*events)]})
    async with gateway_client(upstream=upstream) as client:
        response = await client.post("/v1/chat/completions", json=chat("claude", stream=True))

    assert response.status_code == 200
    events_out = sse_events(response)
    assert "[DONE]" not in events_out
    assert json.loads(events_out[-1])["error"]["code"] == "stream_error"


async def test_all_providers_failing_returns_502() -> None:
    failure = httpx.Response(503, json={"error": {"message": "down"}})
    upstream = Upstream({"primary.test": [failure] * 3, "backup.test": [failure] * 3})
    async with gateway_client(upstream=upstream) as client:
        response = await client.post("/v1/chat/completions", json=chat("chain"))

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


async def test_upstream_bad_request_is_passed_on() -> None:
    rejection = httpx.Response(400, json={"error": {"message": "bad param"}})
    upstream = Upstream({"primary.test": [rejection], "backup.test": [rejection]})
    async with gateway_client(upstream=upstream) as client:
        response = await client.post("/v1/chat/completions", json=chat("chain"))

    assert response.status_code == 400
    assert "bad param" in response.json()["error"]["message"]


# --- limits --------------------------------------------------------------------------


async def test_rate_limit_per_key() -> None:
    config = make_config(keys=[{"name": "alice", "key": API_KEY, "requests_per_minute": 2}])
    async with gateway_client(config) as client:
        statuses = [
            (await client.post("/v1/chat/completions", json=chat())).status_code for _ in range(2)
        ]
        limited = await client.post("/v1/chat/completions", json=chat())

    assert statuses == [200, 200]
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(limited.headers["retry-after"]) >= 1


async def test_monthly_budget() -> None:
    config = make_config(
        keys=[{"name": "alice", "key": API_KEY, "monthly_budget_usd": 0.00001}],
    )
    upstream = Upstream({"primary.test": [httpx.Response(200, json=completion_body())]})
    async with gateway_client(config, upstream) as client:
        first = await client.post("/v1/chat/completions", json=chat("chain"))
        second = await client.post("/v1/chat/completions", json=chat("chain"))

    assert first.status_code == 200  # 10 * $1/M + 5 * $2/M = $0.00002 spent
    assert second.status_code == 429
    assert second.json()["error"]["type"] == "insufficient_quota"


# --- cache ---------------------------------------------------------------------------


async def test_deterministic_requests_hit_the_cache() -> None:
    upstream = Upstream({"primary.test": [httpx.Response(200, json=completion_body())]})
    async with gateway_client(upstream=upstream) as client:
        first = await client.post("/v1/chat/completions", json=chat("chain", temperature=0))
        second = await client.post("/v1/chat/completions", json=chat("chain", temperature=0))
        metrics = (await client.get("/metrics")).text

    assert (first.headers["x-cache"], second.headers["x-cache"]) == ("miss", "hit")
    assert first.json() == second.json()
    assert upstream.calls("primary.test") == 1
    assert "llm_gateway_cache_hits_total 1.0" in metrics
    assert "llm_gateway_cache_misses_total 1.0" in metrics


@pytest.mark.parametrize("extra", [{}, {"temperature": 0.7}])
async def test_sampling_requests_bypass_the_cache(extra: dict[str, Any]) -> None:
    async with gateway_client() as client:
        responses = [
            await client.post("/v1/chat/completions", json=chat(**extra)) for _ in range(2)
        ]
    assert [r.headers["x-cache"] for r in responses] == ["miss", "miss"]


async def test_cache_can_include_nonzero_temperature() -> None:
    config = make_config(cache={"cache_nonzero_temperature": True})
    async with gateway_client(config) as client:
        responses = [
            await client.post("/v1/chat/completions", json=chat(temperature=0.7)) for _ in range(2)
        ]
    assert [r.headers["x-cache"] for r in responses] == ["miss", "hit"]


# --- observability -------------------------------------------------------------------


async def test_metrics_exposition() -> None:
    upstream = Upstream({"primary.test": [httpx.Response(200, json=completion_body())]})
    async with gateway_client(upstream=upstream) as client:
        await client.post("/v1/chat/completions", json=chat("chain"))
        await client.post("/v1/chat/completions", json=chat("nope"))
        response = await client.get("/metrics")

    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    lines = response.text.splitlines()
    labels = 'key="alice",provider="primary",model="gpt-5-mini"'
    expected = [
        'llm_gateway_requests_total{key="alice",model="chain",status="200"} 1.0',
        'llm_gateway_requests_total{key="alice",model="-",status="404"} 1.0',
        f'llm_gateway_tokens_total{{{labels},type="prompt"}} 10.0',
        f'llm_gateway_tokens_total{{{labels},type="completion"}} 5.0',
        'llm_gateway_cost_usd_total{key="alice",provider="primary",model="gpt-5-mini"} 2e-05',
        'llm_gateway_request_duration_seconds_count{model="chain"} 1',
        'llm_gateway_circuit_breaker_open{provider="primary"} 0.0',
        "# TYPE llm_gateway_request_duration_seconds histogram",
    ]
    for line in expected:
        assert line in lines
    assert 'llm_gateway_request_duration_seconds_bucket{model="chain",le="+Inf"} 1' in lines


async def test_healthz() -> None:
    async with gateway_client() as client:
        response = await client.get("/healthz")
    assert response.json() == {"status": "ok"}

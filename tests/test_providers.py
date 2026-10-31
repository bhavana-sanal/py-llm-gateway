import json
from typing import Any

import httpx
import pytest

from llm_gateway.providers.anthropic import AnthropicProvider, to_anthropic_request
from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.openai import OpenAIProvider

from helpers import Upstream, completion_body, sse_response

MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "developer", "content": [{"type": "text", "text": "No emoji."}]},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {
        "role": "user",
        "content": [{"type": "text", "text": "How "}, {"type": "text", "text": "are you?"}],
    },
]


def openai_provider(upstream: Upstream) -> OpenAIProvider:
    return OpenAIProvider(
        "primary",
        base_url="https://primary.test/v1",
        api_key="sk-up",
        timeout=5,
        transport=upstream.transport,
    )


def anthropic_provider(upstream: Upstream) -> AnthropicProvider:
    return AnthropicProvider(
        "claude",
        base_url="https://claude.test",
        api_key="sk-ant-test",
        timeout=5,
        default_max_tokens=1024,
        transport=upstream.transport,
    )


def sent_json(request: httpx.Request) -> Any:
    return json.loads(request.content)


async def collect(provider: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [chunk async for chunk in provider.stream(payload)]


# --- OpenAI-compatible ---------------------------------------------------------------


async def test_openai_complete_passes_request_through() -> None:
    upstream = Upstream({"primary.test": [httpx.Response(200, json=completion_body())]})
    payload = {"model": "gpt-5-mini", "messages": MESSAGES, "temperature": 0, "seed": 7}

    result = await openai_provider(upstream).complete(payload)

    request = upstream.requests[0]
    assert str(request.url) == "https://primary.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-up"
    assert sent_json(request) == {**payload, "stream": False}
    assert result == completion_body()


async def test_openai_stream_requests_usage_and_stops_at_done() -> None:
    chunk = {"id": "c", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "x"}}]}
    usage = {"id": "c", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
    upstream = Upstream({"primary.test": [sse_response(chunk, usage, "[DONE]")]})

    chunks = await collect(openai_provider(upstream), {"model": "m", "messages": MESSAGES})

    assert chunks == [chunk, usage]
    body = sent_json(upstream.requests[0])
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_openai_error_carries_status_and_retry_after() -> None:
    error = {"error": {"message": "slow down", "type": "rate_limit"}}
    reply = httpx.Response(429, json=error, headers={"retry-after": "2"})
    upstream = Upstream({"primary.test": [reply]})

    with pytest.raises(ProviderError) as info:
        await openai_provider(upstream).complete({"model": "m", "messages": MESSAGES})

    assert info.value.status == 429
    assert info.value.retry_after == 2.0
    assert info.value.retryable
    assert "slow down" in str(info.value)


# --- Anthropic -----------------------------------------------------------------------


def test_anthropic_request_translation() -> None:
    payload = {
        "model": "claude-sonnet-5-5",
        "messages": MESSAGES,
        "temperature": 0.2,
        "stop": "END",
        "max_tokens": 50,
        "presence_penalty": 1,  # no Anthropic equivalent; dropped
    }
    assert to_anthropic_request(payload, default_max_tokens=1024, provider="claude") == {
        "model": "claude-sonnet-5-5",
        "system": "Be brief.\n\nNo emoji.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 50,
        "temperature": 0.2,
        "stop_sequences": ["END"],
    }


def test_anthropic_request_defaults_max_tokens() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body = to_anthropic_request(payload, default_max_tokens=1024, provider="claude")
    assert body["max_tokens"] == 1024
    assert "system" not in body


@pytest.mark.parametrize(
    "message",
    [
        {"role": "tool", "content": "42", "tool_call_id": "t1"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
    ],
)
def test_anthropic_rejects_unsupported_input(message: dict[str, Any]) -> None:
    with pytest.raises(ProviderError) as info:
        to_anthropic_request(
            {"model": "m", "messages": [message]}, default_max_tokens=1, provider="c"
        )
    assert info.value.status == 400
    assert not info.value.retryable


async def test_anthropic_complete_translates_both_ways() -> None:
    reply = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "Doing "}, {"type": "text", "text": "well."}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 20, "cache_read_input_tokens": 5, "output_tokens": 4},
    }
    upstream = Upstream({"claude.test": [httpx.Response(200, json=reply)]})

    result = await anthropic_provider(upstream).complete(
        {"model": "claude-sonnet-5-5", "messages": MESSAGES}
    )

    request = upstream.requests[0]
    assert str(request.url) == "https://claude.test/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in request.headers
    assert result["object"] == "chat.completion"
    assert result["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Doing well."},
            "finish_reason": "length",
        }
    ]
    assert result["usage"] == {"prompt_tokens": 25, "completion_tokens": 4, "total_tokens": 29}


async def test_anthropic_stream_translates_events() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude-sonnet-5-5", "usage": {"input_tokens": 9}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    upstream = Upstream({"claude.test": [sse_response(*events)]})

    chunks = await collect(anthropic_provider(upstream), {"model": "m", "messages": MESSAGES})

    assert sent_json(upstream.requests[0])["stream"] is True
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == "msg_1" for c in chunks)
    assert [c["choices"][0]["delta"] for c in chunks[:4]] == [
        {"role": "assistant", "content": ""},
        {"content": "Hel"},
        {"content": "lo"},
        {},
    ]
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"
    assert chunks[4]["choices"] == []
    assert chunks[4]["usage"] == {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}


async def test_anthropic_stream_error_event_raises() -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "model": "m", "usage": {}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    upstream = Upstream({"claude.test": [sse_response(*events)]})
    with pytest.raises(ProviderError, match="overloaded_error"):
        await collect(anthropic_provider(upstream), {"model": "m", "messages": MESSAGES})


# --- Mock ----------------------------------------------------------------------------


async def test_mock_is_deterministic_and_streams_same_text() -> None:
    provider = MockProvider("mock")
    payload = {"model": "mock-1", "messages": MESSAGES}

    first, second = await provider.complete(payload), await provider.complete(payload)
    chunks = await collect(provider, payload)

    text = first["choices"][0]["message"]["content"]
    assert text == "Mock response to: How are you?"
    assert first["id"] == second["id"]
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert streamed == text
    assert chunks[-1]["usage"] == first["usage"]

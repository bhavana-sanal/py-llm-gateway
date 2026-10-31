"""Test helpers: a config factory, scripted fake upstreams and an in-process client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llm_gateway.app import create_app
from llm_gateway.config import GatewayConfig

API_KEY = "sk-test-alice"

type Reply = httpx.Response | Exception


def make_config(**overrides: Any) -> GatewayConfig:
    data: dict[str, Any] = {
        "providers": {
            "mock": {"type": "mock"},
            "primary": {"type": "openai", "base_url": "https://primary.test/v1", "api_key": "k1"},
            "backup": {"type": "openai", "base_url": "https://backup.test/v1", "api_key": "k2"},
            "claude": {"type": "anthropic", "base_url": "https://claude.test", "api_key": "k3"},
        },
        "models": {
            "mock": ["mock/mock-1"],
            "chain": ["primary/gpt-5-mini", "backup/llama3.3"],
            "claude": ["claude/claude-sonnet-5-5"],
        },
        "keys": [{"name": "alice", "key": API_KEY, "requests_per_minute": 1000}],
        "resilience": {"max_retries": 2, "circuit_failure_threshold": 5},
        "pricing": {"gpt-5-mini": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
    }
    data.update(overrides)
    return GatewayConfig.model_validate(data)


def completion_body(content: str = "hi", model: str = "gpt-5-mini") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def sse_response(*events: dict[str, Any] | str) -> httpx.Response:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return httpx.Response(
        200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
    )


class Upstream:
    """A fake set of upstream servers: replies are scripted per host and consumed in order."""

    def __init__(self, replies: dict[str, list[Reply]]) -> None:
        self._replies = {host: list(queue) for host, queue in replies.items()}
        self.requests: list[httpx.Request] = []

    def calls(self, host: str) -> int:
        return sum(1 for r in self.requests if r.url.host == host)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self._replies[request.url.host].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@asynccontextmanager
async def gateway_client(
    config: GatewayConfig | None = None,
    upstream: Upstream | None = None,
    sleep: SleepRecorder | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        config or make_config(),
        transport=upstream.transport if upstream else None,
        sleep=sleep or SleepRecorder(),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers={"authorization": f"Bearer {API_KEY}"},
        ) as client,
    ):
        yield client

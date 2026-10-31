"""Adapter translating between OpenAI chat completions and the Anthropic Messages API."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_gateway.providers.base import (
    JSONObject,
    ProviderError,
    chunk_object,
    message_text,
    raise_for_status,
    sse_data,
    usage_object,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_PASSTHROUGH_PARAMS = ("temperature", "top_p")


def finish_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return FINISH_REASONS.get(stop_reason, "stop")


def to_anthropic_request(
    payload: JSONObject, *, default_max_tokens: int, provider: str
) -> JSONObject:
    system: list[str] = []
    messages: list[JSONObject] = []
    for message in payload["messages"]:
        role = message["role"]
        text = message_text(message.get("content"), provider)
        if role in ("system", "developer"):
            # Anthropic takes instructions as a top-level field, not as a message.
            system.append(text)
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
        else:
            raise ProviderError(provider, f"unsupported message role {role!r}", status=400)

    body: JSONObject = {
        "model": payload["model"],
        "messages": messages,
        "max_tokens": payload.get("max_completion_tokens")
        or payload.get("max_tokens")
        or default_max_tokens,
    }
    if system:
        body["system"] = "\n\n".join(system)
    for param in _PASSTHROUGH_PARAMS:
        if payload.get(param) is not None:
            body[param] = payload[param]
    stop = payload.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    return body


def _prompt_tokens(usage: dict[str, Any]) -> int:
    # Prompt-cache reads and writes are billed input that ``input_tokens`` excludes.
    return sum(
        int(usage.get(field) or 0)
        for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )


def from_anthropic_response(body: JSONObject) -> JSONObject:
    text = "".join(block["text"] for block in body.get("content", []) if block["type"] == "text")
    usage = body.get("usage", {})
    return {
        "id": body["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason(body.get("stop_reason")),
            }
        ],
        "usage": usage_object(_prompt_tokens(usage), int(usage.get("output_tokens", 0))),
    }


class AnthropicProvider:
    def __init__(
        self,
        name: str,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float,
        default_max_tokens: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._default_max_tokens = default_max_tokens
        headers = {"anthropic-version": API_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url or DEFAULT_BASE_URL,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def _request(self, payload: JSONObject) -> JSONObject:
        return to_anthropic_request(
            payload, default_max_tokens=self._default_max_tokens, provider=self.name
        )

    async def complete(self, payload: JSONObject) -> JSONObject:
        response = await self._client.post("/v1/messages", json=self._request(payload))
        await raise_for_status(response, self.name)
        return from_anthropic_response(response.json())

    async def stream(self, payload: JSONObject) -> AsyncGenerator[JSONObject]:
        body = {**self._request(payload), "stream": True}
        chunk_id, model, created = "", payload["model"], int(time.time())
        prompt_tokens = completion_tokens = 0
        async with self._client.stream("POST", "/v1/messages", json=body) as response:
            await raise_for_status(response, self.name)
            async for data in sse_data(response):
                event = json.loads(data)
                match event["type"]:
                    case "message_start":
                        message = event["message"]
                        chunk_id, model = message["id"], message["model"]
                        prompt_tokens = _prompt_tokens(message.get("usage", {}))
                        yield chunk_object(
                            chunk_id, model, created, delta={"role": "assistant", "content": ""}
                        )
                    case "content_block_delta" if event["delta"]["type"] == "text_delta":
                        yield chunk_object(
                            chunk_id, model, created, delta={"content": event["delta"]["text"]}
                        )
                    case "message_delta":
                        completion_tokens = int(event.get("usage", {}).get("output_tokens", 0))
                        yield chunk_object(
                            chunk_id,
                            model,
                            created,
                            delta={},
                            finish_reason=finish_reason(event["delta"].get("stop_reason")),
                        )
                    case "message_stop":
                        usage = usage_object(prompt_tokens, completion_tokens)
                        yield chunk_object(chunk_id, model, created, usage=usage)
                        return
                    case "error":
                        error = event.get("error", {})
                        raise ProviderError(
                            self.name,
                            f"stream error {error.get('type')}: {error.get('message')}",
                            reason="stream_error",
                        )

    async def aclose(self) -> None:
        await self._client.aclose()

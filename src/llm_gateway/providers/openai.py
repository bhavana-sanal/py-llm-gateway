"""Adapter for OpenAI and any server exposing the same API (Ollama, vLLM, Groq, ...)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import httpx

from llm_gateway.providers.base import JSONObject, raise_for_status, sse_data

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    def __init__(
        self,
        name: str,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url or DEFAULT_BASE_URL,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def complete(self, payload: JSONObject) -> JSONObject:
        body = {**payload, "stream": False}
        body.pop("stream_options", None)
        response = await self._client.post("/chat/completions", json=body)
        await raise_for_status(response, self.name)
        result: JSONObject = response.json()
        return result

    async def stream(self, payload: JSONObject) -> AsyncGenerator[JSONObject]:
        # Always ask for the trailing usage chunk: the gateway needs it for cost accounting
        # and strips it again if the client did not request it.
        options = {**payload.get("stream_options", {}), "include_usage": True}
        body = {**payload, "stream": True, "stream_options": options}
        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            await raise_for_status(response, self.name)
            async for data in sse_data(response):
                if data == "[DONE]":
                    return
                yield json.loads(data)

    async def aclose(self) -> None:
        await self._client.aclose()

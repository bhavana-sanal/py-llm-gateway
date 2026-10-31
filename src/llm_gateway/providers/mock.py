"""A deterministic offline provider, for demos and tests that need no API keys."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncGenerator

from llm_gateway.providers.base import JSONObject, chunk_object, message_text, usage_object


class MockProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def _reply(self, payload: JSONObject) -> tuple[str, str, int, int]:
        """Return (id, reply text, prompt tokens, completion tokens); tokens are word counts."""
        texts = [
            (m["role"], message_text(m.get("content"), self.name)) for m in payload["messages"]
        ]
        last_user = next((text for role, text in reversed(texts) if role == "user"), "")
        reply = f"Mock response to: {last_user}".strip()
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        prompt_tokens = sum(len(text.split()) for _, text in texts)
        return f"chatcmpl-mock-{digest[:16]}", reply, prompt_tokens, len(reply.split())

    async def complete(self, payload: JSONObject) -> JSONObject:
        reply_id, reply, prompt_tokens, completion_tokens = self._reply(payload)
        return {
            "id": reply_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_object(prompt_tokens, completion_tokens),
        }

    async def stream(self, payload: JSONObject) -> AsyncGenerator[JSONObject]:
        reply_id, reply, prompt_tokens, completion_tokens = self._reply(payload)
        model, created = payload["model"], int(time.time())
        yield chunk_object(reply_id, model, created, delta={"role": "assistant", "content": ""})
        words = reply.split(" ")
        for i, word in enumerate(words):
            piece = word if i == len(words) - 1 else f"{word} "
            yield chunk_object(reply_id, model, created, delta={"content": piece})
        yield chunk_object(reply_id, model, created, delta={}, finish_reason="stop")
        usage = usage_object(prompt_tokens, completion_tokens)
        yield chunk_object(reply_id, model, created, usage=usage)

    async def aclose(self) -> None:
        return None

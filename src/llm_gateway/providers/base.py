"""The provider protocol and helpers shared by the HTTP adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import httpx

type JSONObject = dict[str, Any]


class ProviderError(Exception):
    """An upstream call failed. ``status`` is None for network-level failures."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status = status
        self.retry_after = retry_after
        self.reason = reason or (str(status) if status is not None else "error")

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in (408, 429) or self.status >= 500


class Provider(Protocol):
    """An upstream LLM API, spoken to in OpenAI chat-completions format on both sides."""

    name: str

    async def complete(self, payload: JSONObject) -> JSONObject: ...

    def stream(self, payload: JSONObject) -> AsyncGenerator[JSONObject]:
        """Yield ``chat.completion.chunk`` objects, ending with a usage-only chunk."""
        ...

    async def aclose(self) -> None: ...


def network_error(provider: str, exc: httpx.TransportError) -> ProviderError:
    reason = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection"
    return ProviderError(provider, f"{reason} error: {exc!r}", reason=reason)


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header given either as delta-seconds or as an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


async def raise_for_status(response: httpx.Response, provider: str) -> None:
    if response.is_success:
        return
    await response.aread()
    try:
        message = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        message = response.text[:500] or response.reason_phrase
    raise ProviderError(
        provider,
        f"upstream returned {response.status_code}: {message}",
        status=response.status_code,
        retry_after=parse_retry_after(response.headers.get("retry-after")),
    )


async def sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the ``data`` payload of each server-sent event (single-line events only)."""
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            yield line.removeprefix("data:").removeprefix(" ")


def message_text(content: Any, provider: str) -> str:
    """Flatten OpenAI message content (a string or a list of text parts) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise ProviderError(provider, f"unsupported content part {kind!r}", status=400)
        texts.append(str(part.get("text", "")))
    return "".join(texts)


def usage_object(prompt_tokens: int, completion_tokens: int) -> JSONObject:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chunk_object(
    chunk_id: str,
    model: str,
    created: int,
    *,
    delta: JSONObject | None = None,
    finish_reason: str | None = None,
    usage: JSONObject | None = None,
) -> JSONObject:
    choices = (
        [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    )
    chunk: JSONObject = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk

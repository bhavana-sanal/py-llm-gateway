"""The gateway core: auth, limits, caching, routing and accounting, independent of HTTP."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator
from typing import NamedTuple

import httpx

from llm_gateway.cache import TTLCache, cache_key
from llm_gateway.config import GatewayConfig, KeyConfig, Target
from llm_gateway.errors import GatewayError
from llm_gateway.limits import SpendLedger, TokenBucket
from llm_gateway.metrics import Metrics
from llm_gateway.providers import JSONObject, ProviderError, build_provider
from llm_gateway.resilience import CircuitState
from llm_gateway.router import AllProvidersFailed, Router, Sleep

logger = logging.getLogger(__name__)

# Upstream rejections that are about the request itself and so are worth passing on.
_CLIENT_ERRORS = frozenset({400, 404, 413, 422})


class Completion(NamedTuple):
    body: JSONObject
    target: Target
    cached: bool


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _sse(data: JSONObject) -> bytes:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def upstream_failure(exc: AllProvidersFailed) -> GatewayError:
    if all(e.reason == "circuit_open" for e in exc.errors):
        return GatewayError(
            503,
            f"All providers for model {exc.alias!r} are temporarily unavailable.",
            error_type="server_error",
            code="providers_unavailable",
        )
    last = exc.errors[-1]
    if last.status in _CLIENT_ERRORS:
        return GatewayError(last.status, str(last), code="upstream_rejected")
    detail = "; ".join(str(e) for e in exc.errors)
    return GatewayError(
        502,
        f"All providers for model {exc.alias!r} failed: {detail}",
        error_type="server_error",
        code="upstream_error",
    )


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.config = config
        self.metrics = Metrics()
        self._providers = {
            name: build_provider(name, provider, transport)
            for name, provider in config.providers.items()
        }
        self._router = Router(
            self._providers, config.models, config.resilience, self.metrics, sleep=sleep
        )
        self._cache: TTLCache[Completion] = TTLCache(
            config.cache.max_entries, config.cache.ttl_seconds
        )
        # Index by digest so the lookup's timing reveals nothing about the presented secret.
        self._keys = {_digest(k.key): k for k in config.keys}
        self._buckets = {k.name: TokenBucket(k.requests_per_minute) for k in config.keys}
        self._ledger = SpendLedger()

    async def aclose(self) -> None:
        await asyncio.gather(*(p.aclose() for p in self._providers.values()))

    def authenticate(self, authorization: str | None) -> KeyConfig:
        scheme, _, token = (authorization or "").partition(" ")
        key = self._keys.get(_digest(token.strip())) if scheme.lower() == "bearer" else None
        if key is None:
            raise GatewayError(
                401,
                "Missing or invalid API key. Send it as 'Authorization: Bearer <key>'.",
                code="invalid_api_key",
            )
        return key

    def admit(self, key: KeyConfig) -> None:
        """Enforce the key's rate limit and monthly budget before any upstream work."""
        wait = self._buckets[key.name].try_acquire()
        if wait:
            raise GatewayError(
                429,
                f"Rate limit of {key.requests_per_minute} requests per minute exceeded.",
                error_type="requests",
                code="rate_limit_exceeded",
                headers={"retry-after": str(math.ceil(wait))},
            )
        budget = key.monthly_budget_usd
        if budget is not None and self._ledger.spent(key.name) >= budget:
            raise GatewayError(
                429,
                f"Monthly budget of ${budget:.2f} for this key is exhausted.",
                error_type="insufficient_quota",
                code="insufficient_quota",
            )

    def check_model(self, alias: str) -> None:
        if alias not in self.config.models:
            raise GatewayError(
                404,
                f"The model {alias!r} does not exist.",
                param="model",
                code="model_not_found",
            )

    def _cacheable(self, payload: JSONObject) -> bool:
        cache = self.config.cache
        # OpenAI's default temperature is 1, so an unset temperature is non-deterministic.
        deterministic = payload.get("temperature", 1.0) == 0
        return cache.enabled and (deterministic or cache.cache_nonzero_temperature)

    async def complete(self, key: KeyConfig, payload: JSONObject) -> Completion:
        # Scoped per key so one tenant can never observe another tenant's prompts.
        lookup = cache_key(key.name, payload) if self._cacheable(payload) else None
        if lookup is not None:
            if (hit := self._cache.get(lookup)) is not None:
                self.metrics.cache_hits.inc()
                return hit._replace(cached=True)
            self.metrics.cache_misses.inc()

        try:
            target, body = await self._router.complete(payload["model"], payload)
        except AllProvidersFailed as exc:
            raise upstream_failure(exc) from exc
        self._account(key, target, body.get("usage"))
        completion = Completion(body, target, cached=False)
        if lookup is not None:
            self._cache.put(lookup, completion)
        return completion

    async def stream(
        self, key: KeyConfig, payload: JSONObject
    ) -> tuple[Target, AsyncIterator[bytes]]:
        include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))
        try:
            target, chunks = await self._router.open_stream(payload["model"], payload)
        except AllProvidersFailed as exc:
            raise upstream_failure(exc) from exc
        return target, self._sse_events(key, target, chunks, include_usage=include_usage)

    async def _sse_events(
        self,
        key: KeyConfig,
        target: Target,
        chunks: AsyncIterator[JSONObject],
        *,
        include_usage: bool,
    ) -> AsyncGenerator[bytes]:
        usage = None
        try:
            async for chunk in chunks:
                usage = chunk.get("usage") or usage
                if not include_usage:
                    if not chunk.get("choices"):
                        continue
                    chunk.pop("usage", None)
                yield _sse(chunk)
            yield b"data: [DONE]\n\n"
        except ProviderError as exc:
            # The 200 status and earlier chunks are already sent, so the failure can only
            # be reported in-band; no [DONE] marker tells the client the stream is incomplete.
            error = GatewayError(502, str(exc), error_type="server_error", code="stream_error")
            yield _sse(error.body())
        finally:
            self._account(key, target, usage)

    def _account(self, key: KeyConfig, target: Target, usage: JSONObject | None) -> None:
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        price = self.config.pricing.get(target.model)
        cost = price.cost(prompt, completion) if price else 0.0
        labels = {"key": key.name, "provider": target.provider, "model": target.model}
        self.metrics.tokens.inc(prompt, **labels, type="prompt")
        self.metrics.tokens.inc(completion, **labels, type="completion")
        self.metrics.cost.inc(cost, **labels)
        self._ledger.add(key.name, cost)
        logger.info(
            "usage",
            extra={
                **labels,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_usd": cost,
            },
        )

    def render_metrics(self) -> str:
        for name, breaker in self._router.breakers.items():
            self.metrics.circuit_state.set(
                0.0 if breaker.state is CircuitState.CLOSED else 1.0, provider=name
            )
        return self.metrics.render()

"""Resolves a model alias to its fallback chain and walks it with retries and circuit breakers."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing

import httpx

from llm_gateway.config import ResilienceConfig, Target
from llm_gateway.metrics import Metrics
from llm_gateway.providers import JSONObject, Provider, ProviderError
from llm_gateway.providers.base import network_error
from llm_gateway.resilience import CircuitBreaker, backoff_delay

logger = logging.getLogger(__name__)

type Sleep = Callable[[float], Awaitable[None]]


class AllProvidersFailed(Exception):
    def __init__(self, alias: str, errors: list[ProviderError]) -> None:
        super().__init__(f"all providers failed for model {alias!r}")
        self.alias = alias
        self.errors = errors


class Router:
    def __init__(
        self,
        providers: Mapping[str, Provider],
        aliases: Mapping[str, Sequence[Target]],
        config: ResilienceConfig,
        metrics: Metrics,
        *,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._providers = providers
        self._aliases = aliases
        self._config = config
        self._metrics = metrics
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.breakers = {
            name: CircuitBreaker(
                config.circuit_failure_threshold, config.circuit_reset_seconds, clock
            )
            for name in providers
        }

    async def complete(self, alias: str, payload: JSONObject) -> tuple[Target, JSONObject]:
        return await self._run(alias, payload, lambda provider, body: provider.complete(body))

    async def open_stream(
        self, alias: str, payload: JSONObject
    ) -> tuple[Target, AsyncIterator[JSONObject]]:
        """Start a stream, retrying and falling back only until the first chunk arrives.

        Once a chunk has been handed to the caller it may already be on the wire, so a
        later failure cannot be retried transparently and is raised as a ProviderError.
        """
        return await self._run(alias, payload, self._start_stream)

    async def _start_stream(
        self, provider: Provider, body: JSONObject
    ) -> AsyncIterator[JSONObject]:
        chunks = provider.stream(body)
        try:
            first = await anext(chunks)
        except StopAsyncIteration:
            raise ProviderError(provider.name, "stream ended before any data") from None
        return self._continue_stream(provider.name, first, chunks)

    async def _continue_stream(
        self, provider: str, first: JSONObject, chunks: AsyncGenerator[JSONObject]
    ) -> AsyncGenerator[JSONObject]:
        async with aclosing(chunks):
            yield first
            try:
                async for chunk in chunks:
                    yield chunk
            except httpx.TransportError as exc:
                error = network_error(provider, exc)
                self._record_error(error)
                raise error from exc
            except ProviderError as exc:
                self._record_error(exc)
                raise

    async def _run[T](
        self,
        alias: str,
        payload: JSONObject,
        call: Callable[[Provider, JSONObject], Awaitable[T]],
    ) -> tuple[Target, T]:
        errors: list[ProviderError] = []
        for target in self._aliases[alias]:
            provider = self._providers[target.provider]
            breaker = self.breakers[target.provider]
            body = {**payload, "model": target.model}
            for attempt in range(self._config.max_retries + 1):
                if not breaker.allow():
                    error = ProviderError(target.provider, "circuit open", reason="circuit_open")
                    self._record_error(error)
                    errors.append(error)
                    break
                try:
                    result = await call(provider, body)
                except httpx.TransportError as exc:
                    error = network_error(target.provider, exc)
                except ProviderError as exc:
                    error = exc
                else:
                    breaker.record_success()
                    return target, result

                self._record_error(error)
                errors.append(error)
                if not error.retryable:
                    # The upstream answered, so it is healthy; the request itself was refused.
                    breaker.record_success()
                    break
                breaker.record_failure()
                if attempt == self._config.max_retries:
                    break
                delay = backoff_delay(self._config, attempt, error.retry_after, self._rng)
                if delay is None:
                    break
                logger.info(
                    "retrying upstream",
                    extra={"target": str(target), "attempt": attempt + 1, "delay_s": delay},
                )
                await self._sleep(delay)
        raise AllProvidersFailed(alias, errors)

    def _record_error(self, error: ProviderError) -> None:
        self._metrics.provider_errors.inc(provider=error.provider, reason=error.reason)
        logger.warning("upstream error", extra={"error": str(error), "reason": error.reason})

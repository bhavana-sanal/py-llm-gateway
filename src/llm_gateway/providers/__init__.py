"""Provider adapters. Each speaks OpenAI chat-completions format to the rest of the gateway."""

from __future__ import annotations

import httpx

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.anthropic import AnthropicProvider
from llm_gateway.providers.base import JSONObject, Provider, ProviderError
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.openai import OpenAIProvider

__all__ = ["JSONObject", "Provider", "ProviderError", "build_provider"]


def build_provider(
    name: str, config: ProviderConfig, transport: httpx.AsyncBaseTransport | None = None
) -> Provider:
    match config.type:
        case "mock":
            return MockProvider(name)
        case "openai":
            return OpenAIProvider(
                name,
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
                transport=transport,
            )
        case "anthropic":
            return AnthropicProvider(
                name,
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
                default_max_tokens=config.default_max_tokens,
                transport=transport,
            )

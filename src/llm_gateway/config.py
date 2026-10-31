"""Gateway configuration: a TOML file validated into immutable pydantic models."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_ENV_REF = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerConfig(_Frozen):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"


class ProviderConfig(_Frozen):
    type: Literal["openai", "anthropic", "mock"]
    base_url: str | None = None
    api_key: str = ""
    timeout_seconds: float = Field(default=60.0, gt=0)
    # Anthropic requires max_tokens on every request; OpenAI-style clients often omit it.
    default_max_tokens: int = Field(default=4096, ge=1)


class ResilienceConfig(_Frozen):
    max_retries: int = Field(default=2, ge=0)
    backoff_base_seconds: float = Field(default=0.5, gt=0)
    backoff_max_seconds: float = Field(default=8.0, gt=0)
    max_retry_after_seconds: float = Field(default=30.0, ge=0)
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_reset_seconds: float = Field(default=30.0, gt=0)


class CacheConfig(_Frozen):
    enabled: bool = True
    max_entries: int = Field(default=1024, ge=1)
    ttl_seconds: float = Field(default=300.0, gt=0)
    cache_nonzero_temperature: bool = False


class KeyConfig(_Frozen):
    name: str = Field(min_length=1)
    key: str = Field(min_length=1, repr=False)
    requests_per_minute: int = Field(default=60, ge=1)
    monthly_budget_usd: float | None = Field(default=None, gt=0)


class Price(_Frozen):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1e6


class Target(_Frozen):
    """One step of a fallback chain: a configured provider plus the upstream model name."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def _parse_target(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    # Split on the first slash only: upstream model ids may contain slashes themselves.
    provider, sep, model = value.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"target {value!r} must look like 'provider/model'")
    return {"provider": provider, "model": model}


TargetSpec = Annotated[Target, BeforeValidator(_parse_target)]


class GatewayConfig(_Frozen):
    server: ServerConfig = ServerConfig()
    providers: dict[str, ProviderConfig] = Field(min_length=1)
    models: dict[str, Annotated[list[TargetSpec], Field(min_length=1)]] = Field(min_length=1)
    keys: list[KeyConfig] = Field(min_length=1)
    resilience: ResilienceConfig = ResilienceConfig()
    cache: CacheConfig = CacheConfig()
    pricing: dict[str, Price] = {}

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        for alias, targets in self.models.items():
            for target in targets:
                if target.provider not in self.providers:
                    raise ValueError(
                        f"model {alias!r} references unknown provider {target.provider!r}"
                    )
        names = [k.name for k in self.keys]
        if len(set(names)) != len(names):
            raise ValueError("gateway key names must be unique")
        if len({k.key for k in self.keys}) != len(self.keys):
            raise ValueError("gateway key values must be unique")
        return self


def expand_env(value: Any) -> Any:
    """Replace ``${VAR}`` and ``${VAR:-default}`` in every string; unset variables become ''."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m["name"], m["default"] or ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def load_config(path: Path) -> GatewayConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return GatewayConfig.model_validate(expand_env(raw))

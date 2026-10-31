from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_gateway.config import Price, Target, expand_env, load_config

from helpers import make_config

ROOT = Path(__file__).parent.parent


def test_example_config_loads() -> None:
    config = load_config(ROOT / "gateway.example.toml")
    assert config.models["smart"] == [
        Target(provider="anthropic", model="claude-sonnet-5-5"),
        Target(provider="openai", model="gpt-5-mini"),
    ]


def test_expand_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW_TOKEN", "secret")
    monkeypatch.delenv("GW_MISSING", raising=False)
    raw = {"a": "${GW_TOKEN}", "b": ["x-${GW_MISSING:-fallback}"], "c": "${GW_MISSING}", "d": 3}
    assert expand_env(raw) == {"a": "secret", "b": ["x-fallback"], "c": "", "d": 3}


def test_target_keeps_slashes_in_model_name() -> None:
    config = make_config(models={"llama": ["primary/meta-llama/llama-4-scout"]})
    assert config.models["llama"] == [Target(provider="primary", model="meta-llama/llama-4-scout")]


@pytest.mark.parametrize(
    "models",
    [
        {"x": ["no-slash"]},
        {"x": ["nowhere/gpt-5-mini"]},
        {"x": []},
    ],
)
def test_invalid_model_aliases_are_rejected(models: dict[str, list[str]]) -> None:
    with pytest.raises(ValidationError):
        make_config(models=models)


def test_duplicate_keys_are_rejected() -> None:
    keys = [{"name": "a", "key": "same"}, {"name": "b", "key": "same"}]
    with pytest.raises(ValidationError, match="unique"):
        make_config(keys=keys)


def test_price_cost() -> None:
    price = Price(input_per_mtok=3.0, output_per_mtok=15.0)
    assert price.cost(1_000_000, 100_000) == pytest.approx(4.5)

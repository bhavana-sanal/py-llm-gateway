"""A minimal Prometheus registry rendering the text exposition format (version 0.0.4)."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escape(value: str) -> str:
    return value.replace("\\", r"\\").replace('"', r"\"").replace("\n", r"\n")


def _format_labels(names: Sequence[str], values: Sequence[str]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{n}="{_escape(v)}"' for n, v in zip(names, values, strict=True))
    return "{" + pairs + "}"


def _format_value(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


class _Metric:
    kind = ""

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labels = tuple(labels)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if labels.keys() != set(self.labels):
            raise ValueError(f"{self.name} expects labels {self.labels}, got {tuple(labels)}")
        return tuple(labels[name] for name in self.labels)

    def render(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.documentation}"
        yield f"# TYPE {self.name} {self.kind}"
        yield from self._samples()

    def _samples(self) -> Iterator[str]:
        raise NotImplementedError


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, documentation: str, labels: Sequence[str] = ()) -> None:
        super().__init__(name, documentation, labels)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        key = self._key(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(self._key(labels), 0.0)

    def _samples(self) -> Iterator[str]:
        for key, value in sorted(self._values.items()):
            yield f"{self.name}{_format_labels(self.labels, key)} {_format_value(value)}"


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels: str) -> None:
        self._values[self._key(labels)] = value


@dataclass
class _Series:
    buckets: list[int]  # cumulative counts, one per bound; the last bound is +Inf
    total: float = 0.0


class Histogram(_Metric):
    kind = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    ) -> None:
        super().__init__(name, documentation, labels)
        self._bounds = (*sorted(buckets), math.inf)
        self._series: dict[tuple[str, ...], _Series] = {}

    def observe(self, value: float, **labels: str) -> None:
        key = self._key(labels)
        series = self._series.setdefault(key, _Series([0] * len(self._bounds)))
        for i, bound in enumerate(self._bounds):
            if value <= bound:
                series.buckets[i] += 1
        series.total += value

    def _samples(self) -> Iterator[str]:
        for key, series in sorted(self._series.items()):
            for bound, count in zip(self._bounds, series.buckets, strict=True):
                labels = _format_labels((*self.labels, "le"), (*key, _format_value(bound)))
                yield f"{self.name}_bucket{labels} {count}"
            labels = _format_labels(self.labels, key)
            yield f"{self.name}_sum{labels} {_format_value(series.total)}"
            yield f"{self.name}_count{labels} {series.buckets[-1]}"


class Metrics:
    def __init__(self) -> None:
        self.requests = Counter(
            "llm_gateway_requests_total",
            "Chat completion requests by key, model alias and HTTP status.",
            ("key", "model", "status"),
        )
        self.latency = Histogram(
            "llm_gateway_request_duration_seconds",
            "Time until response headers are sent; for streams, time to the first chunk.",
            ("model",),
        )
        self.tokens = Counter(
            "llm_gateway_tokens_total",
            "Tokens consumed by key, provider, upstream model and direction.",
            ("key", "provider", "model", "type"),
        )
        self.cost = Counter(
            "llm_gateway_cost_usd_total",
            "Estimated spend in USD from the configured pricing table.",
            ("key", "provider", "model"),
        )
        self.cache_hits = Counter("llm_gateway_cache_hits_total", "Response cache hits.")
        self.cache_misses = Counter("llm_gateway_cache_misses_total", "Response cache misses.")
        self.provider_errors = Counter(
            "llm_gateway_provider_errors_total",
            "Failed upstream attempts by provider and reason (HTTP status, timeout, ...).",
            ("provider", "reason"),
        )
        self.circuit_state = Gauge(
            "llm_gateway_circuit_breaker_open",
            "1 if the provider's circuit breaker is open or half-open, else 0.",
            ("provider",),
        )

    def render(self) -> str:
        metrics: list[_Metric] = [
            self.requests,
            self.latency,
            self.tokens,
            self.cost,
            self.cache_hits,
            self.cache_misses,
            self.provider_errors,
            self.circuit_state,
        ]
        return "\n".join(line for metric in metrics for line in metric.render()) + "\n"

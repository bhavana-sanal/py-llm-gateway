"""Exact-match response cache: an in-memory LRU with per-entry TTL."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

# Fields that never change what the model generates.
_IGNORED_FIELDS = frozenset({"stream", "stream_options", "user"})


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    normalized = {k: v for k, v in payload.items() if k not in _IGNORED_FIELDS}
    blob = json.dumps([namespace, normalized], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class TTLCache[V]:
    def __init__(
        self, max_entries: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, V]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return value

    def put(self, key: str, value: V) -> None:
        self._entries[key] = (self._clock() + self._ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

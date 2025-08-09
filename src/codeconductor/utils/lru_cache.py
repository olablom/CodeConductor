from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import os
import time as _time


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class LRUCacheTTL:
    def __init__(
        self,
        max_entries: int = 100,
        ttl_seconds: int = 1800,
        time_provider: Callable[[], float] | None = None,
        namespace: str = "default",
    ) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._now = time_provider or _time.time
        self.stats = CacheStats()
        self.namespace = namespace
        self.bypass = os.getenv("CACHE_BYPASS") == "1"

    def _expired(self, ts: float) -> bool:
        return (self._now() - ts) > self.ttl_seconds

    def _purge_expired(self) -> None:
        to_delete = []
        for k, (ts, _) in list(self._store.items()):
            if self._expired(ts):
                to_delete.append(k)
        for k in to_delete:
            self._store.pop(k, None)
            self.stats.evictions += 1

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._store.clear()

    def make_key(
        self,
        *,
        prompt: str,
        persona: str,
        policy: str,
        model: str,
        params: Dict[str, Any] | None = None,
    ) -> str:
        p = params or {}
        parts = [
            f"ns={self.namespace}",
            f"persona={persona}",
            f"policy={policy}",
            f"model={model}",
            f"temp={p.get('temperature', '')}",
            f"top_p={p.get('top_p', '')}",
            f"prompt={prompt.strip()}",
        ]
        return "|".join(parts)

    def get(self, key: str) -> Optional[Any]:
        if self.bypass:
            self.stats.misses += 1
            return None
        self._purge_expired()
        item = self._store.get(key)
        if item is None:
            self.stats.misses += 1
            return None
        ts, value = item
        if self._expired(ts):
            self._store.pop(key, None)
            self.stats.evictions += 1
            self.stats.misses += 1
            return None
        # refresh LRU
        self._store.move_to_end(key, last=True)
        self.stats.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if self.bypass:
            return
        self._purge_expired()
        self._store[key] = (self._now(), value)
        self._store.move_to_end(key, last=True)
        self._evict_if_needed()

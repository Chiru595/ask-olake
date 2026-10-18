"""Answer cache with targeted invalidation via a chunk -> cache-key reverse index."""
import hashlib
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from cachetools import TTLCache


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def cache_key(q: str) -> str:
    return hashlib.sha256(normalize_question(q).lower().encode()).hexdigest()


@dataclass
class CachedAnswer:
    answer: str
    sources: list[str]
    titles: list[str]
    chunk_ids: list[str]
    embedding: list[float] | None
    is_refusal: bool
    created_at: float = field(default_factory=time.time)


class _TTL(TTLCache):
    def __init__(self, owner: "AnswerCache", **kw):
        super().__init__(**kw)
        self.owner = owner

    def popitem(self):  # called on LRU/TTL eviction; keep the reverse index in sync
        key, value = super().popitem()
        self.owner._unindex(key, value)
        return key, value

    def expire(self, time=None):
        expired = super().expire(time)
        for key, value in expired or []:
            self.owner._unindex(key, value)
        return expired


class AnswerCache:
    def __init__(self, maxsize: int, ttl: int, new_chunk_sim: float):
        self.lock = threading.RLock()
        self.answers = _TTL(self, maxsize=maxsize, ttl=ttl)
        self.chunk_to_keys: dict[str, set[str]] = {}
        self.sim = new_chunk_sim
        self.hits = 0
        self.misses = 0

    def _unindex(self, key: str, value: CachedAnswer) -> None:
        for cid in value.chunk_ids:
            keys = self.chunk_to_keys.get(cid)
            if keys:
                keys.discard(key)
                if not keys:
                    self.chunk_to_keys.pop(cid, None)

    def get(self, key: str) -> CachedAnswer | None:
        with self.lock:
            v = self.answers.get(key)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, key: str, value: CachedAnswer) -> None:
        with self.lock:
            old = self.answers.get(key)
            if old is not None:
                self._unindex(key, old)
            self.answers[key] = value
            for cid in value.chunk_ids:
                self.chunk_to_keys.setdefault(cid, set()).add(key)

    def _evict(self, keys: set[str]) -> int:
        n = 0
        for k in keys:
            v = self.answers.pop(k, None)
            if v is not None:
                self._unindex(k, v)
                n += 1
        return n

    def invalidate(self, changed: list, deleted_ids: list[str], new_dense: dict[str, list[float]]) -> int:
        """Evict answers that used changed/deleted chunks, refusals, and answers similar to new chunks."""
        with self.lock:
            victims: set[str] = set()
            for cid in [c.chunk_id for c in changed] + list(deleted_ids):
                victims |= self.chunk_to_keys.get(cid, set())
            if changed:
                victims |= {k for k, v in self.answers.items() if v.is_refusal}
                vecs = [new_dense[c.chunk_id] for c in changed if c.chunk_id in new_dense]
                items = [(k, v.embedding) for k, v in self.answers.items() if v.embedding and k not in victims]
                if vecs and items:
                    q = np.asarray([e for _, e in items], dtype=np.float32)
                    m = np.asarray(vecs, dtype=np.float32)
                    sims = (q @ m.T).max(axis=1)
                    victims |= {items[i][0] for i in np.nonzero(sims >= self.sim)[0]}
            return self._evict(victims)

    def clear(self) -> int:
        with self.lock:
            n = len(self.answers)
            self.answers.clear()
            self.chunk_to_keys.clear()
            return n

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self.answers), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}

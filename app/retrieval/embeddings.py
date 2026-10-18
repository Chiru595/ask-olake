"""Dense embeddings (Cohere on Azure, native v2 API) and local BM25 sparse embeddings."""
import asyncio
import os
import random
import shutil
import tempfile
import time
from collections import deque

import httpx
import numpy as np
import structlog
from fastembed import SparseTextEmbedding
from qdrant_client import models

from app.config import MODELS_DIR, Settings
from app.ingest.chunker import count_tokens

log = structlog.get_logger()


async def with_retries(fn, *, attempts: int = 3, base: float = 0.5, what: str = "call"):
    for i in range(attempts):
        try:
            return await fn()
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if i == attempts - 1 or not retryable:
                log.warning("external_call_failed", what=what, error=str(e)[:200], status=status)
                raise
            wait = base * 2**i + random.random() * 0.25
            if status == 429 and resp is not None:
                hdr = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens")
                wait = max(wait, float(hdr) if hdr and hdr.replace(".", "", 1).isdigit() else 5.0)
            log.info("retrying", what=what, status=status, wait=round(wait, 1))
            await asyncio.sleep(wait)


class TokenBudget:
    """Sliding 60 s window so bulk indexing stays under the provider's tokens-per-minute limit."""

    def __init__(self, per_minute: int):
        self.limit = per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.events and now - self.events[0][0] > 60:
                    self.events.popleft()
                used = sum(t for _, t in self.events)
                if used + tokens <= self.limit or not self.events:
                    self.events.append((now, tokens))
                    return
                await asyncio.sleep(max(0.5, 60 - (now - self.events[0][0])))


class Embedder:
    def __init__(self, client: httpx.AsyncClient, s: Settings):
        self.c, self.s = client, s
        self.budget = TokenBudget(s.embed_tokens_per_minute)
        self.bulk = asyncio.Semaphore(2)

    async def _embed_batch(self, texts: list[str], input_type: str, attempts: int = 6) -> list[list[float]]:
        async def call():
            r = await self.c.post(
                self.s.embed_endpoint,
                headers={"Authorization": f"Bearer {self.s.embed_api_key}"},
                json={"model": self.s.embed_model, "texts": texts, "input_type": input_type,
                      "embedding_types": ["float"], "output_dimension": self.s.embed_dim, "truncate": "END"},
                timeout=self.s.embed_timeout_seconds,
            )
            r.raise_for_status()
            return r.json()["embeddings"]["float"]

        vecs = np.asarray(await with_retries(call, attempts=attempts, what="embed"), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batches by count and tokens; bulk calls share a TPM budget so queries keep headroom."""
        out: list[list[float]] = []
        batch: list[str] = []
        tok = 0
        for t in texts + [None]:
            n = int(count_tokens(t) * 1.15) + 8 if t is not None else 0
            if batch and (t is None or len(batch) >= self.s.embed_batch_size or tok + n > 20_000):
                async with self.bulk:
                    await self.budget.acquire(tok)
                    out.extend(await self._embed_batch(batch, "search_document"))
                batch, tok = [], 0
            if t is not None:
                batch.append(t)
                tok += n
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text], "search_query", attempts=2))[0]


def _bm25_cache_dir() -> str | None:
    """Prefer the BM25 files bundled in the repo; copy them to a writable dir since fastembed may write locks."""
    env = os.getenv("FASTEMBED_CACHE_PATH")
    bundled = MODELS_DIR / "fastembed"
    if not bundled.is_dir():
        return env
    target = os.path.join(tempfile.gettempdir(), "ask-olake-fastembed")
    if not os.path.isdir(target):
        shutil.copytree(bundled, target, dirs_exist_ok=True)
    return target


class SparseEmbedder:
    def __init__(self) -> None:
        self._model: SparseTextEmbedding | None = None

    @property
    def model(self) -> SparseTextEmbedding:
        if self._model is None:  # lazy: keeps cold start and import time small
            self._model = SparseTextEmbedding("Qdrant/bm25", cache_dir=_bm25_cache_dir())
        return self._model

    def _to_sv(self, e) -> models.SparseVector:
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    async def embed_documents(self, texts: list[str]) -> list[models.SparseVector]:
        res = await asyncio.to_thread(lambda: list(self.model.embed(texts, batch_size=64)))
        return [self._to_sv(e) for e in res]

    async def embed_query(self, text: str) -> models.SparseVector:
        res = await asyncio.to_thread(lambda: list(self.model.query_embed(text)))
        return self._to_sv(res[0])

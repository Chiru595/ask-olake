"""Cohere rerank on Azure: Pro, then Fast; returns None so callers fall back to RRF order."""
import asyncio

import httpx
import structlog

from app.config import Settings

log = structlog.get_logger()


class Reranker:
    def __init__(self, client: httpx.AsyncClient, s: Settings):
        self.c, self.s = client, s
        self.targets = [(s.rerank_endpoint, s.rerank_api_key, s.rerank_model),
                        (s.rerank_fallback_endpoint, s.rerank_fallback_api_key, s.rerank_fallback_model)]
        self.targets = [t for t in self.targets if t[0] and t[1]]

    async def rerank(self, query: str, docs: list[str], top_n: int) -> tuple[list[tuple[int, float]], str] | None:
        """Returns ([(doc_index, score)], model) sorted best first, or None if every reranker failed."""
        if self.s.rerank_provider == "none" or not docs:
            return None
        for endpoint, key, model in self.targets:
            try:
                r = await self.c.post(endpoint, headers={"Authorization": f"Bearer {key}"},
                                      json={"model": model, "query": query, "documents": docs, "top_n": top_n},
                                      timeout=self.s.rerank_timeout_seconds)
                r.raise_for_status()
                return [(x["index"], float(x["relevance_score"])) for x in r.json()["results"]], model
            except (httpx.HTTPError, asyncio.TimeoutError, KeyError) as e:
                log.warning("rerank_failed", model=model, error=str(e)[:200])
        return None

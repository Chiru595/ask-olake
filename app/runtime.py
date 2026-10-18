"""Lazy, idempotent wiring of clients and services, shared by the lifespan (local) and first request (serverless)."""
import asyncio

import httpx
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app.cache import AnswerCache
from app.config import get_settings
from app.ingest.indexer import Indexer
from app.llm.client import LLMClient
from app.retrieval.embeddings import Embedder, SparseEmbedder
from app.retrieval.pipeline import Pipeline
from app.retrieval.rerank import Reranker
from app.sync.poller import Poller
from app.sync.state import StateStore

_lock = asyncio.Lock()


async def get_state(app: FastAPI, sync: bool = False):
    """Build every client once per process. In serverless mode also load sync state (no poller loop);
    `sync=True` lets a request kick off a best-effort incremental sync when the last one is stale."""
    st = app.state
    if getattr(st, "initialised", False):
        if sync and get_settings().is_serverless and st.poller.ready:
            st.poller.maybe_sync()
        return st
    async with _lock:
        if getattr(st, "initialised", False):
            return st
        s = get_settings()
        st.http = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
        st.qdrant = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None, timeout=int(s.qdrant_timeout_seconds))
        dense, sparse = Embedder(st.http, s), SparseEmbedder()
        st.cache = AnswerCache(s.cache_max_size, s.cache_ttl_seconds, s.cache_new_chunk_sim)
        st.indexer = Indexer(st.qdrant, s, dense, sparse)
        st.llm = LLMClient(s, st.http)
        st.poller = Poller(s, st.http, st.indexer, StateStore(st.qdrant, s), st.cache)
        st.pipeline = Pipeline(s, st.qdrant, dense, sparse, Reranker(st.http, s), st.llm, st.cache)
        if s.is_serverless:
            await st.poller.start_serverless()
        st.initialised = True
    return st


async def close_state(app: FastAPI) -> None:
    st = app.state
    if not getattr(st, "initialised", False):
        return
    await st.poller.stop()
    await st.qdrant.close()
    await st.http.aclose()
    st.initialised = False

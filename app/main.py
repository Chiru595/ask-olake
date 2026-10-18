"""FastAPI app: lifespan wiring (Qdrant, providers, poller, MCP), routes, error contract."""
import asyncio
import contextlib
from pathlib import Path

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from qdrant_client import AsyncQdrantClient
from slowapi.errors import RateLimitExceeded

from app.api import admin, ask, status
from app.cache import AnswerCache
from app.config import get_settings
from app.ingest.indexer import Indexer
from app.llm.client import LLMClient
from app.mcp_server import build_mcp, mcp_asgi
from app.observability import RequestContextMiddleware, setup_logging
from app.retrieval.embeddings import Embedder, SparseEmbedder
from app.retrieval.pipeline import Pipeline
from app.retrieval.rerank import Reranker
from app.security import limiter
from app.sync.poller import Poller
from app.sync.state import StateStore

S = get_settings()
setup_logging(S.log_level)
log = structlog.get_logger()
WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
ASK_PATHS = ("/ask", "/ask/stream", "/ask/debug")

mcp = build_mcp(lambda: app.state.pipeline) if S.mcp_enabled else None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    http = httpx.AsyncClient(http2=False, limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
    q = AsyncQdrantClient(url=S.qdrant_url, api_key=S.qdrant_api_key or None, timeout=int(S.qdrant_timeout_seconds))
    dense, sparse = Embedder(http, S), SparseEmbedder()
    cache = AnswerCache(S.cache_max_size, S.cache_ttl_seconds, S.cache_new_chunk_sim)
    indexer = Indexer(q, S, dense, sparse)
    llm = LLMClient(S, http)
    poller = Poller(S, http, indexer, StateStore(q, S), cache)
    app.state.indexer, app.state.cache, app.state.llm, app.state.poller = indexer, cache, llm, poller
    app.state.pipeline = Pipeline(S, q, dense, sparse, Reranker(http, S), llm, cache)
    # start serving immediately; /health reports 503 until the index is ready
    starter = asyncio.create_task(poller.start())
    starter.add_done_callback(lambda t: t.exception() and log.error("poller_start_failed", error=str(t.exception())))
    async with contextlib.AsyncExitStack() as stack:
        if mcp is not None:
            await stack.enter_async_context(mcp.session_manager.run())
        log.info("startup", qdrant=S.qdrant_url.split("//")[-1][:40], llm=S.llm_model, embed=S.embed_model)
        yield
    starter.cancel()
    await poller.stop()
    await q.close()
    await http.aclose()


app = FastAPI(title="Ask OLake", version=S.app_version, lifespan=lifespan,
              description="RAG assistant for OLake docs. `POST /ask` with `{\"question\": \"...\"}`.")
app.state.limiter = limiter
app.add_middleware(RequestContextMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=S.origins, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], allow_credentials=False,
                   expose_headers=["X-Request-ID", "X-Response-Time-ms", "X-Cache", "X-Model", "Retry-After"])


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    retry = "60"
    resp = JSONResponse({"answer": "Too many requests. Please wait a moment and try again.", "sources": []},
                        status_code=429, headers={"Retry-After": retry})
    return request.app.state.limiter._inject_headers(resp, request.state.view_rate_limit) if hasattr(
        request.state, "view_rate_limit") else resp


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    if request.url.path in ASK_PATHS:
        return JSONResponse({"answer": 'Invalid request. Send JSON like {"question": "..."}.', "sources": []},
                            status_code=422)
    return JSONResponse({"detail": exc.errors()}, status_code=422)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled", error=str(exc))
    if request.url.path in ASK_PATHS:
        return JSONResponse({"answer": ask.SERVICE_BUSY, "sources": []}, status_code=503)
    return JSONResponse({"detail": "internal error"}, status_code=500)


app.include_router(ask.router)
app.include_router(status.router)
app.include_router(admin.router)
if mcp is not None:
    app.mount("/mcp", mcp_asgi(mcp))


@app.get("/widget.js", include_in_schema=False)
async def widget():
    return FileResponse(WIDGET_DIR / "ask-olake.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})


@app.get("/", include_in_schema=False)
async def demo():
    return FileResponse(WIDGET_DIR / "demo.html", media_type="text/html")

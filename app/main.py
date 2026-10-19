"""FastAPI app: lifespan wiring (local poller, MCP), routes, error contract. Also runs serverless on Vercel."""
import asyncio
import contextlib
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from slowapi.errors import RateLimitExceeded

from app.api import admin, ask, meta, status
from app.config import get_settings
from app.observability import RequestContextMiddleware, setup_logging
from app.runtime import close_state, get_state
from app.security import limiter

S = get_settings()
setup_logging(S.log_level)
log = structlog.get_logger()
WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
ASK_PATHS = ("/ask", "/ask/stream", "/ask/debug")

# MCP needs a long-lived session manager started in the lifespan, so it's local/container-only
mcp = None
if S.mcp_enabled and not S.is_serverless:
    from app.mcp_server import build_mcp, mcp_asgi

    mcp = build_mcp(lambda: app.state.pipeline)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if S.is_serverless:
        # clients are built lazily on the first request; nothing long-running here
        yield
        return
    st = await get_state(app)
    # start serving immediately; /health reports 503 until the index is ready
    starter = asyncio.create_task(st.poller.start())
    starter.add_done_callback(lambda t: t.exception() and log.error("poller_start_failed", error=str(t.exception())))
    async with contextlib.AsyncExitStack() as stack:
        if mcp is not None:
            await stack.enter_async_context(mcp.session_manager.run())
        log.info("startup", qdrant=S.qdrant_url.split("//")[-1][:40], llm=S.llm_model, embed=S.embed_model)
        yield
    starter.cancel()
    await close_state(app)


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
app.include_router(meta.router)
if mcp is not None:
    app.mount("/mcp", mcp_asgi(mcp))


FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" '
           'fill="#2563eb"/><text x="16" y="22" font-family="Arial,sans-serif" font-size="18" font-weight="700" '
           'fill="#fff" text-anchor="middle">O</text></svg>')


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(FAVICON, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/widget.js", include_in_schema=False)
async def widget():
    return FileResponse(WIDGET_DIR / "ask-olake.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})


@app.get("/", include_in_schema=False)
@app.get("/demo", include_in_schema=False)
async def demo():
    return FileResponse(WIDGET_DIR / "demo.html", media_type="text/html")

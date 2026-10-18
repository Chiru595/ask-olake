"""/ask (exact contract), /ask/stream (SSE), /ask/debug (admin)."""
import asyncio
import json

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.llm.client import LLMUnavailable
from app.retrieval.pipeline import AskResult, InvalidQuestion
from app.security import limiter, require_admin

router = APIRouter()
log = structlog.get_logger()
S = get_settings()

SERVICE_BUSY = "The assistant is temporarily unavailable. Please try again shortly."
TIMEOUT_MSG = "The request took too long. Please try again."
NOT_READY = "The index is still warming up. Please try again in a minute."


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    question: str


def contract(answer: str, sources: list[str], status: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"answer": answer, "sources": sources}, status_code=status, headers=headers)


async def _run(request: Request, question: str) -> AskResult | JSONResponse:
    state = request.app.state
    if not state.poller.ready:
        return contract(NOT_READY, [], 503, {"Retry-After": "30"})
    try:
        return await asyncio.wait_for(state.pipeline.ask(question), S.request_timeout_seconds)
    except InvalidQuestion as e:
        return contract(str(e), [], 422)
    except asyncio.TimeoutError:
        log.warning("ask_timeout")
        return contract(TIMEOUT_MSG, [], 504)
    except LLMUnavailable:
        return contract(SERVICE_BUSY, [], 503, {"Retry-After": "10"})
    except Exception as e:  # noqa: BLE001 - never break the contract
        log.exception("ask_failed", error=str(e))
        return contract(SERVICE_BUSY, [], 503)


def _headers(res: AskResult) -> dict:
    h = {"X-Cache": "HIT" if res.cache_hit else "MISS", "X-Refusal": str(res.refusal).lower(),
         "X-Model": res.model or "unknown"}
    if res.top_score is not None:
        h["X-Top-Score"] = f"{res.top_score:.4f}"
    return h


@router.post("/ask")
@limiter.limit(S.rate_limit)
async def ask(request: Request, body: AskRequest):
    res = await _run(request, body.question)
    if isinstance(res, JSONResponse):
        return res
    log.info("ask", cache=res.cache_hit, refusal=res.refusal, top_score=res.top_score,
             timings=res.debug.get("timings"), n_sources=len(res.sources))
    return contract(res.answer, res.sources, headers=_headers(res))


@router.post("/ask/debug", dependencies=[Depends(require_admin)])
async def ask_debug(request: Request, body: AskRequest):
    res = await _run(request, body.question)
    if isinstance(res, JSONResponse):
        return res
    return {"answer": res.answer, "sources": res.sources, "refusal": res.refusal, "cache_hit": res.cache_hit,
            "model": res.model,
            "debug": res.debug}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream")
@limiter.limit(S.rate_limit)
async def ask_stream(request: Request, body: AskRequest):
    state = request.app.state

    async def gen():
        if not state.poller.ready:
            yield _sse("error", {"message": NOT_READY, "status": 503, "retry_after": 30})
            yield _sse("done", {})
            return
        try:
            async with asyncio.timeout(S.request_timeout_seconds):
                async for event, data in state.pipeline.ask_stream(body.question):
                    yield _sse(event, data)
        except InvalidQuestion as e:
            yield _sse("error", {"message": str(e), "status": 422, "retry_after": None})
        except TimeoutError:
            yield _sse("error", {"message": TIMEOUT_MSG, "status": 504, "retry_after": None})
        except LLMUnavailable:
            yield _sse("error", {"message": SERVICE_BUSY, "status": 503, "retry_after": 10})
        except Exception as e:  # noqa: BLE001
            log.exception("ask_stream_failed", error=str(e))
            yield _sse("error", {"message": SERVICE_BUSY, "status": 503, "retry_after": None})
        yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

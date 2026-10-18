"""/status (freshness proof) and /health (liveness + readiness)."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.runtime import get_state

router = APIRouter()
S = get_settings()


@router.get("/status")
async def status(request: Request):
    st = await get_state(request.app, sync=True)
    info = st.poller.info
    try:
        total = await st.indexer.count()
    except Exception:  # noqa: BLE001
        total = None
    return {
        "status": "ok" if st.poller.ready else "warming_up",
        "last_poll": info["last_poll"],
        "last_change_detected": info["last_change_detected"],
        "last_cycle_seconds": info["last_cycle_seconds"],
        "poll_interval_seconds": S.poll_interval_seconds,
        "sync_mode": "lazy (on request) + external cron /reindex" if S.is_serverless else "background poller",
        "total_chunks": total,
        "sources": info["sources"],
        "last_error": info["last_error"],
        "embed_model": S.embed_model,
        "embed_dim": S.embed_dim,
        "rerank_model": S.rerank_model,
        "llm_model": S.llm_model,
        "llm": {"calls": st.llm.calls, "spent_usd_this_process": round(st.llm.cost_usd, 5),
                "credit_remaining_usd": st.llm.credit_remaining},
        "cache": {**st.cache.stats(), "evicted_last_sync": info["evicted_last_sync"]},
        "version": S.app_version,
    }


@router.get("/health")
async def health(request: Request):
    st = await get_state(request.app)
    try:
        n = await st.indexer.count()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "unhealthy", "error": str(e)[:200]}, status_code=503)
    if not st.poller.ready or n == 0:
        return JSONResponse({"status": "warming_up", "chunks": n}, status_code=503)
    return {"status": "ok", "chunks": n}

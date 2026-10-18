"""/reindex (admin): incremental or full re-sync. Serverless runs incremental inline so a cron can drive freshness."""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.runtime import get_state
from app.security import require_admin

router = APIRouter()


class ReindexRequest(BaseModel):
    scope: Literal["incremental", "full"] = "incremental"


@router.post("/reindex", dependencies=[Depends(require_admin)])
async def reindex(request: Request, body: ReindexRequest):
    poller = (await get_state(request.app)).poller
    if get_settings().is_serverless:
        if body.scope == "full":
            return JSONResponse({"status": "error", "message": "full re-index is not run serverless; "
                                 "run scripts/full_index.py locally"}, status_code=400)
        # the request keeps the function alive, so the cycle actually completes (background work may be frozen)
        return {"status": "done", "scope": body.scope, **await poller.run_inline(timeout=120)}
    if poller.lock.locked():
        return {"status": "busy", "message": "a sync cycle is already running"}
    asyncio.create_task(poller.run_cycle(full=body.scope == "full"))
    return {"status": "started", "scope": body.scope}

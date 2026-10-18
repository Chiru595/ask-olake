"""/reindex (admin): incremental or full re-sync, run in the background."""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.security import require_admin

router = APIRouter()


class ReindexRequest(BaseModel):
    scope: Literal["incremental", "full"] = "incremental"


@router.post("/reindex", dependencies=[Depends(require_admin)])
async def reindex(request: Request, body: ReindexRequest):
    poller = request.app.state.poller
    if poller.lock.locked():
        return {"status": "busy", "message": "a sync cycle is already running"}
    asyncio.create_task(poller.run_cycle(full=body.scope == "full"))
    return {"status": "started", "scope": body.scope}

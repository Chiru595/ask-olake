"""/version (which commit is live) and /models (OpenRouter models this key can use), per the event rules."""
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.runtime import get_state

router = APIRouter()


@router.get("/version")
async def version():
    # read at call time: Vercel sets these per deployment
    return {"commit": os.getenv("VERCEL_GIT_COMMIT_SHA", ""), "deployment": os.getenv("VERCEL_DEPLOYMENT_ID", "")}


@router.get("/models")
async def models(request: Request):
    s = get_settings()
    st = await get_state(request.app)
    try:
        r = await st.http.get(f"{s.openrouter_base_url}/models/user", timeout=10,
                              headers={"Authorization": f"Bearer {s.openrouter_api_key}"})
        r.raise_for_status()
        return {"models": [m["id"] for m in r.json()["data"]]}
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return JSONResponse({"models": [], "error": str(e)[:200]}, status_code=502)

"""One-off full (re)index into Qdrant from a local machine, e.g. before deploying the serverless app.

Usage: python -m scripts.full_index [--recreate]
"""
import argparse
import asyncio

from fastapi import FastAPI

from app.config import get_settings
from app.observability import setup_logging
from app.runtime import close_state, get_state
from app.sync.poller import SCHEMA_VERSION, now


async def main(recreate: bool) -> None:
    s = get_settings()
    setup_logging(s.log_level)
    app = FastAPI()
    st = await get_state(app)
    await st.indexer.ensure_collections(recreate=recreate)
    print(await st.poller.run_cycle(full=True))
    await st.poller.state.put("meta", {"embed_model": s.embed_model, "embed_dim": s.embed_dim,
                                       "schema_version": SCHEMA_VERSION, "last_full_reindex": now()})
    print("chunks:", await st.indexer.count())
    await close_state(app)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    asyncio.run(main(p.parse_args().recreate))

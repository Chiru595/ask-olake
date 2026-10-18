"""Persist poller state in a tiny Qdrant collection so restarts resume where they left off."""
import uuid

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings

NS = uuid.UUID("0b8a3c62-9f8e-4e1e-8b8f-2d9b4f6a7c11")


class StateStore:
    def __init__(self, q: AsyncQdrantClient, s: Settings):
        self.q, self.col = q, s.qdrant_state_collection

    @staticmethod
    def _id(key: str) -> str:
        return str(uuid.uuid5(NS, key))

    async def get(self, key: str) -> dict:
        pts = await self.q.retrieve(self.col, ids=[self._id(key)], with_payload=True)
        return dict(pts[0].payload) if pts else {}

    async def put(self, key: str, value: dict) -> None:
        await self.q.upsert(self.col, wait=True,
                            points=[models.PointStruct(id=self._id(key), vector=[1.0], payload={"key": key, **value})])

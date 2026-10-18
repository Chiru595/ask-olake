"""Qdrant collection management, embed + upsert of changed chunks, deletes."""
import structlog
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.ingest.chunker import Chunk
from app.retrieval.embeddings import Embedder, SparseEmbedder

log = structlog.get_logger()
KEYWORD_FIELDS = ["doc_url", "source_type", "repo", "path", "component", "connector"]


class Indexer:
    def __init__(self, q: AsyncQdrantClient, s: Settings, dense: Embedder, sparse: SparseEmbedder):
        self.q, self.s, self.dense, self.sparse = q, s, dense, sparse
        self.col = s.qdrant_collection

    async def ensure_collections(self, recreate: bool = False) -> None:
        exists = await self.q.collection_exists(self.col)
        if exists and recreate:
            await self.q.delete_collection(self.col)
            exists = False
        if not exists:
            await self.q.create_collection(
                self.col,
                vectors_config={"dense": models.VectorParams(size=self.s.embed_dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            for f in KEYWORD_FIELDS:
                await self.q.create_payload_index(self.col, f, models.PayloadSchemaType.KEYWORD)
        if not await self.q.collection_exists(self.s.qdrant_state_collection):
            await self.q.create_collection(self.s.qdrant_state_collection,
                                           vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT))

    async def count(self, flt: models.Filter | None = None) -> int:
        return (await self.q.count(self.col, count_filter=flt, exact=True)).count

    async def existing(self, flt: models.Filter) -> dict[str, str]:
        """chunk_id -> content_hash for all points matching a filter."""
        out: dict[str, str] = {}
        offset = None
        while True:
            pts, offset = await self.q.scroll(self.col, scroll_filter=flt, limit=512, offset=offset,
                                              with_payload=["content_hash"], with_vectors=False)
            for p in pts:
                out[str(p.id)] = p.payload.get("content_hash", "")
            if offset is None:
                return out

    async def sync_chunks(self, flt: models.Filter, chunks: list[Chunk]) -> tuple[list[Chunk], list[str], dict[str, list[float]]]:
        """Make the points under `flt` equal `chunks`. Returns (upserted, deleted_ids, new_dense_by_id)."""
        old = await self.existing(flt)
        new_ids = {c.chunk_id for c in chunks}
        changed = [c for c in chunks if old.get(c.chunk_id) != c.content_hash]
        deleted = [i for i in old if i not in new_ids]
        dense_by_id = await self.upsert(changed) if changed else {}
        if deleted:
            await self.q.delete(self.col, points_selector=models.PointIdsList(points=deleted), wait=True)
        return changed, deleted, dense_by_id

    async def upsert(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        texts = [c.embed_text for c in chunks]
        dense = await self.dense.embed_documents(texts)
        sparse = await self.sparse.embed_documents(texts)
        points = [models.PointStruct(id=c.chunk_id, vector={"dense": d, "bm25": sp}, payload=c.payload())
                  for c, d, sp in zip(chunks, dense, sparse)]
        for i in range(0, len(points), 128):
            await self.q.upsert(self.col, points=points[i:i + 128], wait=True)
        return {c.chunk_id: d for c, d in zip(chunks, dense)}

    async def delete_where(self, flt: models.Filter) -> list[str]:
        ids = list((await self.existing(flt)).keys())
        if ids:
            await self.q.delete(self.col, points_selector=models.PointIdsList(points=ids), wait=True)
        return ids


def f_doc(doc_url: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_url", match=models.MatchValue(value=doc_url))])


def f_repo_path(repo: str, path: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="repo", match=models.MatchValue(value=repo)),
                               models.FieldCondition(key="path", match=models.MatchValue(value=path))])


def f_eq(key: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])

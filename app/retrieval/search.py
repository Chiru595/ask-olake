"""Hybrid dense + BM25 search in one Qdrant Query API call, fused with RRF, with a soft filter boost."""
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.ingest.classify import detect_query_filters


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


async def hybrid_search(q: AsyncQdrantClient, s: Settings, question: str, dense: list[float],
                        sparse: models.SparseVector) -> list[Hit]:
    k = s.top_k_prefetch
    prefetch = [
        models.Prefetch(query=dense, using="dense", limit=k),
        models.Prefetch(query=sparse, using="bm25", limit=k),
    ]
    component, connector = detect_query_filters(question)
    conds = []
    if connector:
        conds.append(models.FieldCondition(key="connector", match=models.MatchValue(value=connector)))
    if component:
        conds.append(models.FieldCondition(key="component", match=models.MatchValue(value=component)))
    if conds:
        # boost, never hard-filter: an extra filtered list joins the fusion
        prefetch.append(models.Prefetch(query=dense, using="dense", limit=10, filter=models.Filter(should=conds)))
    res = await q.query_points(
        s.qdrant_collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=s.top_k_fused,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=128),
        timeout=int(s.qdrant_timeout_seconds),
    )
    return [Hit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in res.points]

import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from ..config import (
    QDRANT_URL,
    COLLECTION_NAME,
    DENSE_MODEL,
    SPARSE_MODEL,
    PREFETCH_LIMIT,
)


class HybridRetriever:
    def __init__(self) -> None:
        self.client = QdrantClient(QDRANT_URL)
        self.dense_model = SentenceTransformer(DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def _build_filter(self, filters: dict[str, str]) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    def retrieve(
        self,
        query: str,
        filters: dict[str, str] = {},
        top_k: int = 20,
    ) -> tuple[list, float]:
        t0 = time.perf_counter()

        dense_vec = self.dense_model.encode(query).tolist()

        sparse_result = list(self.sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        qdrant_filter = self._build_filter(filters)

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=PREFETCH_LIMIT),
                Prefetch(query=sparse_vec, using="sparse", limit=PREFETCH_LIMIT),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return results.points, elapsed_ms

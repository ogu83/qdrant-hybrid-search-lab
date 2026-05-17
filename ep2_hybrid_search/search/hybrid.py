"""
Hybrid search: dense + sparse (BM25) via Qdrant prefetch API with RRF fusion.
Single network request. Both searches run in parallel on the server.
"""
import sys
sys.path.append("..")

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from config import (
    COLLECTION_NAME,
    DENSE_MODEL,
    PREFETCH_LIMIT,
    QDRANT_URL,
    SPARSE_MODEL,
)


class HybridSearcher:
    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def _embed(self, query: str) -> tuple[list[float], SparseVector]:
        dense_vec = list(self.dense_model.embed([query]))[0].tolist()
        sparse_result = list(self.sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )
        return dense_vec, sparse_vec

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[ScoredPoint]:
        dense_vec, sparse_vec = self._embed(query)

        qdrant_filter = None
        if category:
            qdrant_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

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
        return results.points


def print_results(results: list[ScoredPoint], label: str = "Hybrid (RRF)") -> None:
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    for i, r in enumerate(results, 1):
        name = r.payload.get("name", "?")
        print(f"  {i}. [{r.score:.3f}]  {name}")


if __name__ == "__main__":
    query = "iPhone 15 Pro Max 256GB"
    print(f"Query: '{query}'")

    client = QdrantClient(url=QDRANT_URL)
    searcher = HybridSearcher(client)
    results = searcher.search(query)
    print_results(results)

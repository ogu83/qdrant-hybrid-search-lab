"""
Hybrid search: dense + sparse (BM25) via Qdrant prefetch API with RRF fusion.
Single network request. Both searches run in parallel on the server.
"""
import argparse
import json
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

    def build_query_request(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> dict:
        dense_vec, sparse_vec = self._embed(query)
        request: dict = {
            "prefetch": [
                {"query": dense_vec, "using": "dense", "limit": PREFETCH_LIMIT},
                {
                    "query": {
                        "indices": sparse_vec.indices,
                        "values": sparse_vec.values,
                    },
                    "using": "sparse",
                    "limit": PREFETCH_LIMIT,
                },
            ],
            "query": {"fusion": "rrf"},
            "limit": top_k,
            "with_payload": True,
        }
        if category:
            request["filter"] = {
                "must": [{"key": "category", "match": {"value": category}}]
            }
        return request

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
            query_filter=qdrant_filter,
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
        sku = r.payload.get("sku", "?")
        print(f"  {i}. [{r.score:.3f}]  id={r.id} sku={sku}  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="iPhone 15 Pro Max 256GB",
        help="Query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final fused results.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter.",
    )
    parser.add_argument(
        "--show-request",
        action="store_true",
        help="Print query payload for Qdrant dashboard/API explorer demo.",
    )
    args = parser.parse_args()

    print(f"Query: '{args.query}'")

    client = QdrantClient(url=QDRANT_URL)
    searcher = HybridSearcher(client)
    if args.show_request:
        request = searcher.build_query_request(
            query=args.query,
            top_k=args.top_k,
            category=args.category,
        )
        print("\nPOST /collections/products/points/query payload:")
        print(json.dumps(request, indent=2))
        print("\nDashboard: http://localhost:6333/dashboard")

    results = searcher.search(
        query=args.query,
        top_k=args.top_k,
        category=args.category,
    )
    print_results(results)

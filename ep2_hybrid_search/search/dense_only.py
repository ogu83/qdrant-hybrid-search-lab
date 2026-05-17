"""
Baseline: pure dense semantic search.
Used for comparison against hybrid in comparison.py.
"""
import sys
sys.path.append("..")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from config import COLLECTION_NAME, DENSE_MODEL, QDRANT_URL


def dense_search(client: QdrantClient, query: str, top_k: int = 5) -> list[ScoredPoint]:
    model = TextEmbedding(model_name=DENSE_MODEL)
    query_vec = list(model.embed([query]))[0].tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", query_vec),
        limit=top_k,
        with_payload=True,
    )
    return results


def print_results(results: list[ScoredPoint], label: str = "Dense-Only") -> None:
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
    results = dense_search(client, query)
    print_results(results)

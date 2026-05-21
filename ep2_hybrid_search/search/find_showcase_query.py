"""
Find slide-ready demo queries where hybrid ranks the target product higher than dense-only.

Strategy:
- Use SKU as query text (lexical stress test).
- Evaluate target rank for dense-only vs hybrid over a sample of products.
- Print the biggest positive rank improvements first.
"""
import json
import sys
from pathlib import Path

sys.path.append("..")

from qdrant_client import QdrantClient

from config import QDRANT_URL
from search.dense_only import dense_search
from search.hybrid import HybridSearcher


def _rank_of_id(results, target_id: int) -> int | None:
    for i, r in enumerate(results, 1):
        if int(r.id) == target_id:
            return i
    return None


def main() -> None:
    data_path = Path(__file__).parent.parent / "data" / "sample_products.json"
    products = json.loads(data_path.read_text(encoding="utf-8"))

    client = QdrantClient(url=QDRANT_URL)
    hybrid = HybridSearcher(client)

    candidates = []
    for product in products[:120]:
        target_id = int(product["id"])
        query = str(product["sku"])
        dense = dense_search(client, query, top_k=10)
        hyb = hybrid.search(query, top_k=10)

        dense_rank = _rank_of_id(dense, target_id)
        hybrid_rank = _rank_of_id(hyb, target_id)

        if dense_rank is None and hybrid_rank is None:
            continue
        dense_score = dense_rank if dense_rank is not None else 99
        hybrid_score = hybrid_rank if hybrid_rank is not None else 99
        improvement = dense_score - hybrid_score
        if improvement > 0:
            candidates.append(
                {
                    "query": query,
                    "id": target_id,
                    "name": product["name"],
                    "dense_rank": dense_rank,
                    "hybrid_rank": hybrid_rank,
                    "improvement": improvement,
                }
            )

    candidates.sort(key=lambda x: x["improvement"], reverse=True)
    print("\nTop showcase queries (hybrid improves rank vs dense):")
    for item in candidates[:10]:
        print(
            f"- query={item['query']} id={item['id']} "
            f"dense_rank={item['dense_rank']} hybrid_rank={item['hybrid_rank']} "
            f"delta=+{item['improvement']}  name={item['name']}"
        )

    if not candidates:
        print("- No positive-rank examples found in first 120 products. Try re-indexing or increasing scan range.")


if __name__ == "__main__":
    main()

"""
Side-by-side benchmark: dense-only vs hybrid on 10 SKU-style test queries.
Measures precision@1 and precision@5 for each strategy.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.append("..")

from qdrant_client import QdrantClient

from config import QDRANT_URL
from search.dense_only import dense_search
from search.hybrid import HybridSearcher

TEST_QUERIES = [
    {"query": "LAP-0000", "correct_sku": "LAP-0000"},
    {"query": "HEA-0010", "correct_sku": "HEA-0010"},
    {"query": "STO-0005", "correct_sku": "STO-0005"},
    {"query": "TV-0013", "correct_sku": "TV-0013"},
    {"query": "TAB-0014", "correct_sku": "TAB-0014"},
    {"query": "PER-0021", "correct_sku": "PER-0021"},
    {"query": "SMA-0016", "correct_sku": "SMA-0016"},
    {"query": "LAP-0029", "correct_sku": "LAP-0029"},
    {"query": "TAB-0022", "correct_sku": "TAB-0022"},
    {"query": "SMA-0028", "correct_sku": "SMA-0028"},
]


def _load_sku_to_id() -> dict[str, int]:
    data_path = Path(__file__).parent.parent / "data" / "sample_products.json"
    products = json.loads(data_path.read_text(encoding="utf-8"))
    return {str(p["sku"]): int(p["id"]) for p in products}


def precision_at_k(results, correct_id: int, k: int) -> bool:
    return any(int(r.id) == correct_id for r in results[:k])


def run_benchmark(client: QdrantClient) -> None:
    hybrid_searcher = HybridSearcher(client)
    sku_to_id = _load_sku_to_id()

    dense_p1 = dense_p5 = hybrid_p1 = hybrid_p5 = 0

    print(f"\n{'Query':<45} {'Dense P@1':>9} {'Dense P@5':>9} {'Hybrid P@1':>10} {'Hybrid P@5':>10}")
    print("-" * 87)

    for item in TEST_QUERIES:
        query = item["query"]
        correct_sku = item["correct_sku"]
        correct_id = sku_to_id[correct_sku]

        dense_results = dense_search(client, query, top_k=5)
        hybrid_results = hybrid_searcher.search(query, top_k=5)

        dp1 = precision_at_k(dense_results, correct_id, 1)
        dp5 = precision_at_k(dense_results, correct_id, 5)
        hp1 = precision_at_k(hybrid_results, correct_id, 1)
        hp5 = precision_at_k(hybrid_results, correct_id, 5)

        dense_p1  += dp1
        dense_p5  += dp5
        hybrid_p1 += hp1
        hybrid_p5 += hp5

        print(f"{query:<45} {'✓' if dp1 else '✗':>9} {'✓' if dp5 else '✗':>9} "
              f"{'✓' if hp1 else '✗':>10} {'✓' if hp5 else '✗':>10}")

    n = len(TEST_QUERIES)
    print("-" * 87)
    print(f"\nPrecision@1:   Dense {dense_p1/n:.0%}   Hybrid {hybrid_p1/n:.0%}   Delta {(hybrid_p1 - dense_p1)/n:+.0%}")
    print(f"Precision@5:   Dense {dense_p5/n:.0%}   Hybrid {hybrid_p5/n:.0%}   Delta {(hybrid_p5 - dense_p5)/n:+.0%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=QDRANT_URL,
        help="Qdrant URL.",
    )
    args = parser.parse_args()
    client = QdrantClient(url=args.url)
    run_benchmark(client)

"""
Side-by-side benchmark: dense-only vs hybrid on 10 SKU-style test queries.
Measures precision@1 and precision@5 for each strategy.

Ground truth: correct_id is the product id that should appear at rank 1.
"""
import sys
sys.path.append("..")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, DENSE_MODEL, QDRANT_URL
from search.dense_only import dense_search
from search.hybrid import HybridSearcher

TEST_QUERIES = [
    {"query": "iPhone 15 Pro Max 256GB",    "correct_name": "iPhone 15 Pro Max 256GB"},
    {"query": "Samsung Galaxy S24 Ultra 512GB", "correct_name": "Samsung Galaxy S24 Ultra 512GB"},
    {"query": "MacBook Pro 14 M3 Pro 18GB RAM", "correct_name": "MacBook Pro 14-inch M3 Pro 18GB"},
    {"query": "AirPods Pro 2nd generation USB-C", "correct_name": "AirPods Pro (2nd Gen) USB-C"},
    {"query": "Sony WH-1000XM5 noise cancelling headphones", "correct_name": "Sony WH-1000XM5"},
    {"query": "iPad Air 11 M2 256GB WiFi",   "correct_name": "iPad Air 11-inch M2 256GB Wi-Fi"},
    {"query": "Samsung 990 Pro NVMe SSD 2TB", "correct_name": "Samsung 990 Pro 2TB NVMe SSD"},
    {"query": "LG C4 OLED 65 inch 4K TV",    "correct_name": "LG C4 65-inch OLED 4K TV"},
    {"query": "Logitech MX Master 3S mouse",  "correct_name": "Logitech MX Master 3S"},
    {"query": "Apple Watch Series 10 45mm GPS", "correct_name": "Apple Watch Series 10 45mm GPS"},
]


def precision_at_k(results, correct_name: str, k: int) -> bool:
    top_k_names = [r.payload.get("name", "") for r in results[:k]]
    return any(correct_name.lower() in name.lower() for name in top_k_names)


def run_benchmark(client: QdrantClient) -> None:
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    hybrid_searcher = HybridSearcher(client)

    dense_p1 = dense_p5 = hybrid_p1 = hybrid_p5 = 0

    print(f"\n{'Query':<45} {'Dense P@1':>9} {'Dense P@5':>9} {'Hybrid P@1':>10} {'Hybrid P@5':>10}")
    print("-" * 87)

    for item in TEST_QUERIES:
        query = item["query"]
        correct = item["correct_name"]

        dense_results = dense_search(client, query, top_k=5)
        hybrid_results = hybrid_searcher.search(query, top_k=5)

        dp1 = precision_at_k(dense_results, correct, 1)
        dp5 = precision_at_k(dense_results, correct, 5)
        hp1 = precision_at_k(hybrid_results, correct, 1)
        hp5 = precision_at_k(hybrid_results, correct, 5)

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
    client = QdrantClient(url=QDRANT_URL)
    run_benchmark(client)

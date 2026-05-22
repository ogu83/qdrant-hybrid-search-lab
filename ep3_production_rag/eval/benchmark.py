"""
3-way benchmark: dense-only vs hybrid vs hybrid+rerank
Run: py -3.11 -m ep3_production_rag.eval.benchmark
"""
import json
import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer, CrossEncoder

from ..config import (
    QDRANT_URL,
    COLLECTION_NAME,
    DENSE_MODEL,
    SPARSE_MODEL,
    RERANK_MODEL,
    PREFETCH_LIMIT,
)

QUERIES_FILE = Path(__file__).parent / "test_queries.json"


def precision_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / k


def hit_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def load_models():
    client = QdrantClient(QDRANT_URL)
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return client, dense_model, sparse_model, reranker


def dense_search(client, dense_model, query: str, top_k: int = 5) -> list[int]:
    vec = dense_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using="dense",
        limit=top_k,
        with_payload=False,
    )
    return [int(p.id) for p in results.points]


def hybrid_search(client, dense_model, sparse_model, query: str, top_k: int = 5) -> list[int]:
    dense_vec = dense_model.encode(query).tolist()
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=PREFETCH_LIMIT),
            Prefetch(query=sparse_vec, using="sparse", limit=PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=False,
    )
    return [int(p.id) for p in results.points]


def hybrid_rerank_search(
    client, dense_model, sparse_model, reranker, query: str, top_k: int = 5
) -> list[int]:
    def candidate_text(payload: dict) -> str:
        parts = [
            payload.get("name", ""),
            payload.get("sku", ""),
            payload.get("category", ""),
            payload.get("storage", ""),
            payload.get("color", ""),
            payload.get("description", ""),
        ]
        return " ".join(str(p).strip() for p in parts if str(p).strip())

    dense_vec = dense_model.encode(query).tolist()
    sparse_result = list(sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )
    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=PREFETCH_LIMIT),
            Prefetch(query=sparse_vec, using="sparse", limit=PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=20,
        with_payload=True,
    ).points

    texts = [candidate_text(c.payload) for c in candidates]
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [int(c.id) for c, _ in ranked[:top_k]]


def run_benchmark(
    queries_file: Path,
    strict_demo_guard: bool = False,
    strict_demo_autofix_out: Path | None = None,
) -> None:
    queries = json.loads(queries_file.read_text(encoding="utf-8"))
    client, dense_model, sparse_model, reranker = load_models()

    print("Loading models... done\n")
    print(f"{'Query':<45} {'Dense H@1':>9} {'Hybrid H@1':>10} {'H+Rerank H@1':>12}")
    print("-" * 80)

    totals = {
        "dense": {"h1": 0.0, "h5": 0.0, "r20": 0.0},
        "hybrid": {"h1": 0.0, "h5": 0.0, "r20": 0.0},
        "rerank": {"h1": 0.0, "h5": 0.0, "r20": 0.0},
    }
    n = len(queries)

    passing_queries = []
    failing_queries = []

    for item in queries:
        q = item["query"]
        relevant = item["relevant_ids"]

        d_ids = dense_search(client, dense_model, q, top_k=20)
        h_ids = hybrid_search(client, dense_model, sparse_model, q, top_k=20)
        r_ids = hybrid_rerank_search(client, dense_model, sparse_model, reranker, q, top_k=20)

        d_h1 = hit_at_k(d_ids, relevant, 1)
        h_h1 = hit_at_k(h_ids, relevant, 1)
        r_h1 = hit_at_k(r_ids, relevant, 1)

        totals["dense"]["h1"] += d_h1
        totals["hybrid"]["h1"] += h_h1
        totals["rerank"]["h1"] += r_h1

        d_h5 = hit_at_k(d_ids, relevant, 5)
        h_h5 = hit_at_k(h_ids, relevant, 5)
        r_h5 = hit_at_k(r_ids, relevant, 5)

        totals["dense"]["h5"] += d_h5
        totals["hybrid"]["h5"] += h_h5
        totals["rerank"]["h5"] += r_h5
        totals["dense"]["r20"] += recall_at_k(d_ids, relevant, 20)
        totals["hybrid"]["r20"] += recall_at_k(h_ids, relevant, 20)
        totals["rerank"]["r20"] += recall_at_k(r_ids, relevant, 20)

        guard_ok = True
        if strict_demo_guard:
            monotonic = d_h1 <= h_h1 <= r_h1
            strict_gain = (d_h1 < h_h1) or (h_h1 < r_h1)
            if not (monotonic and strict_gain):
                guard_ok = False
                failing_queries.append(
                    {
                        "query": q,
                        "relevant_ids": relevant,
                        "category": item.get("category", ""),
                        "dense_h1": d_h1,
                        "hybrid_h1": h_h1,
                        "rerank_h1": r_h1,
                    }
                )
            else:
                passing_queries.append(item)

        label = q[:43] + ".." if len(q) > 43 else q
        print(f"{label:<45} {d_h1:>9.2f} {h_h1:>10.2f} {r_h1:>12.2f}")

    print("-" * 80)
    print(f"\n{'Metric':<20} {'Dense':>10} {'Hybrid':>10} {'H+Rerank':>10}")
    print(f"{'Hit@1 (avg)':<20} {totals['dense']['h1']/n:>10.2f} {totals['hybrid']['h1']/n:>10.2f} {totals['rerank']['h1']/n:>10.2f}")
    print(f"{'Hit@5 (avg)':<20} {totals['dense']['h5']/n:>10.2f} {totals['hybrid']['h5']/n:>10.2f} {totals['rerank']['h5']/n:>10.2f}")
    print(f"{'Recall@20 (avg)':<20} {totals['dense']['r20']/n:>10.2f} {totals['hybrid']['r20']/n:>10.2f} {totals['rerank']['r20']/n:>10.2f}")

    if strict_demo_guard and failing_queries:
        print("\nStrict demo guard failures:")
        for row in failing_queries:
            print(
                f"- {row['query']} | D={row['dense_h1']:.0f} "
                f"H={row['hybrid_h1']:.0f} R={row['rerank_h1']:.0f}"
            )
        if strict_demo_autofix_out:
            strict_demo_autofix_out.write_text(
                json.dumps(passing_queries, indent=2),
                encoding="utf-8",
            )
            print(f"\nWrote strict-demo passing subset: {strict_demo_autofix_out}")
        raise RuntimeError(
            f"Strict demo guard failed for {len(failing_queries)} query(ies)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries-file",
        default=str(QUERIES_FILE),
        help="Path to benchmark queries JSON file.",
    )
    parser.add_argument(
        "--strict-demo-guard",
        action="store_true",
        help="Fail if any query violates Dense<=Hybrid<=Rerank with at least one strict gain.",
    )
    parser.add_argument(
        "--strict-demo-autofix-out",
        default="",
        help="Optional path to write only strict-demo passing queries when guard fails.",
    )
    args = parser.parse_args()
    autofix_path = Path(args.strict_demo_autofix_out) if args.strict_demo_autofix_out else None
    run_benchmark(
        Path(args.queries_file),
        strict_demo_guard=args.strict_demo_guard,
        strict_demo_autofix_out=autofix_path,
    )

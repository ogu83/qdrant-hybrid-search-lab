"""
3-way benchmark: dense-only vs hybrid vs hybrid+rerank
Run: py -3.11 -m ep3_production_rag.eval.benchmark
"""
import json
import time
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


def load_models():
    client = QdrantClient(QDRANT_URL)
    dense_model = SentenceTransformer(DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    return client, dense_model, sparse_model, reranker


def dense_search(client, dense_model, query: str, top_k: int = 5) -> list[int]:
    vec = dense_model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", vec),
        limit=top_k,
        with_payload=False,
    )
    return [int(r.id) for r in results]


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

    texts = [
        c.payload.get("name", "") + " " + c.payload.get("description", "")
        for c in candidates
    ]
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [int(c.id) for c, _ in ranked[:top_k]]


def run_benchmark() -> None:
    queries = json.loads(QUERIES_FILE.read_text())
    client, dense_model, sparse_model, reranker = load_models()

    print("Loading models... done\n")
    print(f"{'Query':<45} {'Dense P@1':>9} {'Hybrid P@1':>10} {'H+Rerank P@1':>12}")
    print("-" * 80)

    totals = {"dense": {"p1": 0.0, "p5": 0.0}, "hybrid": {"p1": 0.0, "p5": 0.0}, "rerank": {"p1": 0.0, "p5": 0.0}}
    n = len(queries)

    for item in queries:
        q = item["query"]
        relevant = item["relevant_ids"]

        d_ids = dense_search(client, dense_model, q, top_k=5)
        h_ids = hybrid_search(client, dense_model, sparse_model, q, top_k=5)
        r_ids = hybrid_rerank_search(client, dense_model, sparse_model, reranker, q, top_k=5)

        d_p1 = precision_at_k(d_ids, relevant, 1)
        h_p1 = precision_at_k(h_ids, relevant, 1)
        r_p1 = precision_at_k(r_ids, relevant, 1)

        totals["dense"]["p1"] += d_p1
        totals["hybrid"]["p1"] += h_p1
        totals["rerank"]["p1"] += r_p1

        d_p5 = precision_at_k(d_ids, relevant, 5)
        h_p5 = precision_at_k(h_ids, relevant, 5)
        r_p5 = precision_at_k(r_ids, relevant, 5)

        totals["dense"]["p5"] += d_p5
        totals["hybrid"]["p5"] += h_p5
        totals["rerank"]["p5"] += r_p5

        label = q[:43] + ".." if len(q) > 43 else q
        print(f"{label:<45} {d_p1:>9.2f} {h_p1:>10.2f} {r_p1:>12.2f}")

    print("-" * 80)
    print(f"\n{'Metric':<20} {'Dense':>10} {'Hybrid':>10} {'H+Rerank':>10}")
    print(f"{'P@1 (avg)':<20} {totals['dense']['p1']/n:>10.2f} {totals['hybrid']['p1']/n:>10.2f} {totals['rerank']['p1']/n:>10.2f}")
    print(f"{'P@5 (avg)':<20} {totals['dense']['p5']/n:>10.2f} {totals['hybrid']['p5']/n:>10.2f} {totals['rerank']['p5']/n:>10.2f}")


if __name__ == "__main__":
    run_benchmark()

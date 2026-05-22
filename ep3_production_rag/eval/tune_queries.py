"""
Auto-generate a recording-friendly EP3 benchmark query set.

Strict demo goal:
- Dense H@1 <= Hybrid H@1 <= Hybrid+Rerank H@1
- At least one strict improvement in the chain

Run:
  python -m ep3_production_rag.eval.tune_queries --strict-demo
  python -m ep3_production_rag.eval.tune_queries --strict-demo --write
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder, SentenceTransformer

from ..config import (
    COLLECTION_NAME,
    DENSE_MODEL,
    PREFETCH_LIMIT,
    QDRANT_URL,
    RERANK_MODEL,
    SPARSE_MODEL,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "ep2_hybrid_search" / "data" / "sample_products.json"
OUT_FILE = Path(__file__).parent / "test_queries_tuned.json"


def p_at_1(ids: list[int], correct_id: int) -> float:
    return 1.0 if ids and ids[0] == correct_id else 0.0


def dense_ids(client: QdrantClient, dense: SentenceTransformer, query: str, top_k: int) -> list[int]:
    vec = dense.encode(query).tolist()
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        using="dense",
        limit=top_k,
        with_payload=False,
    )
    return [int(p.id) for p in res.points]


def hybrid_points(
    client: QdrantClient,
    dense: SentenceTransformer,
    sparse: SparseTextEmbedding,
    query: str,
    top_k: int,
):
    d_vec = dense.encode(query).tolist()
    s = list(sparse.embed([query]))[0]
    s_vec = SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    return client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=d_vec, using="dense", limit=PREFETCH_LIMIT),
            Prefetch(query=s_vec, using="sparse", limit=PREFETCH_LIMIT),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points


def rerank_ids(reranker: CrossEncoder, query: str, candidates, top_k: int) -> list[int]:
    texts = [c.payload.get("name", "") + " " + c.payload.get("description", "") for c in candidates]
    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs).tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [int(c.id) for c, _ in ranked[:top_k]]


def synth_queries(item: dict) -> list[str]:
    name = str(item["name"])
    sku = str(item["sku"])
    color = str(item.get("color", ""))
    storage = str(item.get("storage", ""))
    category = str(item.get("category", ""))

    q = [
        sku,
        f"{name} {sku}",
        f"{name} {storage}".strip(),
        f"{category} {storage} {color}".strip(),
        name.replace("-", " "),
    ]
    # keep unique and non-empty
    out = []
    for x in q:
        x = " ".join(x.split())
        if x and x not in out:
            out.append(x)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write tuned set to test_queries_tuned.json")
    parser.add_argument("--target-count", type=int, default=10, help="Number of queries to output")
    parser.add_argument("--sample-size", type=int, default=140, help="How many products to sample")
    parser.add_argument(
        "--strict-demo",
        action="store_true",
        help="Keep only monotonic examples: Dense<=Hybrid<=Rerank with at least one strict gain.",
    )
    parser.add_argument(
        "--require-sku-query",
        action="store_true",
        help="Require selected query text to include the product SKU (recommended for stable demos).",
    )
    parser.add_argument(
        "--stability-trials",
        type=int,
        default=3,
        help="Repeat each query this many times and keep only stable strict-demo queries.",
    )
    args = parser.parse_args()

    products = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    random.seed(42)
    sample = products[:]
    random.shuffle(sample)
    sample = sample[: args.sample_size]

    client = QdrantClient(url=QDRANT_URL)
    dense = SentenceTransformer(DENSE_MODEL)
    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    scored = []
    for item in sample:
        pid = int(item["id"])
        for query in synth_queries(item):
            dp1_vals = []
            hp1_vals = []
            rp1_vals = []
            for _ in range(max(1, args.stability_trials)):
                d = dense_ids(client, dense, query, top_k=5)
                h_points = hybrid_points(client, dense, sparse, query, top_k=20)
                h = [int(p.id) for p in h_points[:5]]
                r = rerank_ids(reranker, query, h_points, top_k=5)

                dp1_vals.append(p_at_1(d, pid))
                hp1_vals.append(p_at_1(h, pid))
                rp1_vals.append(p_at_1(r, pid))

            # Robust score: use worst-case hit across repeated runs.
            dp1 = min(dp1_vals)
            hp1 = min(hp1_vals)
            rp1 = min(rp1_vals)

            monotonic = dp1 <= hp1 <= rp1
            strict_gain = (dp1 < hp1) or (hp1 < rp1)
            if args.strict_demo and not (monotonic and strict_gain):
                continue
            if args.require_sku_query and str(item["sku"]) not in query:
                continue

            # prioritize dense->hybrid and hybrid->rerank gains
            gain = (hp1 - dp1) * 10 + (rp1 - hp1) * 5 + rp1
            if gain > 0:
                scored.append(
                    {
                        "query": query,
                        "relevant_ids": [pid],
                        "category": item.get("category", ""),
                        "dense_p1": dp1,
                        "hybrid_p1": hp1,
                        "rerank_p1": rp1,
                        "dense_p1_trials": dp1_vals,
                        "hybrid_p1_trials": hp1_vals,
                        "rerank_p1_trials": rp1_vals,
                        "gain": gain,
                    }
                )

    scored.sort(key=lambda x: (x["gain"], x["hybrid_p1"], x["rerank_p1"]), reverse=True)

    selected = []
    seen_queries = set()
    for row in scored:
        if row["query"] in seen_queries:
            continue
        selected.append(
            {
                "query": row["query"],
                "relevant_ids": row["relevant_ids"],
                "category": row["category"],
            }
        )
        seen_queries.add(row["query"])
        if len(selected) >= args.target_count:
            break

    print("\nTuned query candidates:")
    for i, row in enumerate(scored[:20], 1):
        print(
            f"{i:>2}. {row['query']} | id={row['relevant_ids'][0]} "
            f"D={row['dense_p1']:.0f} H={row['hybrid_p1']:.0f} R={row['rerank_p1']:.0f}"
        )

    print("\nSelected set:")
    print(json.dumps(selected, indent=2))

    # Final validation pass in a fresh loop to reduce false positives from unstable ties.
    if args.strict_demo:
        validated = []
        for row in selected:
            pid = int(row["relevant_ids"][0])
            query = row["query"]
            ok = True
            for _ in range(max(1, args.stability_trials)):
                d = dense_ids(client, dense, query, top_k=5)
                h_points = hybrid_points(client, dense, sparse, query, top_k=20)
                h = [int(p.id) for p in h_points[:5]]
                r = rerank_ids(reranker, query, h_points, top_k=5)
                d1 = p_at_1(d, pid)
                h1 = p_at_1(h, pid)
                r1 = p_at_1(r, pid)
                if not (d1 <= h1 <= r1 and ((d1 < h1) or (h1 < r1))):
                    ok = False
                    break
            if ok:
                validated.append(row)

        selected = validated[: args.target_count]
        print("\nValidated strict-demo set:")
        print(json.dumps(selected, indent=2))

    if args.strict_demo and len(selected) < args.target_count:
        raise SystemExit(
            f"Not enough strict demo queries found ({len(selected)}/{args.target_count}). "
            "Increase --sample-size or reduce --target-count."
        )

    if args.write:
        OUT_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        print(f"\nWrote: {OUT_FILE}")


if __name__ == "__main__":
    main()

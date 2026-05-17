# Episode 1 — Why Hybrid Search: Architecture & Benchmarks

This folder contains the reference material for Episode 1 of the Qdrant Hybrid Search Engineering series.

**No code in this episode.** The content here is architecture diagrams, benchmark data, and the conceptual foundation for Episodes 2 and 3.

## Contents

```
ep1_concepts/
  benchmarks/
    vector_db_comparison.md     # Qdrant vs pgvector vs Pinecone vs Weaviate at 1M vectors
  architecture/
    searcher_spectrum.md        # Expert vs casual user failure modes
    bm25_mechanics.md           # TF saturation, IDF, length normalization
    rrf_example.md              # Step-by-step RRF fusion walkthrough
    hnsw_filtering.md           # In-graph vs post-filter comparison
```

## Key Takeaways

### The Two Failure Modes

**Failure Mode 1 — Semantic search on exact identifiers**
```
Query: "iPhone 15 Pro Max 256GB"
Dense result #1: iPhone 15 Pro Max 128GB  (cosine: 0.982)  ← WRONG
Dense result #2: iPhone 15 Pro Max 256GB  (cosine: 0.971)  ← correct
Gap: 0.011 — semantic model cannot distinguish storage size
```

**Failure Mode 2 — BM25 vocabulary mismatch**
```
Query: "lift maintenance schedule"
BM25 result: 0 matches — all documents use "elevator"
Dense result: correct documents returned (lift ↔ elevator cosine: 0.94)
```

### Why You Cannot Add BM25 and Cosine Scores Directly

| Score type | Range | Problem |
|---|---|---|
| BM25 | Unbounded (0 → 100+) | Swamps cosine if added naively |
| Cosine | 0 to 1 | Invisible next to BM25 |

Use **RRF** (rank-based, no tuning) or **DBSF** (score-magnitude, e-commerce).

### Production Benchmark (1M Vectors)

| Database | Peak QPS | p95 Latency | Notes |
|---|---|---|---|
| **Qdrant (self-hosted)** | **850–1,840** | **~8ms** | Rust, HNSW, in-graph filtering |
| Pinecone (serverless) | 340–500 | ~28ms | Managed, zero-ops |
| pgvector (HNSW) | 220–360 | ~48ms | OK to ~2-5M vectors |
| Weaviate (cloud) | ~380 | ~18ms | Multimodal, schema-first |

Source: [Qdrant benchmarks](https://qdrant.tech/benchmarks/)

### RRF Step-by-Step (k=2)

| Document | Dense Rank | Sparse Rank | Dense Score | Sparse Score | Total | Final |
|---|---|---|---|---|---|---|
| D1 | 1 | 3 | 1/(2+1)=0.333 | 1/(2+3)=0.200 | **0.533** | **1** |
| D3 | 3 | 2 | 1/(2+3)=0.200 | 1/(2+2)=0.250 | 0.450 | 2 |
| D2 | 2 | 4 | 1/(2+2)=0.250 | 1/(2+4)=0.167 | 0.417 | 3 |
| D5 | — | 1 | 0 | 1/(2+1)=0.333 | 0.333 | 4 |
| D4 | 4 | — | 1/(2+4)=0.167 | 0 | 0.167 | 5 |

**Principle:** D1 wins by consensus — high in both lists. D4 and D5 penalized for appearing in only one.

## Next

→ **Episode 2** (`ep2-hybrid-search` branch) — Build the Python implementation.

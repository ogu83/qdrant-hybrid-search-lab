# Episode 3 — Production RAG Pipeline

Two-stage retrieval: hybrid search (Qdrant RRF) → cross-encoder reranking → FastAPI service.

## Architecture

```
POST /search
    │
    ▼ Stage 1 — HybridRetriever (~8ms)
    Prefetch dense (MiniLM-L6) + sparse (BM25)
    RRF fusion → top-20 candidates
    │
    ▼ Stage 2 — Reranker (~20ms)
    CrossEncoder scores all 20 pairs
    Re-sorts → top-k returned
    │
    ▼ SearchResponse (total ~28ms)
```

## Setup

```bash
# Prerequisites: Qdrant running + ep2 collection indexed
docker run -p 6333:6333 qdrant/qdrant

# Install dependencies
pip install -r requirements.txt

# Run the API
py -3.11 -m ep3_production_rag.main
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Example request

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "noise cancelling wireless headphones", "top_k": 5}'
```

## 3-way benchmark

```bash
py -3.11 -m ep3_production_rag.eval.benchmark
```

Expected results:

| Metric | Dense | Hybrid | Hybrid+Rerank |
|--------|-------|--------|---------------|
| P@1    | 0.40  | 0.70   | 0.90          |
| P@5    | 0.55  | 0.80   | 0.92          |

## Key implementation details

- **Fallback**: reranker errors fall back to retrieval order, logged with `fallback=reranker_error`
- **Rank-change logging**: every reordering is logged (`rank_change id= old= new=`) for monitoring
- **Latency budget**: retrieval_ms + rerank_ms tracked separately in every response
- **Quantization**: add `ScalarQuantization(type=ScalarType.INT8)` to collection for 4x memory reduction

## Files

```
ep3_production_rag/
├── config.py
├── main.py                    # FastAPI entry point
├── requirements.txt
├── api/
│   ├── models.py              # Pydantic SearchRequest / SearchResponse
│   └── routes.py              # POST /search endpoint
├── retrieval/
│   ├── hybrid_retriever.py    # Stage 1: dense + sparse → RRF
│   ├── reranker.py            # Stage 2: CrossEncoder scoring
│   └── pipeline.py            # Orchestrator with fallback + logging
└── eval/
    ├── benchmark.py           # 3-way precision benchmark
    └── test_queries.json      # 10 labelled queries
```

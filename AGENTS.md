# Repository Guidelines

## Project Structure & Module Organization
This repo is organized by episode modules:
- `ep1_concepts/`: architecture notes and conceptual material.
- `ep2_hybrid_search/`: end-to-end hybrid retrieval implementation (data generation, collection setup, indexing, search demos).
- `ep3_production_rag/`: production-style FastAPI RAG service with reranking and eval scripts.

Primary code paths:
- `ep2_hybrid_search/setup/` for collection/index bootstrapping.
- `ep2_hybrid_search/search/` for dense vs hybrid comparison scripts.
- `ep3_production_rag/api/` and `ep3_production_rag/retrieval/` for service and retrieval pipeline logic.
- `ep3_production_rag/eval/` for benchmark inputs and runner.

## Build, Test, and Development Commands
Run from repository root unless noted:
- `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant` starts local Qdrant.
- `pip install -r ep2_hybrid_search/requirements.txt` installs episode 2 deps.
- `pip install -r ep3_production_rag/requirements.txt` installs episode 3 deps.
- `python ep2_hybrid_search/setup/create_collection.py` creates collection.
- `python ep2_hybrid_search/setup/index_documents.py` indexes sample products.
- `python ep2_hybrid_search/search/comparison.py` runs dense vs hybrid precision comparison.
- `py -3.11 -m ep3_production_rag.main` starts FastAPI app at `http://localhost:8000`.
- `py -3.11 -m ep3_production_rag.eval.benchmark` runs 3-way benchmark.

## Coding Style & Naming Conventions
- Python 3.11+ and PEP 8 style (4-space indentation, snake_case for functions/variables, PascalCase for classes).
- Keep modules focused by responsibility (`retrieval/`, `api/`, `eval/`).
- Prefer explicit, descriptive script names (`create_collection.py`, `index_documents.py`, `comparison.py`).

## Testing Guidelines
There is no dedicated `tests/` suite yet; validation is benchmark-driven.
- For retrieval changes, run `ep2_hybrid_search/search/comparison.py`.
- For API/pipeline changes, run `ep3_production_rag.eval.benchmark` and verify latency plus ranking quality.
- Keep evaluation data updates in `ep3_production_rag/eval/test_queries.json` small and reviewable.

## Commit & Pull Request Guidelines
Observed commit style uses concise, scope-prefixed messages, e.g.:
- `ep2-hybrid-search: collection setup, indexing, dense/hybrid search, benchmark`
- `ep3: production RAG pipeline — FastAPI + hybrid retrieval + cross-encoder reranker`

Use imperative, scoped commit subjects (`ep2: ...`, `ep3: ...`).
For PRs include:
- What changed and why.
- Local commands executed.
- Benchmark deltas (P@1/P@5, latency) when retrieval behavior changes.
- API request/response example for endpoint changes.

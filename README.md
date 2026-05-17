# Qdrant Hybrid Search Lab

Companion repository for the **Qdrant Hybrid Search Engineering** YouTube series by [Beyond The Developer](https://www.youtube.com/@BeyondTheDeveloper).

## Series

| Episode | Branch | Topic |
|---|---|---|
| [Ep 1 — Why Hybrid Search](https://youtube.com/@BeyondTheDeveloper) | `ep1-concepts` | Architecture, failure modes, BM25 mechanics, RRF vs DBSF |
| [Ep 2 — Building in Python](https://youtube.com/@BeyondTheDeveloper) | `ep2-hybrid-search` | Qdrant collection, FastEmbed BM25, prefetch API, side-by-side demo |
| [Ep 3 — Production RAG](https://youtube.com/@BeyondTheDeveloper) | `ep3-production-rag` | FastAPI service, cross-encoder reranking, quantization, benchmark |

## Quick Start

```bash
# Spin up Qdrant locally
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Qdrant dashboard → http://localhost:6333/dashboard
```

Each branch has its own `README.md` and `requirements.txt`. Switch to the branch you want and follow the instructions there.

## Stack

- **Qdrant** ≥ 1.10.0 — vector database with hybrid search and prefetch API
- **FastEmbed** — in-process BM25 sparse embeddings (no Elasticsearch)
- **sentence-transformers** — dense embeddings + cross-encoder reranking
- **FastAPI** — production search API (ep3)
- **Python** 3.11+

## Author

Oguz Koroglu — [Upwork](https://www.upwork.com/freelancers/oguzkoroglu) · [YouTube](https://www.youtube.com/@BeyondTheDeveloper)

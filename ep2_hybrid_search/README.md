# Episode 2 — Building Hybrid Search in Python

Dense + sparse BM25 vectors in one Qdrant prefetch query. No Elasticsearch.

## Setup

```bash
# 1. Spin up Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset (already included, run only if regenerating)
cd data && python generate_products.py

# 4. Create collection (with production settings)
cd setup && python create_collection.py

# 5. Index 500 products
python index_documents.py
```

## Run the Demo

```bash
cd search

# Dense-only baseline
python dense_only.py

# Hybrid search (BM25 + dense + RRF)
python hybrid.py

# Side-by-side precision benchmark
python comparison.py
```

## Slide-Ready Demo Commands

```bash
# Slide 4/5: show sparse vector indices/values while indexing
cd ../setup
python index_documents.py --preview-sparse

# Slide 6: print exact prefetch request payload for dashboard/API explorer
cd ../search
python hybrid.py --show-request

# Slide 7: filtered hybrid search demo (category-aware)
python hybrid.py --query "noise cancelling wireless headphones" --category headphones --top-k 5
```

Use the printed JSON from `--show-request` with Qdrant endpoint:
`POST /collections/products/points/query` in `http://localhost:6333/dashboard`.

## Expected Benchmark Output

```
Query                                         Dense P@1  Dense P@5  Hybrid P@1  Hybrid P@5
iPhone 15 Pro Max 256GB                           x          v           v           v
Samsung Galaxy S24 Ultra 512GB                    x          v           v           v
...

Precision@1:   Dense 40%   Hybrid 90%   Delta +50%
Precision@5:   Dense 70%   Hybrid 100%  Delta +30%
```

## Key Implementation Details

- **`on_disk=False`** in `SparseIndexParams` — keeps sparse index in memory (~8ms vs ~40ms p95)
- **`modifier=Modifier.IDF`** — server-side IDF stays accurate as collection grows
- **`Prefetch(limit=20)`** per leg — default 10 too small for good fusion
- **`create_payload_index`** on `category` — filtered search at unfiltered speed
- **Requires Qdrant ≥ 1.10.0** for the query/prefetch API

## Next

→ **Episode 3** (`ep3-production-rag` branch) — wrap this in FastAPI, add cross-encoder reranking, quantization.

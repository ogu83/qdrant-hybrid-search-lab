"""
Index 500 product records into Qdrant with dense + sparse vectors.
Run after create_collection.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append("..")

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from config import (
    BATCH_SIZE,
    COLLECTION_NAME,
    DENSE_MODEL,
    QDRANT_URL,
    SPARSE_MODEL,
)


def load_products() -> list[dict]:
    data_path = Path(__file__).parent.parent / "data" / "sample_products.json"
    with open(data_path) as f:
        return json.load(f)


def build_text(product: dict) -> str:
    return f"{product['name']} {product['description']}"


def _print_sparse_preview(text: str, indices: list[int], values: list[float]) -> None:
    print("\nSparse vector preview")
    print(f"Text: {text[:120]}...")
    print(f"Non-zero terms: {len(indices)}")
    print(f"indices[:12]: {indices[:12]}")
    print(f"values[:12]:  {[round(v, 4) for v in values[:12]]}")


def index_products(client: QdrantClient, preview_sparse: bool = False) -> None:
    print("Loading embedding models...")
    dense_model = TextEmbedding(model_name=DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    products = load_products()
    print(f"Indexing {len(products)} products in batches of {BATCH_SIZE}...")

    points: list[PointStruct] = []

    for i, product in enumerate(products):
        text = build_text(product)

        dense_vec = list(dense_model.embed([text]))[0].tolist()

        sparse_result = list(sparse_model.embed([text]))[0]
        sparse_vec = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )
        if preview_sparse and i == 0:
            _print_sparse_preview(text, sparse_vec.indices, sparse_vec.values)

        points.append(
            PointStruct(
                id=i,
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload=product,
            )
        )

        if len(points) == BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  Upserted {i + 1}/{len(products)}")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Done. {len(products)} products indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-sparse",
        action="store_true",
        help="Print sparse vector indices/values for the first indexed product.",
    )
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL)
    index_products(client, preview_sparse=args.preview_sparse)

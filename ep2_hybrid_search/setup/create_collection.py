"""
Create a Qdrant collection with named dense + sparse vector spaces.
Run once before indexing.
"""
import sys
sys.path.append("..")

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

from config import COLLECTION_NAME, DENSE_DIM, QDRANT_URL


def create_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,  # keep sparse index in memory → ~8ms p95
                ),
                modifier=Modifier.IDF,  # server-side IDF stays accurate as collection grows
            )
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

    # Payload index for fast filtered search
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema="keyword",
    )
    print("Payload index created on 'category'.")


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)
    create_collection(client)

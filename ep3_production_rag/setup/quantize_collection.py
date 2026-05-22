"""
Enable INT8 scalar quantization on the EP2/EP3 products collection.

Run:
  python -m ep3_production_rag.setup.quantize_collection
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)

from ..config import COLLECTION_NAME, QDRANT_URL


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    client.update_collection(
        collection_name=COLLECTION_NAME,
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            )
        ),
    )
    print(
        f"Quantization enabled for '{COLLECTION_NAME}' "
        "(INT8, quantile=0.99, always_ram=True)."
    )


if __name__ == "__main__":
    main()

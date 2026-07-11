import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION_NAME = "financial"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Deleta a Collection caso seja necessário
# qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(),
    },
    quant_vectors_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    ),
)

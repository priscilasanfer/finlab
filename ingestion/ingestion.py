import os
import uuid

from dotenv import load_dotenv
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from utils.edgar_client import EdgarClient
from utils.semantic_chunker import SemanticChunker

load_dotenv()

DENSE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
MAX_TOKENS = 300


qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

edgar = EdgarClient(email=os.getenv("EDGAR_EMAIL"))

data_10k = edgar.fetch_filing_data(ticker="AAPL", form_type="10-K")
text_10K = edgar.get_combined_text(data_10k)

data_10q = edgar.fetch_filing_data(ticker="AAPL", form_type="10-Q")
text_10q = edgar.get_combined_text(data_10q)

chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
for data, text in [(data_10k, text_10K), (data_10q, text_10q)]:
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})


dense_model = TextEmbedding(model_name=DENSE_NAME)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

points = []
for chunk_data in all_chunks:
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embeddings = list(dense_model.passage_embed([chunk]))[0].tolist()
    dense_sparse = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embeddings = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embeddings,
            "sparse": dense_sparse,
            "colbert": colbert_embeddings,
        },
        payload={
            "text": chunk,
            "metadata": metadata,
        },
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points, batch_size=5)

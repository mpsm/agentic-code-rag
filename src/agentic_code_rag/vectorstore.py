import logging
from typing import List

import ollama
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import VectorParams
from qdrant_client.models import Distance

from .config import (
    OLLAMA_EMBEDDING_MODEL,
    QDRANT_HOST,
    VECTOR_SIZE,
)

logger = logging.getLogger("agentic-code-rag")


def init_vector_database(project_name: str) -> QdrantClient:
    """Connect to Qdrant and ensure the collection exists, creating it if necessary."""
    vectordb = QdrantClient(QDRANT_HOST)

    try:
        vectordb.get_collection(project_name)
    except Exception:
        logger.warning(f"Collection {project_name} does not exists, creating")
        vectordb.create_collection(
            collection_name=project_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection {project_name}, vector size {VECTOR_SIZE}")

    return vectordb


def calculate_vector(text: str) -> List[float]:
    """Generate and return an embedding vector for the given text using the configured embedding model."""
    embedding_response = ollama.embed(model=OLLAMA_EMBEDDING_MODEL, input=text)
    embeddings = embedding_response["embeddings"][0]
    logger.debug(f"Calculated embedding, len={len(embeddings)}")

    return embeddings

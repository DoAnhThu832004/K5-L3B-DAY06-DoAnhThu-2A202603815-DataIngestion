import logging
import os
from typing import Protocol

os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error("Failed to load SentenceTransformer for %s: %s", model_name, e)
            raise RuntimeError(f"Cannot load embedding model {model_name}: {e}") from e

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, question: str) -> list[float]:
        if not question:
            raise ValueError("Query string cannot be empty.")
        embeddings = self.model.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings[0].tolist()

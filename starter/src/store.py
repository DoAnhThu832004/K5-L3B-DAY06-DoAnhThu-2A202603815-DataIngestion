import json
import logging
from pathlib import Path
from typing import Any
from .models import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str | Path | None = None,
        dimensions: int = 1024,
    ) -> None:
        import chromadb

        if persist_directory is None:
            persist_directory = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def create_schema(self, reset: bool = True) -> None:
        for strat in ["a", "b", "c"]:
            c_name = f"strategy_{strat}"
            if reset:
                try:
                    self.client.delete_collection(name=c_name)
                except Exception:
                    pass
            self.client.get_or_create_collection(
                name=c_name,
                metadata={"hnsw:space": "cosine"},
            )

    def count(self, strategy: str) -> int:
        collection = self.client.get_collection(name=f"strategy_{strategy.lower()}")
        return collection.count()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors")
        if not chunks:
            return

        collection = self.client.get_or_create_collection(
            name=f"strategy_{strategy.lower()}",
            metadata={"hnsw:space": "cosine"},
        )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]

        metadatas = []
        for c in chunks:
            flat_meta: dict[str, Any] = {
                "document_id": c.document_id,
                "source": c.source,
                "page": int(c.page),
            }
            for k, v in c.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[k] = v
                elif isinstance(v, list):
                    flat_meta[k] = ",".join(str(item) for item in v)
                elif v is None:
                    flat_meta[k] = ""
                else:
                    flat_meta[k] = str(v)
            metadatas.append(flat_meta)

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=vectors[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection_name = f"strategy_{strategy.lower()}"
        collection = self.client.get_collection(name=collection_name)

        where_filter = None
        if filters:
            clauses = [{k: {"$eq": v}} for k, v in filters.items()]
            if len(clauses) == 1:
                where_filter = clauses[0]
            elif len(clauses) > 1:
                where_filter = {"$and": clauses}

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            score = max(0.0, 1.0 - float(dist))
            hits.append({
                "chunk_id": cid,
                "document_id": meta.get("document_id", ""),
                "source": meta.get("source", ""),
                "page": int(meta.get("page", 0)),
                "text": doc,
                "metadata": meta,
                "score": round(score, 4),
            })

        return hits


VectorStore = ChromaVectorStore

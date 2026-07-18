from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from docintel.chunking import ProvenanceChunk


class TextEncoder(Protocol):
    def encode_documents(self, documents: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]: ...

    def encode_query(self, query: str, *, normalize_embeddings: bool) -> Sequence[float]: ...


@dataclass(frozen=True)
class SemanticHit:
    chunk_id: str
    distance: float


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def encode_documents(self, documents: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode_document(documents, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query: str, *, normalize_embeddings: bool) -> Sequence[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode_query(query, normalize_embeddings=normalize_embeddings)


class ChromaSemanticIndex:
    def __init__(
        self,
        path: Path,
        *,
        encoder: TextEncoder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.encoder = encoder or SentenceTransformerEncoder(model_name)
        self.model_name = model_name
        fingerprint = hashlib.sha256(model_name.encode()).hexdigest()[:12]
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=f"chunks-{fingerprint}",
            metadata={"embedding_model": model_name, "hnsw:space": "cosine"},
        )

    def replace_document(self, document_id: str, chunks: Sequence[ProvenanceChunk]) -> None:
        self.collection.delete(where={"document_id": document_id})
        if not chunks:
            return
        vectors = self.encoder.encode_documents([chunk.text for chunk in chunks], normalize_embeddings=True)
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=[[float(value) for value in vector] for vector in vectors],
            metadatas=[
                {
                    "document_id": document_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                }
                for chunk in chunks
            ],
        )

    def query(self, query: str, *, limit: int = 20) -> list[SemanticHit]:
        query = query.strip()
        if not query or limit <= 0 or self.collection.count() == 0:
            return []
        vector = self.encoder.encode_query(query, normalize_embeddings=True)
        result = self.collection.query(
            query_embeddings=[[float(value) for value in vector]],
            n_results=min(limit, self.collection.count()),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [SemanticHit(chunk_id=chunk_id, distance=float(distance)) for chunk_id, distance in zip(ids, distances)]

    def close(self) -> None:
        self.client.close()

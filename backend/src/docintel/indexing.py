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
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model_revision: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.query_prompt = query_prompt
        self._model = None

    def encode_documents(self, documents: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.model_revision)
        return self._model.encode_document(documents, normalize_embeddings=normalize_embeddings)

    def encode_query(self, query: str, *, normalize_embeddings: bool) -> Sequence[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.model_revision)
        return self._model.encode_query(
            query,
            prompt=self.query_prompt,
            normalize_embeddings=normalize_embeddings,
        )


class ChromaSemanticIndex:
    def __init__(
        self,
        path: Path,
        *,
        encoder: TextEncoder | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_revision: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        self.encoder = encoder or SentenceTransformerEncoder(
            model_name,
            model_revision=model_revision,
            query_prompt=query_prompt,
        )
        self.base_model_name = model_name
        self.model_revision = model_revision
        self.model_name = f"{model_name}@{model_revision}" if model_revision else model_name
        self.query_prompt = query_prompt
        fingerprint = hashlib.sha256(self.model_name.encode()).hexdigest()[:12]
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = f"chunks-{fingerprint}"
        self.collection = self._create_collection()

    def _create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.model_name, "hnsw:space": "cosine"},
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

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def cleanup_stale_collections(self) -> list[str]:
        deleted = []
        for collection in self.client.list_collections():
            if collection.name.startswith("chunks-") and collection.name != self.collection_name:
                self.client.delete_collection(collection.name)
                deleted.append(collection.name)
        return sorted(deleted)

    def reset(self) -> list[str]:
        deleted = []
        for collection in self.client.list_collections():
            if collection.name.startswith("chunks-"):
                self.client.delete_collection(collection.name)
                deleted.append(collection.name)
        self.collection = self._create_collection()
        return sorted(deleted)

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

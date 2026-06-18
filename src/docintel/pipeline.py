from __future__ import annotations

from pathlib import Path

from .chunking import PassageChunker
from .loaders import DocumentLoader
from .models import Chunk, Document, SearchResult
from .search import HashVectorIndex, HybridRanker, KeywordIndex


class RetrievalPipeline:
    def __init__(self, data_folder: Path, max_words: int = 120, loader: DocumentLoader | None = None) -> None:
        self.loader = loader or DocumentLoader()
        self.documents = self.loader.load_folder(data_folder)
        self._build_indexes(max_words=max_words)

    @classmethod
    def from_documents(cls, documents: list[Document], max_words: int = 120) -> "RetrievalPipeline":
        pipeline = cls.__new__(cls)
        pipeline.documents = documents
        pipeline._build_indexes(max_words=max_words)
        return pipeline

    def _build_indexes(self, max_words: int) -> None:
        self.chunker = PassageChunker(max_words=max_words)
        self.chunks: list[Chunk] = self.chunker.build_chunks(self.documents)
        self.keyword_index = KeywordIndex(self.chunks)
        self.vector_index = HashVectorIndex(self.chunks)
        self.hybrid_ranker = HybridRanker()

    def search(self, query: str, mode: str = "combined", limit: int = 5) -> list[SearchResult]:
        if mode == "keyword":
            return self.keyword_index.search(query, limit=limit)
        if mode == "vector":
            return self.vector_index.search(query, limit=limit)
        if mode == "combined":
            return self.hybrid_ranker.rank(
                self.chunks,
                self.keyword_index.score_all(query),
                self.vector_index.score_all(query),
                query,
                limit=limit,
            )
        if mode != "both":
            raise ValueError("mode must be one of: keyword, vector, combined, both")

        merged: dict[str, SearchResult] = {}
        for result in self.keyword_index.search(query, limit=limit):
            merged[result.chunk.id] = result
        for result in self.vector_index.search(query, limit=limit):
            existing = merged.get(result.chunk.id)
            if existing is None or result.score > existing.score:
                merged[result.chunk.id] = result
        return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:limit]

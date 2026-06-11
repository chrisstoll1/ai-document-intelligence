from __future__ import annotations

from pathlib import Path

from .chunking import split_into_chunks
from .loaders import load_documents
from .models import Chunk, SearchResult
from .search import HashVectorIndex, KeywordIndex


class RetrievalPipeline:
    def __init__(self, data_folder: Path, max_words: int = 120) -> None:
        self.documents = load_documents(data_folder)
        self.chunks: list[Chunk] = split_into_chunks(self.documents, max_words=max_words)
        self.keyword_index = KeywordIndex(self.chunks)
        self.vector_index = HashVectorIndex(self.chunks)

    def search(self, query: str, mode: str = "both", limit: int = 5) -> list[SearchResult]:
        if mode == "keyword":
            return self.keyword_index.search(query, limit=limit)
        if mode == "vector":
            return self.vector_index.search(query, limit=limit)
        if mode != "both":
            raise ValueError("mode must be one of: keyword, vector, both")

        merged: dict[str, SearchResult] = {}
        for result in self.keyword_index.search(query, limit=limit):
            merged[result.chunk.id] = result
        for result in self.vector_index.search(query, limit=limit):
            existing = merged.get(result.chunk.id)
            if existing is None or result.score > existing.score:
                merged[result.chunk.id] = result
        return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:limit]

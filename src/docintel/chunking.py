from __future__ import annotations

import re
from typing import Iterable

from .metadata import MetadataExtractor
from .models import Chunk, Document


class PassageChunker:
    def __init__(self, max_words: int = 120, overlap: int = 25, metadata: MetadataExtractor | None = None) -> None:
        self.max_words = max_words
        self.overlap = overlap
        self.metadata = metadata or MetadataExtractor()

    def build_chunks(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._build_document_chunks(document))
        return chunks

    def _build_document_chunks(self, document: Document) -> list[Chunk]:
        metadata = self.metadata.extract_for_document(document)
        passages = self._paragraph_aware_passages(document.text)
        return [
            Chunk(
                id=f"{document.id}:{index}",
                document_id=document.id,
                document_title=document.title,
                text=text,
                source_path=document.source_path,
                metadata=metadata,
            )
            for index, text in enumerate(passages)
        ]

    def _paragraph_aware_passages(self, text: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        passages: list[str] = []
        current_words: list[str] = []

        for paragraph in paragraphs:
            words = re.findall(r"\S+", paragraph)
            if not words:
                continue
            if current_words and len(current_words) + len(words) > self.max_words:
                passages.append(" ".join(current_words))
                current_words = current_words[-self.overlap:] if self.overlap > 0 else []
            if len(words) > self.max_words:
                passages.extend(self._window_words(words))
                current_words = words[-self.overlap:] if self.overlap > 0 else []
            else:
                current_words.extend(words)

        if current_words:
            passages.append(" ".join(current_words))
        return passages

    def _window_words(self, words: list[str]) -> list[str]:
        step = max(1, self.max_words - self.overlap)
        windows = []
        for start in range(0, len(words), step):
            window = words[start:start + self.max_words]
            if window:
                windows.append(" ".join(window))
            if start + self.max_words >= len(words):
                break
        return windows


def split_into_chunks(documents: Iterable[Document], max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    return PassageChunker(max_words=max_words, overlap=overlap).build_chunks(documents)

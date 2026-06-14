from __future__ import annotations

import re
from typing import Iterable

from .metadata import extract_metadata
from .models import Chunk, Document


def split_into_chunks(documents: Iterable[Document], max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        metadata = extract_metadata(document)
        passages = _paragraph_aware_passages(document.text, max_words=max_words, overlap=overlap)
        if not passages:
            continue

        for index, text in enumerate(passages):
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}",
                    document_id=document.id,
                    document_title=document.title,
                    text=text,
                    source_path=document.source_path,
                    metadata=metadata,
                )
            )
    return chunks


def _paragraph_aware_passages(text: str, max_words: int, overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    passages: list[str] = []
    current_words: list[str] = []

    for paragraph in paragraphs:
        words = re.findall(r"\S+", paragraph)
        if not words:
            continue
        if current_words and len(current_words) + len(words) > max_words:
            passages.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap > 0 else []
        if len(words) > max_words:
            passages.extend(_window_words(words, max_words=max_words, overlap=overlap))
            current_words = words[-overlap:] if overlap > 0 else []
        else:
            current_words.extend(words)

    if current_words:
        passages.append(" ".join(current_words))
    return passages


def _window_words(words: list[str], max_words: int, overlap: int) -> list[str]:
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        window = words[start:start + max_words]
        if window:
            windows.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return windows

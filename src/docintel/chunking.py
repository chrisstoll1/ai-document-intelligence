from __future__ import annotations

import re
from typing import Iterable

from .models import Chunk, Document


def split_into_chunks(documents: Iterable[Document], max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        words = re.findall(r"\S+", document.text)
        if not words:
            continue

        step = max(1, max_words - overlap)
        for index, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start:start + max_words]).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}:{index}",
                    document_id=document.id,
                    document_title=document.title,
                    text=text,
                    source_path=document.source_path,
                )
            )
            if start + max_words >= len(words):
                break
    return chunks

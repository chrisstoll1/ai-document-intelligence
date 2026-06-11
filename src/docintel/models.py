from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source_path: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    document_title: str
    text: str
    source_path: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    method: str

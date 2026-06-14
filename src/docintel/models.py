from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


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
    metadata: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    method: str
    components: dict[str, float] = field(default_factory=dict)

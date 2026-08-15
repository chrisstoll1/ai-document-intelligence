from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from docintel.chunking import ChunkRepository
from docintel.db import database_connection
from docintel.indexing import SemanticHit

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
TEXT_EXTENSIONS = {".md", ".txt"}
SEMANTIC_WEIGHT = 0.5
KEYWORD_WEIGHT = 0.5
RRF_K = 60


class TextEncoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]: ...


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
    keyword_score: float
    semantic_score: float


class SearchEngine:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        max_words: int = 120,
        overlap: int = 25,
        encoder: TextEncoder | None = None,
    ) -> None:
        self.documents = list(documents)
        self.chunks = _build_chunks(self.documents, max_words=max_words, overlap=overlap)
        self._encoder = encoder
        self._vectors: list[list[float]] | None = None
        self._term_counts = [Counter(_tokenize(chunk.text)) for chunk in self.chunks]
        self._document_frequency: dict[str, int] = defaultdict(int)
        for counts in self._term_counts:
            for term in counts:
                self._document_frequency[term] += 1
        self._average_length = sum(sum(counts.values()) for counts in self._term_counts) / max(1, len(self.chunks))

    @classmethod
    def from_folder(cls, folder: Path, **kwargs: object) -> SearchEngine:
        return cls(_load_documents(folder), **kwargs)

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query or limit <= 0 or not self.chunks:
            return []

        keyword_scores = _normalise(self._keyword_scores(query))
        semantic_scores = _normalise(self._semantic_scores(query))
        results = [
            SearchResult(
                chunk=chunk,
                score=SEMANTIC_WEIGHT * semantic_scores[chunk] + KEYWORD_WEIGHT * keyword_scores[chunk],
                keyword_score=keyword_scores[chunk],
                semantic_score=semantic_scores[chunk],
            )
            for chunk in self.chunks
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _keyword_scores(self, query: str) -> dict[Chunk, float]:
        scores: dict[Chunk, float] = {}
        total_chunks = max(1, len(self.chunks))
        for chunk, counts in zip(self.chunks, self._term_counts, strict=True):
            chunk_length = max(1, sum(counts.values()))
            score = 0.0
            for term in _tokenize(query):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self._document_frequency.get(term, 0)
                inverse_frequency = math.log(1 + (total_chunks - document_frequency + 0.5) / (document_frequency + 0.5))
                score += inverse_frequency * (
                    (frequency * 2.2) / (frequency + 1.2 * (0.25 + 0.75 * chunk_length / self._average_length))
                )
            scores[chunk] = score
        return scores

    def _semantic_scores(self, query: str) -> dict[Chunk, float]:
        if self._vectors is None:
            self._vectors = self._encode([chunk.text for chunk in self.chunks])
        query_vector = self._encode([query])[0]
        return {
            chunk: sum(left * right for left, right in zip(query_vector, vector, strict=True))
            for chunk, vector in zip(self.chunks, self._vectors, strict=True)
        }

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vectors = self._encoder.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def _load_documents(folder: Path) -> list[Document]:
    return [
        Document(
            id=path.stem.lower().replace(" ", "-"),
            title=path.stem.replace("_", " ").title(),
            text=path.read_text(encoding="utf-8"),
            source_path=str(path),
        )
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS
    ]


def _build_chunks(documents: Sequence[Document], *, max_words: int, overlap: int) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be between zero and max_words")

    chunks: list[Chunk] = []
    for document in documents:
        passages = _split_text(document.text, max_words=max_words, overlap=overlap)
        chunks.extend(
            Chunk(
                id=f"{document.id}:{index}",
                document_id=document.id,
                document_title=document.title,
                text=passage,
                source_path=document.source_path,
            )
            for index, passage in enumerate(passages)
        )
    return chunks


def _split_text(text: str, *, max_words: int, overlap: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    passages: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                passages.append(" ".join(current))
                current = []
            step = max_words - overlap
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                passages.append(" ".join(window))
                if start + max_words >= len(words):
                    break
            continue

        if current and len(current) + len(words) > max_words:
            passages.append(" ".join(current))
            carry_size = min(overlap, max_words - len(words))
            current = current[-carry_size:] if carry_size else []
        current.extend(words)

    if current:
        passages.append(" ".join(current))
    return passages


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _normalise(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0:
        return dict.fromkeys(scores, 0.0)
    return {chunk: max(0.0, score) / maximum for chunk, score in scores.items()}


class SemanticSearchIndex(Protocol):
    def query(self, query: str, *, limit: int = 20) -> list[SemanticHit]: ...


@dataclass(frozen=True)
class PersistentSearchResult:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    score: float
    keyword_rank: int | None
    semantic_rank: int | None


class HybridSearchService:
    def __init__(
        self,
        database_path: Path,
        chunks: ChunkRepository,
        semantic_index: SemanticSearchIndex,
        *,
        keyword_weight: float = KEYWORD_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
        rrf_k: int = RRF_K,
    ) -> None:
        if keyword_weight < 0 or semantic_weight < 0 or keyword_weight + semantic_weight <= 0:
            raise ValueError("Fusion weights must be non-negative with at least one positive weight")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self.database_path = database_path
        self.chunks = chunks
        self.semantic_index = semantic_index
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 5) -> list[PersistentSearchResult]:
        query = query.strip()
        if not query or limit <= 0:
            return []
        candidate_limit = max(20, limit * 4)
        keyword_hits = self.chunks.search(query, limit=candidate_limit)
        semantic_hits = self.semantic_index.query(query, limit=candidate_limit)
        keyword_ranks = {hit.chunk_id: rank for rank, hit in enumerate(keyword_hits, start=1)}
        semantic_ranks = {hit.chunk_id: rank for rank, hit in enumerate(semantic_hits, start=1)}
        candidate_ids = list(dict.fromkeys([*keyword_ranks, *semantic_ranks]))
        if not candidate_ids:
            return []

        placeholders = ", ".join("?" for _ in candidate_ids)
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT chunks.id, chunks.document_id, chunks.text, chunks.page_start, chunks.page_end,
                       COALESCE(
                           (SELECT original_filename FROM document_names
                            WHERE document_id = chunks.document_id ORDER BY id DESC LIMIT 1),
                           chunks.document_id
                       ) AS document_name
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.id IN ({placeholders})
                  AND documents.status IN ('indexed_lexical', 'ready')
                """,
                candidate_ids,
            ).fetchall()

        results = []
        for row in rows:
            keyword_rank = keyword_ranks.get(row["id"])
            semantic_rank = semantic_ranks.get(row["id"])
            score = 0.0
            if keyword_rank is not None:
                score += self.keyword_weight / (self.rrf_k + keyword_rank)
            if semantic_rank is not None:
                score += self.semantic_weight / (self.rrf_k + semantic_rank)
            results.append(
                PersistentSearchResult(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    document_name=row["document_name"],
                    text=row["text"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    score=score,
                    keyword_rank=keyword_rank,
                    semantic_rank=semantic_rank,
                )
            )
        missing_rank = candidate_limit + 1
        return sorted(
            results,
            key=lambda result: (
                -result.score,
                result.keyword_rank if result.keyword_rank is not None else missing_rank,
                result.semantic_rank if result.semantic_rank is not None else missing_rank,
                result.chunk_id,
            ),
        )[:limit]

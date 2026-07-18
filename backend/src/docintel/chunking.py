from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from docintel.db import database_connection

WORD_RE = re.compile(r"\S+")
QUERY_TERM_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class SourceBlock:
    id: int
    page_number: int
    text: str


@dataclass(frozen=True)
class ChunkSpan:
    block_id: int
    block_start: int
    block_end: int
    chunk_start: int
    chunk_end: int


@dataclass(frozen=True)
class ProvenanceChunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    page_start: int
    page_end: int
    chunker_version: str
    spans: tuple[ChunkSpan, ...]


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    score: float


@dataclass(frozen=True)
class _Segment:
    block: SourceBlock
    start: int
    end: int
    text: str
    word_count: int


class ProvenanceChunker:
    def __init__(self, *, max_words: int = 120, overlap: int = 20) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        if overlap < 0 or overlap >= max_words:
            raise ValueError("overlap must be between zero and max_words")
        self.max_words = max_words
        self.overlap = overlap
        self.version = f"blocks-v1-{max_words}-{overlap}"

    def chunk(self, document_id: str, blocks: list[SourceBlock]) -> list[ProvenanceChunk]:
        chunks: list[ProvenanceChunk] = []
        pending: list[_Segment] = []
        pending_words = 0

        def flush() -> None:
            nonlocal pending, pending_words
            if pending:
                chunks.append(self._build_chunk(document_id, len(chunks) + 1, pending))
            pending = []
            pending_words = 0

        for block in blocks:
            for segment in self._segments(block):
                if pending and pending_words + segment.word_count > self.max_words:
                    flush()
                pending.append(segment)
                pending_words += segment.word_count
                if segment.word_count >= self.max_words:
                    flush()
        flush()
        return chunks

    def _segments(self, block: SourceBlock) -> list[_Segment]:
        words = list(WORD_RE.finditer(block.text))
        if not words:
            return []
        if len(words) <= self.max_words:
            return [_Segment(block, 0, len(block.text), block.text, len(words))]

        segments = []
        step = self.max_words - self.overlap
        for start_index in range(0, len(words), step):
            selected = words[start_index : start_index + self.max_words]
            start = selected[0].start()
            end = selected[-1].end()
            segments.append(_Segment(block, start, end, block.text[start:end], len(selected)))
            if start_index + self.max_words >= len(words):
                break
        return segments

    def _build_chunk(self, document_id: str, ordinal: int, segments: list[_Segment]) -> ProvenanceChunk:
        text_parts = []
        spans = []
        chunk_offset = 0
        for segment in segments:
            if text_parts:
                chunk_offset += 1
            text_parts.append(segment.text)
            chunk_end = chunk_offset + len(segment.text)
            spans.append(
                ChunkSpan(
                    block_id=segment.block.id,
                    block_start=segment.start,
                    block_end=segment.end,
                    chunk_start=chunk_offset,
                    chunk_end=chunk_end,
                )
            )
            chunk_offset = chunk_end
        text = "\n".join(text_parts)
        identity = f"{document_id}\0{self.version}\0{ordinal}\0{text}".encode()
        return ProvenanceChunk(
            id=hashlib.sha256(identity).hexdigest(),
            document_id=document_id,
            ordinal=ordinal,
            text=text,
            page_start=min(segment.block.page_number for segment in segments),
            page_end=max(segment.block.page_number for segment in segments),
            chunker_version=self.version,
            spans=tuple(spans),
        )


class ChunkRepository:
    def __init__(self, database_path: Path, chunker: ProvenanceChunker | None = None) -> None:
        self.database_path = database_path
        self.chunker = chunker or ProvenanceChunker()

    def rebuild(self, document_id: str) -> list[ProvenanceChunk]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT blocks.id, pages.page_number, blocks.text
                FROM blocks
                JOIN pages ON pages.id = blocks.page_id
                WHERE pages.document_id = ?
                ORDER BY pages.page_number, blocks.block_order
                """,
                (document_id,),
            ).fetchall()
        blocks = [SourceBlock(id=row["id"], page_number=row["page_number"], text=row["text"]) for row in rows]
        chunks = self.chunker.chunk(document_id, blocks)

        with database_connection(self.database_path) as connection, connection:
            connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, text, page_start, page_end, chunker_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.chunker_version,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO chunk_spans (
                        chunk_id, block_id, span_order, block_start, block_end, chunk_start, chunk_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            span.block_id,
                            order,
                            span.block_start,
                            span.block_end,
                            span.chunk_start,
                            span.chunk_end,
                        )
                        for order, span in enumerate(chunk.spans, start=1)
                    ],
                )
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (chunk.id, chunk.text)
                )
            connection.execute(
                "UPDATE documents SET status = 'indexed_lexical', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (document_id,),
            )
        return chunks

    def search(self, query: str, *, limit: int = 20) -> list[LexicalHit]:
        terms = QUERY_TERM_RE.findall(query)
        if not terms or limit <= 0:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT chunks.id, chunks.text, chunks.page_start, chunks.page_end,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks ON chunks.id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            LexicalHit(
                chunk_id=row["id"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                score=row["score"],
            )
            for row in rows
        ]

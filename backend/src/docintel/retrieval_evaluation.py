from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Protocol

from docintel.chunking import ChunkRepository
from docintel.db import database_connection
from docintel.indexing import SemanticHit
from docintel.search import HybridSearchService

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_CUTOFFS = (1, 3, 5, 10)


class SemanticSearchIndex(Protocol):
    def query(self, query: str, *, limit: int = 20) -> list[SemanticHit]: ...


@dataclass(frozen=True)
class EvaluationChunk:
    id: str
    document_id: str
    text: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class QueryJudgment:
    document_id: str
    relevant_pages: frozenset[int]
    evidence_texts: tuple[str, ...]


def relevance_grade(chunk: EvaluationChunk, judgment: QueryJudgment) -> int:
    if chunk.document_id != judgment.document_id:
        return 0
    if not any(chunk.page_start <= page <= chunk.page_end for page in judgment.relevant_pages):
        return 0
    chunk_tokens = TOKEN_RE.findall(chunk.text.casefold())
    for evidence in judgment.evidence_texts:
        evidence_tokens = TOKEN_RE.findall(evidence.casefold())
        if evidence_tokens and _contains_sequence(chunk_tokens, evidence_tokens):
            return 2
    return 1


def _contains_sequence(tokens: list[str], expected: list[str]) -> bool:
    width = len(expected)
    return any(tokens[index : index + width] == expected for index in range(len(tokens) - width + 1))


def ranking_metrics(
    ranked_ids: list[str],
    grades_by_id: dict[str, int],
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, float]:
    relevant_count = sum(grade > 0 for grade in grades_by_id.values())
    if relevant_count == 0:
        raise ValueError("A query judgment must identify at least one relevant indexed chunk")

    metrics = {}
    for cutoff in cutoffs:
        relevant_retrieved = sum(grades_by_id.get(chunk_id, 0) > 0 for chunk_id in ranked_ids[:cutoff])
        metrics[f"recall_at_{cutoff}"] = relevant_retrieved / relevant_count
        metrics[f"hit_at_{cutoff}"] = float(relevant_retrieved > 0)

    evaluation_cutoff = max(cutoffs)
    first_relevant = next(
        (
            rank
            for rank, chunk_id in enumerate(ranked_ids[:evaluation_cutoff], start=1)
            if grades_by_id.get(chunk_id, 0) > 0
        ),
        None,
    )
    metrics[f"mrr_at_{evaluation_cutoff}"] = 0.0 if first_relevant is None else 1.0 / first_relevant

    ranked_grades = [grades_by_id.get(chunk_id, 0) for chunk_id in ranked_ids[:evaluation_cutoff]]
    ideal_grades = sorted(grades_by_id.values(), reverse=True)[:evaluation_cutoff]
    ideal_dcg = _discounted_cumulative_gain(ideal_grades)
    metrics[f"ndcg_at_{evaluation_cutoff}"] = (
        _discounted_cumulative_gain(ranked_grades) / ideal_dcg if ideal_dcg else 0.0
    )
    return metrics


def _discounted_cumulative_gain(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def aggregate_metrics(query_results: list[dict], modes: tuple[str, ...]) -> dict[str, dict[str, float]]:
    aggregate = {}
    for mode in modes:
        metric_names = query_results[0]["modes"][mode]["metrics"]
        aggregate[mode] = {
            name: mean(result["modes"][mode]["metrics"][name] for result in query_results)
            for name in metric_names
        }
        aggregate[mode]["mean_latency_ms"] = mean(
            result["modes"][mode]["latency_ms"] for result in query_results
        )
    return aggregate


class RetrievalEvaluator:
    MODES = ("keyword", "semantic", "hybrid")

    def __init__(
        self,
        database_path: Path,
        chunks: ChunkRepository,
        semantic_index: SemanticSearchIndex,
        hybrid_search: HybridSearchService,
    ) -> None:
        self.chunks = chunks
        self.semantic_index = semantic_index
        self.hybrid_search = hybrid_search
        self.chunk_by_id = _load_chunks(database_path)

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        judgment: QueryJudgment,
        *,
        limit: int = 10,
    ) -> dict:
        grades_by_id = {chunk_id: relevance_grade(chunk, judgment) for chunk_id, chunk in self.chunk_by_id.items()}
        if not any(grades_by_id.values()):
            raise ValueError(f"No indexed chunk matches the judgment for query {query_id}")

        rankings = {}
        started = time.perf_counter()
        rankings["keyword"] = [hit.chunk_id for hit in self.chunks.search(query, limit=limit)]
        keyword_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        rankings["semantic"] = [hit.chunk_id for hit in self.semantic_index.query(query, limit=limit)]
        semantic_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        rankings["hybrid"] = [result.chunk_id for result in self.hybrid_search.search(query, limit=limit)]
        hybrid_ms = (time.perf_counter() - started) * 1000
        latencies = {"keyword": keyword_ms, "semantic": semantic_ms, "hybrid": hybrid_ms}

        return {
            "query_id": query_id,
            "query": query,
            "judgment": {
                "document_id": judgment.document_id,
                "relevant_pages": sorted(judgment.relevant_pages),
                "relevant_chunk_count": sum(grade > 0 for grade in grades_by_id.values()),
                "evidence_chunk_count": sum(grade == 2 for grade in grades_by_id.values()),
            },
            "modes": {
                mode: {
                    "latency_ms": latencies[mode],
                    "metrics": ranking_metrics(ranked_ids, grades_by_id),
                    "ranking": [self._ranked_chunk(chunk_id, grades_by_id) for chunk_id in ranked_ids],
                }
                for mode, ranked_ids in rankings.items()
            },
        }

    def _ranked_chunk(self, chunk_id: str, grades_by_id: dict[str, int]) -> dict:
        chunk = self.chunk_by_id[chunk_id]
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "relevance_grade": grades_by_id[chunk_id],
        }


def _load_chunks(database_path: Path) -> dict[str, EvaluationChunk]:
    with database_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT id, document_id, text, page_start, page_end FROM chunks ORDER BY document_id, ordinal"
        ).fetchall()
    return {row["id"]: EvaluationChunk(**dict(row)) for row in rows}

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from .metadata import metadata_match_score
from .models import Chunk, SearchResult


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
DEFAULT_RANKING_WEIGHTS = {"vector": 0.6, "keyword": 0.3, "metadata": 0.1}


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


class KeywordIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.term_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_freq: dict[str, int] = defaultdict(int)
        for counts in self.term_counts:
            for term in counts:
                self.doc_freq[term] += 1
        self.avg_len = sum(sum(counts.values()) for counts in self.term_counts) / max(1, len(chunks))

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        scores = self.score_all(query)
        results = [
            SearchResult(chunk=chunk, score=score, method="keyword", components={"keyword": score})
            for chunk, score in scores.items()
            if score > 0
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def score_all(self, query: str) -> dict[Chunk, float]:
        query_terms = tokenize(query)
        scores: dict[Chunk, float] = {}
        total_docs = max(1, len(self.chunks))

        for chunk, counts in zip(self.chunks, self.term_counts):
            score = 0.0
            doc_len = max(1, sum(counts.values()))
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * doc_len / self.avg_len)))
            scores[chunk] = score

        return scores


class HashVectorIndex:
    _MODEL_CACHE = {}

    def __init__(self, chunks: list[Chunk], dimensions: int = 384, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.chunks = chunks
        self.dimensions = dimensions
        self.model_name = model_name
        self.backend = "hash"
        self.model = self._load_sentence_transformer(model_name)
        if self.model is None:
            self.vectors = [self._embed_hash(chunk.text) for chunk in chunks]
        else:
            self.backend = "sentence-transformers"
            self.vectors = self._embed_model([chunk.text for chunk in chunks])

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        scores = self.score_all(query)
        results = [
            SearchResult(chunk=chunk, score=score, method="vector", components={"vector": score})
            for chunk, score in scores.items()
            if score > 0
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def score_all(self, query: str) -> dict[Chunk, float]:
        query_vector = self._embed(query)
        return {
            chunk: self._cosine(query_vector, vector)
            for chunk, vector in zip(self.chunks, self.vectors)
        }

    def _embed(self, text: str) -> list[float]:
        if self.model is not None:
            return self._embed_model([text])[0]
        return self._embed_hash(text)

    def _embed_hash(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    @staticmethod
    def _load_sentence_transformer(model_name: str):
        if model_name in HashVectorIndex._MODEL_CACHE:
            return HashVectorIndex._MODEL_CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        model = SentenceTransformer(model_name)
        HashVectorIndex._MODEL_CACHE[model_name] = model
        return model

    def _embed_model(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]


class HybridRanker:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_RANKING_WEIGHTS

    def rank(
        self,
        chunks: list[Chunk],
        keyword_scores: dict[Chunk, float],
        vector_scores: dict[Chunk, float],
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        keyword_norm = self._normalise(keyword_scores)
        vector_norm = self._normalise(vector_scores)
        results: list[SearchResult] = []

        for chunk in chunks:
            keyword = keyword_norm.get(chunk, 0.0)
            vector = vector_norm.get(chunk, 0.0)
            metadata = metadata_match_score(query, chunk.metadata)
            score = self._weighted_score(keyword=keyword, vector=vector, metadata=metadata)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    method="combined",
                    components={"keyword": keyword, "vector": vector, "metadata": metadata},
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _weighted_score(self, keyword: float, vector: float, metadata: float) -> float:
        return (
            self.weights["vector"] * vector
            + self.weights["keyword"] * keyword
            + self.weights["metadata"] * metadata
        )

    @staticmethod
    def _normalise(scores: dict[Chunk, float]) -> dict[Chunk, float]:
        max_score = max(scores.values()) if scores else 0.0
        if max_score <= 0:
            return {chunk: 0.0 for chunk in scores}
        return {chunk: score / max_score for chunk, score in scores.items()}


def combined_search(
    chunks: list[Chunk],
    keyword_scores: dict[Chunk, float],
    vector_scores: dict[Chunk, float],
    query: str,
    limit: int = 5,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.3,
    metadata_weight: float = 0.1,
) -> list[SearchResult]:
    weights = {"vector": vector_weight, "keyword": keyword_weight, "metadata": metadata_weight}
    return HybridRanker(weights).rank(chunks, keyword_scores, vector_scores, query, limit=limit)


def _normalise(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    return HybridRanker._normalise(scores)

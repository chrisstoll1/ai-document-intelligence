from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from .models import Chunk, SearchResult


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


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
        query_terms = tokenize(query)
        scores: list[SearchResult] = []
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
            if score > 0:
                scores.append(SearchResult(chunk=chunk, score=score, method="keyword"))

        return sorted(scores, key=lambda result: result.score, reverse=True)[:limit]


class HashVectorIndex:
    def __init__(self, chunks: list[Chunk], dimensions: int = 384) -> None:
        self.chunks = chunks
        self.dimensions = dimensions
        self.vectors = [self._embed(chunk.text) for chunk in chunks]

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_vector = self._embed(query)
        results = [
            SearchResult(chunk=chunk, score=self._cosine(query_vector, vector), method="vector")
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        results = [result for result in results if result.score > 0]
        return sorted(results, key=lambda result: result.score, reverse=True)[:limit]

    def _embed(self, text: str) -> list[float]:
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

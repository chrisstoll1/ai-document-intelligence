from collections.abc import Sequence

from docintel.search import Document, SearchEngine


class FakeEncoder:
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]:
        vectors = []
        for sentence in sentences:
            text = sentence.lower()
            if any(term in text for term in ("confidential", "exposure", "personal")):
                vectors.append([1.0, 0.0])
            elif "weather" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def test_hybrid_search_returns_semantically_relevant_document() -> None:
    engine = SearchEngine(
        [
            Document("privacy", "Privacy", "Personal data exposure requires access controls.", "privacy.md"),
            Document("weather", "Weather", "Tomorrow's weather will be warm and dry.", "weather.md"),
        ],
        encoder=FakeEncoder(),
    )

    results = engine.search("confidential information")

    assert results[0].chunk.document_id == "privacy"


def test_keyword_score_contributes_to_hybrid_ranking() -> None:
    engine = SearchEngine(
        [
            Document("target", "Target", "The zebra appears in this document.", "target.md"),
            Document("other", "Other", "This document discusses another animal.", "other.md"),
        ],
        encoder=FakeEncoder(),
    )

    results = engine.search("zebra")

    assert results[0].chunk.document_id == "target"
    assert results[0].keyword_score == 1.0


def test_search_preserves_chunk_source_and_handles_empty_queries() -> None:
    engine = SearchEngine(
        [Document("source", "Source", "one two three four five six seven eight nine", "source.md")],
        max_words=5,
        overlap=2,
        encoder=FakeEncoder(),
    )

    assert [chunk.id for chunk in engine.chunks] == ["source:0", "source:1", "source:2"]
    assert all(chunk.source_path == "source.md" for chunk in engine.chunks)
    assert engine.search("   ") == []

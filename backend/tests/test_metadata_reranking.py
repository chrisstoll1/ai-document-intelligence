from docintel.metadata_reranking import metrics_for_ranking, stable_page_match_rerank


def _candidate(chunk_id: str, document_id: str, page: int, grade: int = 0) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page_start": page,
        "page_end": page,
        "relevance_grade": grade,
    }


def test_metadata_rerank_stably_promotes_exact_document_page_matches() -> None:
    ranking = [
        _candidate("first", "other", 1),
        _candidate("second", "target", 2),
        _candidate("third", "target", 3),
        _candidate("fourth", "other", 2),
    ]
    matches = [{"document_id": "target", "page_number": 2}]

    reranked = stable_page_match_rerank(ranking, matches)

    assert [candidate["chunk_id"] for candidate in reranked] == ["second", "first", "third", "fourth"]
    assert ranking[0]["chunk_id"] == "first"


def test_metadata_rerank_is_noop_without_candidate_page_match() -> None:
    ranking = [_candidate("first", "target", 1), _candidate("second", "other", 2)]

    assert stable_page_match_rerank(ranking, []) == ranking
    assert stable_page_match_rerank(ranking, [{"document_id": "target", "page_number": 2}]) == ranking


def test_metadata_metrics_retain_relevant_chunks_outside_top_ten() -> None:
    ranking = [_candidate("irrelevant", "other", 1), _candidate("relevant", "target", 1, 2)]

    metrics = metrics_for_ranking(ranking, relevant_count=3, evidence_count=1)

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["mrr_at_10"] == 0.5

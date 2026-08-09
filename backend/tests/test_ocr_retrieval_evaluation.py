import pytest
from docintel.ocr_retrieval_evaluation import paired_retrieval_summary, reciprocal_rank_at


def _query(query_id: str, grades: list[int]) -> dict:
    return {
        "query_id": query_id,
        "modes": {
            "hybrid": {
                "ranking": [
                    {"chunk_id": f"chunk-{index}", "relevance_grade": grade}
                    for index, grade in enumerate(grades)
                ]
            }
        },
    }


def test_reciprocal_rank_respects_cutoff() -> None:
    ranking = _query("q", [0, 0, 1, 0])["modes"]["hybrid"]["ranking"]

    assert reciprocal_rank_at(ranking, 2) == 0.0
    assert reciprocal_rank_at(ranking, 5) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_at(ranking, 0)


def test_paired_summary_records_query_level_wins_ties_and_losses() -> None:
    clean = [_query("a", [1]), _query("b", [0, 1]), _query("c", [0, 0])]
    degraded = [_query("a", [0, 1]), _query("b", [1]), _query("c", [0, 0])]

    summary = paired_retrieval_summary(clean, degraded)

    assert summary["clean_mrr"] == 0.5
    assert summary["degraded_mrr"] == 0.5
    assert summary["wins"] == 1
    assert summary["ties"] == 1
    assert summary["losses"] == 1


def test_paired_summary_rejects_mismatched_queries() -> None:
    with pytest.raises(ValueError, match="must match"):
        paired_retrieval_summary([_query("a", [1])], [_query("b", [1])])

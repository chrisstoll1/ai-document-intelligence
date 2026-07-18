import pytest
from docintel.retrieval_evaluation import (
    EvaluationChunk,
    QueryJudgment,
    ranking_metrics,
    relevance_grade,
)


def test_relevance_grades_page_matches_and_evidence_matches() -> None:
    judgment = QueryJudgment("document", frozenset({2}), ("$0.9 million",))

    assert relevance_grade(EvaluationChunk("a", "other", "$0.9 million", 2, 2), judgment) == 0
    assert relevance_grade(EvaluationChunk("b", "document", "$0.9 million", 1, 1), judgment) == 0
    assert relevance_grade(EvaluationChunk("c", "document", "Background context", 2, 2), judgment) == 1
    assert relevance_grade(EvaluationChunk("d", "document", "Accrued expenses were 0.9 million.", 2, 2), judgment) == 2


def test_ranking_metrics_measure_recall_rank_and_graded_gain() -> None:
    grades = {"best": 2, "relevant": 1, "other": 0}

    metrics = ranking_metrics(["other", "relevant", "best"], grades, cutoffs=(1, 3))

    assert metrics["recall_at_1"] == 0.0
    assert metrics["hit_at_1"] == 0.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["mrr_at_3"] == 0.5
    assert metrics["ndcg_at_3"] == pytest.approx(0.5868826714)


def test_ranking_metrics_reject_judgments_without_indexed_relevance() -> None:
    with pytest.raises(ValueError, match="at least one relevant"):
        ranking_metrics(["chunk"], {"chunk": 0})

import pytest
from docintel.generation import GroundedAnswer, GroundedClaim, GroundingContext
from docintel.generation_evaluation import (
    aggregate_metrics,
    evidence_page_citation_relevance,
    reference_coverage,
    token_f1,
)
from docintel.search import PersistentSearchResult

from scripts.evaluate_generation import evaluate


def _context(context_id: str, document_id: str, page: int) -> GroundingContext:
    return GroundingContext(
        context_id,
        PersistentSearchResult("chunk", document_id, "report.pdf", "Evidence", page, page, 1.0, 1, 1),
    )


def test_reference_coverage_handles_span_lists_and_numeric_tolerance() -> None:
    assert reference_coverage("The values were 50% and 43%.", ["50%", "43%"]) == 1.0
    assert reference_coverage("The result is -4.59%.", -4.59) == 1.0
    assert reference_coverage("Only 50% was stated.", ["50%", "43%"]) == 0.5
    assert reference_coverage(None, "answer") == 0.0


def test_token_f1_uses_bag_of_words_overlap() -> None:
    assert token_f1("Woolworths Group plan", "Woolworths Group Superannuation Plan") == pytest.approx(6 / 7)


def test_citation_relevance_checks_server_owned_document_and_page() -> None:
    answer = GroundedAnswer(
        status="answered",
        answer="Claim",
        claims=(GroundedClaim("Claim", ("C1", "C2")),),
        contexts=(_context("C1", "expected", 2), _context("C2", "other", 2)),
    )
    question = {
        "expected_status": "answered",
        "document_sha256": "expected",
        "relevant_pages": [2],
    }

    assert evidence_page_citation_relevance(answer, question) == 0.5


def test_aggregate_metrics_separates_answering_refusal_and_failures() -> None:
    results = [
        {
            "expected_status": "answered",
            "context_expected_status": "answered",
            "status": "answered",
            "reference_coverage": 1.0,
            "token_f1": 0.8,
            "evidence_page_citation_relevance": 1.0,
            "retrieval_evidence_available": True,
            "latency_ms": 100.0,
        },
        {
            "expected_status": "insufficient_evidence",
            "context_expected_status": "insufficient_evidence",
            "status": "generation_failed",
            "reference_coverage": None,
            "token_f1": None,
            "evidence_page_citation_relevance": None,
            "retrieval_evidence_available": False,
            "latency_ms": 200.0,
        },
    ]

    metrics = aggregate_metrics(results)

    assert metrics["valid_generation_rate"] == 0.5
    assert metrics["end_to_end_status_accuracy"] == 0.5
    assert metrics["context_status_accuracy"] == 0.5
    assert metrics["refusal_accuracy"] == 0.0
    assert metrics["answerable_reference_coverage"] == 1.0
    assert metrics["retrieval_conditioned_reference_coverage"] == 1.0
    assert metrics["unsupported_answer_rate"] == 0.0
    assert metrics["latency"]["p95_ms"] == 200.0


def test_candidate_evaluation_refuses_to_overwrite_frozen_results(tmp_path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        evaluate(tmp_path / "missing-inputs.json", output, "qwen")

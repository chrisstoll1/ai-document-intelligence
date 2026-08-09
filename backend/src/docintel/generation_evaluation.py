from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from docintel.generation import GroundedAnswer

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def normalize_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(str(value).casefold()))


def reference_values(answer: object) -> list[object]:
    return list(answer) if isinstance(answer, list) else [answer]


def reference_coverage(prediction: str | None, answer: object) -> float:
    if not prediction:
        return 0.0
    normalized_prediction = normalize_text(prediction)
    predicted_numbers = [float(match.group().replace(",", "")) for match in NUMBER_RE.finditer(prediction)]
    matches = []
    for reference in reference_values(answer):
        if isinstance(reference, int | float) and not isinstance(reference, bool):
            tolerance = max(0.01, abs(float(reference)) * 0.001)
            matches.append(any(math.isclose(value, float(reference), abs_tol=tolerance) for value in predicted_numbers))
        else:
            normalized_reference = normalize_text(reference)
            matches.append(bool(normalized_reference) and normalized_reference in normalized_prediction)
    return sum(matches) / len(matches) if matches else 0.0


def token_f1(prediction: str | None, answer: object) -> float:
    predicted = normalize_text(prediction or "").split()
    reference = normalize_text(" ".join(str(value) for value in reference_values(answer))).split()
    if not predicted or not reference:
        return float(predicted == reference)
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def cited_context_ids(answer: GroundedAnswer) -> tuple[str, ...]:
    return tuple(dict.fromkeys(citation_id for claim in answer.claims for citation_id in claim.citation_ids))


def evidence_page_citation_relevance(answer: GroundedAnswer, question: dict) -> float | None:
    if question["expected_status"] != "answered" or answer.status != "answered":
        return None
    relevant_pages = set(question["relevant_pages"])
    context_by_id = {context.context_id: context.result for context in answer.contexts}
    citations = cited_context_ids(answer)
    if not citations:
        return 0.0
    relevant = sum(
        context_by_id[citation_id].document_id == question["document_sha256"]
        and any(
            context_by_id[citation_id].page_start <= page <= context_by_id[citation_id].page_end
            for page in relevant_pages
        )
        for citation_id in citations
    )
    return relevant / len(citations)


def aggregate_metrics(results: Sequence[dict]) -> dict:
    if not results:
        raise ValueError("At least one generation result is required")
    answerable = [result for result in results if result["expected_status"] == "answered"]
    unanswerable = [result for result in results if result["expected_status"] == "insufficient_evidence"]
    conditioned_answerable = [result for result in answerable if result["retrieval_evidence_available"]]
    unsupported = [result for result in results if not result["retrieval_evidence_available"]]
    valid = [result for result in results if result["status"] != "generation_failed"]
    citation_relevance = [
        result["evidence_page_citation_relevance"]
        for result in answerable
        if result["evidence_page_citation_relevance"] is not None
    ]
    latencies = sorted(float(result["latency_ms"]) for result in results)
    percentile_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "valid_generation_rate": len(valid) / len(results),
        "end_to_end_status_accuracy": sum(result["status"] == result["expected_status"] for result in results)
        / len(results),
        "context_status_accuracy": sum(result["status"] == result["context_expected_status"] for result in results)
        / len(results),
        "answerable_reference_coverage": sum(result["reference_coverage"] for result in answerable)
        / max(1, len(answerable)),
        "retrieval_conditioned_reference_coverage": sum(
            result["reference_coverage"] for result in conditioned_answerable
        )
        / max(1, len(conditioned_answerable)),
        "answerable_token_f1": sum(result["token_f1"] for result in answerable) / max(1, len(answerable)),
        "refusal_accuracy": sum(result["status"] == "insufficient_evidence" for result in unanswerable)
        / max(1, len(unanswerable)),
        "unsupported_answer_rate": sum(result["status"] == "answered" for result in unsupported)
        / max(1, len(unsupported)),
        "evidence_page_citation_relevance": sum(citation_relevance) / max(1, len(citation_relevance)),
        "retrieval_evidence_available_rate": sum(result["retrieval_evidence_available"] for result in answerable)
        / max(1, len(answerable)),
        "latency": {
            "mean_ms": sum(latencies) / len(latencies),
            "p95_ms": latencies[percentile_index],
        },
    }

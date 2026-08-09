from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from docintel.retrieval_evaluation import ranking_metrics


def stable_page_match_rerank(ranking: Sequence[dict], matches: Sequence[dict]) -> list[dict]:
    matched_pages = {(match["document_id"], int(match["page_number"])) for match in matches}

    def matches_metadata(candidate: dict) -> bool:
        return any(
            candidate["document_id"] == document_id
            and int(candidate["page_start"]) <= page <= int(candidate["page_end"])
            for document_id, page in matched_pages
        )

    promoted = [candidate for candidate in ranking if matches_metadata(candidate)]
    remaining = [candidate for candidate in ranking if not matches_metadata(candidate)]
    return [*promoted, *remaining]


def metrics_for_ranking(ranking: Sequence[dict], *, relevant_count: int, evidence_count: int) -> dict:
    grades = {candidate["chunk_id"]: int(candidate["relevance_grade"]) for candidate in ranking}
    ranked_relevant = sum(grade > 0 for grade in grades.values())
    ranked_evidence = sum(grade == 2 for grade in grades.values())
    for index in range(max(0, evidence_count - ranked_evidence)):
        grades[f"missing-evidence-{index}"] = 2
    for index in range(max(0, relevant_count - evidence_count - (ranked_relevant - ranked_evidence))):
        grades[f"missing-relevant-{index}"] = 1
    return ranking_metrics([candidate["chunk_id"] for candidate in ranking], grades)


def aggregate_query_metrics(query_results: Sequence[dict], key: str) -> dict:
    metric_names = query_results[0][key]
    return {
        name: mean(result[key][name] for result in query_results if result[key][name] is not None)
        for name in metric_names
    }

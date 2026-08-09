from scripts.prepare_generation_benchmark import ANSWERABLE_QUOTAS, index_gold_questions, select_answerable_queries
from scripts.prepare_generation_inputs import has_retrieved_evidence


def _query(uid: str, answer_type: str) -> dict:
    return {"uid": uid, "answer_type": answer_type}


def test_generation_query_selection_is_deterministic_and_balanced() -> None:
    queries = [
        _query(f"{answer_type}-{index}", answer_type)
        for answer_type, count in ANSWERABLE_QUOTAS.items()
        for index in range(count + 2)
    ]

    first = select_answerable_queries(queries)
    second = select_answerable_queries(list(reversed(queries)))

    assert first == second
    for answer_type, expected in ANSWERABLE_QUOTAS.items():
        assert sum(query["answer_type"] == answer_type for query in first) == expected


def test_gold_question_index_uses_question_uid() -> None:
    records = [
        {"questions": [{"uid": "question-a", "answer": ["A"]}]},
        {"questions": [{"uid": "question-b", "answer": 2.5}]},
    ]

    indexed = index_gold_questions(records)

    assert indexed["question-a"]["answer"] == ["A"]
    assert indexed["question-b"]["answer"] == 2.5


def test_retrieved_evidence_requires_matching_document_and_page() -> None:
    question = {"document_sha256": "document-a", "relevant_pages": [2]}

    assert has_retrieved_evidence(question, [{"document_id": "document-a", "page_start": 1, "page_end": 2}])
    assert not has_retrieved_evidence(question, [{"document_id": "document-b", "page_start": 2, "page_end": 2}])

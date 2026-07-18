from scripts.prepare_tat_dqa import build_evidence, select_documents, select_questions


def _record(uid: str, question_ids: tuple[str, ...] = ("q1", "q2", "q3")) -> dict:
    return {
        "doc": {"uid": uid},
        "questions": [{"uid": question_id} for question_id in question_ids],
    }


def test_document_selection_is_deterministic_and_preserves_multipage_quota() -> None:
    records = [_record(f"doc-{index}") for index in range(8)]
    page_counts = {record["doc"]["uid"]: 2 if index < 3 else 1 for index, record in enumerate(records)}

    first = select_documents(records, page_counts, split="development", count=5, multipage_count=2)
    second = select_documents(list(reversed(records)), page_counts, split="development", count=5, multipage_count=2)

    assert [record["doc"]["uid"] for record in first] == [record["doc"]["uid"] for record in second]
    assert sum(page_counts[record["doc"]["uid"]] > 1 for record in first) == 2


def test_question_selection_is_deterministic() -> None:
    record = _record("document", ("question-a", "question-b", "question-c"))

    first = select_questions(record, split="locked_test")
    record["questions"].reverse()
    second = select_questions(record, split="locked_test")

    assert [question["uid"] for question in first] == [question["uid"] for question in second]
    assert len(first) == 2


def test_evidence_uses_official_character_spans_and_page_numbers() -> None:
    converted = {
        "pages": [
            {"blocks": [{"uuid": "first", "text": "irrelevant"}]},
            {"blocks": [{"uuid": "target", "text": "Revenue was 50 percent"}]},
        ]
    }
    question = {
        "uid": "question",
        "block_mapping": [{"target": [12, 14]}],
        "facts": ["50"],
    }

    assert build_evidence(question, converted) == [
        {
            "block_id": "target",
            "page": 2,
            "char_start": 12,
            "char_end": 14,
            "text": "50",
            "block_text": "Revenue was 50 percent",
        }
    ]

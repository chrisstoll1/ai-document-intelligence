from scripts.prepare_ocr_benchmark import Candidate, classify_line, select_candidates


def _candidate(uid: str, category: str, line: int) -> Candidate:
    return Candidate(
        uid,
        f"{uid}.pdf",
        1,
        line,
        f"Reference text for {uid} candidate line {line}",
        (1, 2, 100, 20),
        category,
    )


def test_line_classification_separates_prose_and_number_heavy_text() -> None:
    assert classify_line("This sentence contains enough alphabetic prose to form a realistic OCR reference.") == "prose"
    numeric = "Outstanding shares in 2019 were 13,477,758 compared with 10,692,594 in 2018"
    assert classify_line(numeric) == "number_heavy"
    assert classify_line("Short heading") is None


def test_candidate_selection_is_deterministic_and_uses_distinct_documents() -> None:
    candidates = [
        _candidate(f"doc-{index}", category, index)
        for category in ("prose", "number_heavy")
        for index in range(4)
    ]

    first = select_candidates(candidates, count_per_category=3)
    second = select_candidates(list(reversed(candidates)), count_per_category=3)

    assert first == second
    for category in ("prose", "number_heavy"):
        documents = [candidate.document_uid for candidate in first if candidate.category == category]
        assert len(documents) == len(set(documents)) == 3

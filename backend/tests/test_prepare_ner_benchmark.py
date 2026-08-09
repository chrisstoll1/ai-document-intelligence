from scripts.prepare_ner_benchmark import CandidatePassage, normalize_passage, select_candidates


def _candidate(document_uid: str, block_id: str) -> CandidatePassage:
    return CandidatePassage(
        document_uid=document_uid,
        source_json=f"{document_uid}.json",
        source_json_sha256="a" * 64,
        page=1,
        block_id=block_id,
        text=(
            f"Example Holdings published a sufficiently long candidate passage for {document_uid} and block {block_id}."
            if block_id == "block-0"
            else f"A sufficiently long general passage for {document_uid} and block {block_id}."
        ),
    )


def test_passage_normalization_is_deterministic() -> None:
    assert normalize_passage("  Alpha\u00a0Ltd\nLondon  ") == "Alpha Ltd London"


def test_candidate_selection_is_deterministic_and_balanced_by_document() -> None:
    candidates = [_candidate(document, f"block-{index}") for document in ("a", "b") for index in range(4)]

    first = select_candidates(candidates, count_per_document=2)
    second = select_candidates(list(reversed(candidates)), count_per_document=2)

    assert first == second
    assert [item.candidate.document_uid for item in first].count("a") == 2
    assert [item.candidate.document_uid for item in first].count("b") == 2
    assert [item.stratum for item in first].count("proper_name_challenge") == 2
    assert [item.stratum for item in first].count("general") == 2

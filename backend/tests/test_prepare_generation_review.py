from scripts.prepare_generation_review import candidate_aliases


def test_candidate_aliases_are_deterministic_and_model_blind() -> None:
    results = [{"model": {"key": "qwen"}}, {"model": {"key": "mistral"}}]

    first = candidate_aliases(results)
    second = candidate_aliases(list(reversed(results)))

    assert first == second
    assert set(first.values()) == {"Candidate A", "Candidate B"}

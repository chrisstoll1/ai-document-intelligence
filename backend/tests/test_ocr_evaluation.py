import pytest
from docintel.ocr_evaluation import aggregate_error_counts, edit_distance, normalize_ocr_text, text_error_counts


def test_ocr_normalization_preserves_case_and_punctuation() -> None:
    assert normalize_ocr_text("  Share\u2011based\nPAYMENTS  ") == "Share\u2010based PAYMENTS"


def test_edit_distance_handles_insertions_deletions_and_substitutions() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["revenue", "rose"], ["revenue", "has", "risen"]) == 2


def test_text_error_counts_measure_characters_and_words() -> None:
    counts = text_error_counts("Net income was 50", "Net income was 5O")

    assert counts["reference_characters"] == 17
    assert counts["character_edits"] == 1
    assert counts["reference_words"] == 4
    assert counts["word_edits"] == 1


def test_aggregate_error_counts_uses_micro_averaging_and_latency_percentile() -> None:
    samples = [
        {**text_error_counts("abcd", "abxd"), "latency_ms": 10.0},
        {**text_error_counts("one two", "one"), "latency_ms": 30.0, "error": "failure"},
    ]

    result = aggregate_error_counts(samples)

    assert result["cer"] == pytest.approx(5 / 11)
    assert result["wer"] == pytest.approx(2 / 3)
    assert result["mean_latency_ms"] == 20.0
    assert result["p95_latency_ms"] == 30.0
    assert result["failures"] == 1

import pytest
from docintel.ner_evaluation import (
    EntitySpan,
    merge_annotations,
    relaxed_span_metrics,
    strict_span_metrics,
    validate_annotations,
)


def _manifest() -> dict:
    return {
        "split": "development",
        "labels": ["PERSON", "ORGANIZATION", "LOCATION"],
        "passages": [
            {
                "id": "passage",
                "text": "Alice joined Example Ltd in London.",
                "annotation_status": "reviewed",
                "entities": [
                    {"label": "PERSON", "start": 0, "end": 5, "text": "Alice"},
                    {"label": "ORGANIZATION", "start": 13, "end": 24, "text": "Example Ltd"},
                    {"label": "LOCATION", "start": 28, "end": 34, "text": "London"},
                ],
            }
        ],
    }


def test_annotation_validation_accepts_exact_reviewed_spans() -> None:
    validate_annotations(_manifest(), require_reviewed=True)


def test_annotation_validation_rejects_mismatched_and_overlapping_spans() -> None:
    manifest = _manifest()
    manifest["passages"][0]["entities"][0]["text"] = "Alicia"
    with pytest.raises(ValueError, match="does not match"):
        validate_annotations(manifest)

    manifest = _manifest()
    manifest["passages"][0]["entities"][1]["start"] = 4
    manifest["passages"][0]["entities"][1]["text"] = manifest["passages"][0]["text"][4:24]
    with pytest.raises(ValueError, match="overlapping"):
        validate_annotations(manifest)


def test_strict_span_metrics_require_exact_label_and_boundaries() -> None:
    references = [
        EntitySpan("PERSON", 0, 5, "Alice"),
        EntitySpan("ORGANIZATION", 13, 24, "Example Ltd"),
    ]
    predictions = [
        EntitySpan("PERSON", 0, 5, "Alice"),
        EntitySpan("ORGANIZATION", 13, 20, "Example"),
        EntitySpan("LOCATION", 28, 34, "London"),
    ]

    metrics = strict_span_metrics(references, predictions)

    assert metrics["overall"]["true_positives"] == 1
    assert metrics["overall"]["false_positives"] == 2
    assert metrics["overall"]["false_negatives"] == 1
    assert metrics["overall"]["precision"] == pytest.approx(1 / 3)
    assert metrics["overall"]["recall"] == 0.5


def test_merge_annotations_requires_exact_passage_coverage() -> None:
    manifest = _manifest()
    manifest["passages"][0]["annotation_status"] = "pending"
    manifest["passages"][0]["entities"] = []
    annotations = {
        "annotation": {"status": "awaiting_human_review"},
        "passages": [
            {
                "id": "passage",
                "annotation_status": "preannotated",
                "entities": [{"label": "PERSON", "start": 0, "end": 5, "text": "Alice"}],
            }
        ],
    }

    merged = merge_annotations(manifest, annotations)

    assert merged["annotation"]["status"] == "awaiting_human_review"
    assert merged["passages"][0]["entities"][0]["text"] == "Alice"
    with pytest.raises(ValueError, match="each manifest passage"):
        merge_annotations(manifest, {"passages": []})


def test_relaxed_metrics_match_overlapping_boundaries_once() -> None:
    references = [EntitySpan("ORGANIZATION", 10, 21, "Example Ltd")]
    predictions = [
        EntitySpan("ORGANIZATION", 10, 17, "Example"),
        EntitySpan("ORGANIZATION", 12, 21, "ample Ltd"),
    ]

    metrics = relaxed_span_metrics(references, predictions)

    assert metrics["overall"]["true_positives"] == 1
    assert metrics["overall"]["false_positives"] == 1
    assert metrics["overall"]["false_negatives"] == 0

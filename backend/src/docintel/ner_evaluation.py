from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

ENTITY_LABELS = ("PERSON", "ORGANIZATION", "LOCATION")


@dataclass(frozen=True, order=True)
class EntitySpan:
    label: str
    start: int
    end: int
    text: str


def validate_annotations(manifest: dict, *, require_reviewed: bool = False) -> None:
    if manifest.get("split") != "development":
        raise ValueError("NER annotation manifest must use the development split")
    if tuple(manifest.get("labels", ())) != ENTITY_LABELS:
        raise ValueError("NER annotation manifest has an unexpected label taxonomy")

    for passage in manifest.get("passages", []):
        status = passage.get("annotation_status")
        if status not in {"pending", "preannotated", "reviewed"}:
            raise ValueError(f"Passage {passage.get('id')} has an invalid annotation status")
        if require_reviewed and status != "reviewed":
            raise ValueError(f"Passage {passage.get('id')} has not been reviewed")
        text = passage["text"]
        previous_end = -1
        seen = set()
        for entity in sorted(passage.get("entities", []), key=lambda item: (item["start"], item["end"])):
            span = EntitySpan(entity["label"], int(entity["start"]), int(entity["end"]), entity["text"])
            if span.label not in ENTITY_LABELS:
                raise ValueError(f"Passage {passage['id']} uses unsupported label {span.label}")
            if not 0 <= span.start < span.end <= len(text):
                raise ValueError(f"Passage {passage['id']} has an out-of-range entity span")
            if text[span.start : span.end] != span.text:
                raise ValueError(f"Passage {passage['id']} entity text does not match its offsets")
            if span.start < previous_end:
                raise ValueError(f"Passage {passage['id']} contains overlapping entity spans")
            identity = (span.label, span.start, span.end)
            if identity in seen:
                raise ValueError(f"Passage {passage['id']} contains a duplicate entity span")
            seen.add(identity)
            previous_end = span.end


def merge_annotations(manifest: dict, annotations: dict) -> dict:
    passage_annotations = annotations.get("passages", [])
    by_id = {item["id"]: item for item in passage_annotations}
    expected_ids = {passage["id"] for passage in manifest.get("passages", [])}
    if set(by_id) != expected_ids or len(by_id) != len(passage_annotations):
        raise ValueError("NER annotations must contain each manifest passage exactly once")

    merged = copy.deepcopy(manifest)
    merged["annotation"] = {**merged.get("annotation", {}), **annotations.get("annotation", {})}
    annotation_set_reviewed = merged["annotation"].get("status") == "reviewed"
    for passage in merged["passages"]:
        annotation = by_id[passage["id"]]
        passage_status = annotation["annotation_status"]
        passage["annotation_status"] = "reviewed" if annotation_set_reviewed else passage_status
        passage["entities"] = copy.deepcopy(annotation.get("entities", []))
    validate_annotations(merged)
    return merged


def strict_span_metrics(
    references: Sequence[EntitySpan],
    predictions: Sequence[EntitySpan],
    *,
    labels: Sequence[str] = ENTITY_LABELS,
) -> dict:
    reference_set = {(span.label, span.start, span.end) for span in references}
    prediction_set = {(span.label, span.start, span.end) for span in predictions}
    return {
        "overall": _counts(reference_set, prediction_set),
        "by_label": {
            label: _counts(
                {span for span in reference_set if span[0] == label},
                {span for span in prediction_set if span[0] == label},
            )
            for label in labels
        },
    }


def relaxed_span_metrics(
    references: Sequence[EntitySpan],
    predictions: Sequence[EntitySpan],
    *,
    labels: Sequence[str] = ENTITY_LABELS,
) -> dict:
    return {
        "overall": _overlap_counts(references, predictions),
        "by_label": {
            label: _overlap_counts(
                [span for span in references if span.label == label],
                [span for span in predictions if span.label == label],
            )
            for label in labels
        },
    }


def _overlap_counts(references: Sequence[EntitySpan], predictions: Sequence[EntitySpan]) -> dict[str, float | int]:
    unmatched_predictions = list(predictions)
    true_positives = 0
    for reference in references:
        match = next(
            (
                prediction
                for prediction in unmatched_predictions
                if prediction.label == reference.label
                and prediction.start < reference.end
                and reference.start < prediction.end
            ),
            None,
        )
        if match is not None:
            true_positives += 1
            unmatched_predictions.remove(match)
    return _metric_counts(true_positives, len(unmatched_predictions), len(references) - true_positives)


def _counts(reference: set[tuple], prediction: set[tuple]) -> dict[str, float | int]:
    true_positives = len(reference & prediction)
    false_positives = len(prediction - reference)
    false_negatives = len(reference - prediction)
    return _metric_counts(true_positives, false_positives, false_negatives)


def _metric_counts(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float | int]:
    predicted = true_positives + false_positives
    reference = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference": reference,
        "predicted": predicted,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

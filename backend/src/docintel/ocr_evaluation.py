from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def normalize_ocr_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1] + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def text_error_counts(reference: str, prediction: str) -> dict[str, int | str]:
    normalized_reference = normalize_ocr_text(reference)
    normalized_prediction = normalize_ocr_text(prediction)
    if not normalized_reference:
        raise ValueError("OCR reference text must not be empty")
    reference_words = normalized_reference.split()
    prediction_words = normalized_prediction.split()
    return {
        "reference": normalized_reference,
        "prediction": normalized_prediction,
        "reference_characters": len(normalized_reference),
        "character_edits": edit_distance(normalized_reference, normalized_prediction),
        "reference_words": len(reference_words),
        "word_edits": edit_distance(reference_words, prediction_words),
    }


def aggregate_error_counts(samples: Sequence[dict]) -> dict[str, float | int]:
    reference_characters = sum(int(sample["reference_characters"]) for sample in samples)
    reference_words = sum(int(sample["reference_words"]) for sample in samples)
    character_edits = sum(int(sample["character_edits"]) for sample in samples)
    word_edits = sum(int(sample["word_edits"]) for sample in samples)
    if reference_characters == 0 or reference_words == 0:
        raise ValueError("OCR samples must contain reference text")
    latencies = sorted(float(sample["latency_ms"]) for sample in samples)
    percentile_index = max(0, (95 * len(latencies) + 99) // 100 - 1)
    return {
        "samples": len(samples),
        "failures": sum(bool(sample.get("error")) for sample in samples),
        "reference_characters": reference_characters,
        "character_edits": character_edits,
        "cer": character_edits / reference_characters,
        "reference_words": reference_words,
        "word_edits": word_edits,
        "wer": word_edits / reference_words,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "p95_latency_ms": latencies[percentile_index],
    }

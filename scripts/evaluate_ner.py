from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from docintel.ner_evaluation import (
    EntitySpan,
    merge_annotations,
    relaxed_span_metrics,
    strict_span_metrics,
    validate_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "ner" / "development_manifest.json"
DEFAULT_ANNOTATIONS = ROOT / "evaluation" / "ner" / "development_preannotations.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "ner_development_candidate_comparison.json"
BERT_MODEL = "dslim/bert-base-NER"
BERT_REVISION = "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc"
SPACY_MODEL = "en_core_web_trf"
SPACY_MODEL_VERSION = "3.8.0"


def load_benchmark(manifest_path: Path, annotation_path: Path, *, require_reviewed: bool) -> tuple[dict, bytes, bytes]:
    manifest_bytes = manifest_path.read_bytes()
    annotation_bytes = annotation_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    annotations = json.loads(annotation_bytes)
    if annotations.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise RuntimeError("NER annotation file does not match the benchmark manifest")
    merged = merge_annotations(manifest, annotations)
    if require_reviewed and merged.get("annotation", {}).get("status") != "reviewed":
        raise RuntimeError("NER annotations require human review before candidate evaluation")
    validate_annotations(merged, require_reviewed=require_reviewed)
    return merged, manifest_bytes, annotation_bytes


def spacy_recognizer():
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError('spaCy NER evaluation requires: pip install -e ".[ner-eval]"') from error

    pipeline = spacy.load(SPACY_MODEL)
    label_map = {
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
    }

    def recognize(text: str) -> list[EntitySpan]:
        document = pipeline(text)
        return [
            EntitySpan(label_map[entity.label_], entity.start_char, entity.end_char, entity.text)
            for entity in document.ents
            if entity.label_ in label_map
        ]

    return recognize


def bert_recognizer():
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError('BERT NER evaluation requires: pip install -e ".[ner-eval]"') from error

    classifier = pipeline(
        "token-classification",
        model=BERT_MODEL,
        revision=BERT_REVISION,
        aggregation_strategy="simple",
        device=-1,
    )
    label_map = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}

    def recognize(text: str) -> list[EntitySpan]:
        return [
            EntitySpan(
                label_map[item["entity_group"]],
                int(item["start"]),
                int(item["end"]),
                text[item["start"] : item["end"]],
            )
            for item in classifier(text)
            if item["entity_group"] in label_map
        ]

    return recognize


def passage_spans(passage: dict) -> list[EntitySpan]:
    return [
        EntitySpan(entity["label"], int(entity["start"]), int(entity["end"]), entity["text"])
        for entity in passage["entities"]
    ]


def metrics_for_results(results: list[dict]) -> dict:
    references = [span for result in results for span in result["reference_spans"]]
    predictions = [span for result in results for span in result["prediction_spans"]]
    metrics = {
        "strict": strict_span_metrics(references, predictions),
        "relaxed_overlap": relaxed_span_metrics(references, predictions),
        "by_stratum": {},
    }
    for stratum in sorted({result["stratum"] for result in results}):
        selected = [result for result in results if result["stratum"] == stratum]
        stratum_references = [span for result in selected for span in result["reference_spans"]]
        stratum_predictions = [span for result in selected for span in result["prediction_spans"]]
        metrics["by_stratum"][stratum] = strict_span_metrics(stratum_references, stratum_predictions)["overall"]
    latencies = sorted(result["latency_ms"] for result in results)
    percentile_index = max(0, (95 * len(latencies) + 99) // 100 - 1)
    metrics["latency"] = {
        "mean_ms": sum(latencies) / len(latencies),
        "p95_ms": latencies[percentile_index],
    }
    metrics["failures"] = sum(bool(result.get("error")) for result in results)
    return metrics


def evaluate_engine(name: str, recognizer, passages: list[dict]) -> dict:
    warmup_started = time.perf_counter()
    recognizer(passages[0]["text"])
    warmup_seconds = time.perf_counter() - warmup_started
    internal_results = []
    serialized_results = []
    for index, passage in enumerate(passages, start=1):
        started = time.perf_counter()
        error = None
        try:
            predictions = recognizer(passage["text"])
        except Exception as exception:
            predictions = []
            error = f"{type(exception).__name__}: {exception}"
        latency_ms = (time.perf_counter() - started) * 1000
        references = passage_spans(passage)
        internal_results.append(
            {
                "stratum": passage["stratum"],
                "reference_spans": references,
                "prediction_spans": predictions,
                "latency_ms": latency_ms,
                "error": error,
            }
        )
        serialized = {
            "id": passage["id"],
            "stratum": passage["stratum"],
            "latency_ms": latency_ms,
            "references": [span.__dict__ for span in references],
            "predictions": [span.__dict__ for span in predictions],
        }
        if error:
            serialized["error"] = error
        serialized_results.append(serialized)
        print(f"{name}: evaluated {index}/{len(passages)} ({passage['id']})")
    return {
        "warmup_seconds": warmup_seconds,
        "metrics": metrics_for_results(internal_results),
        "passages": serialized_results,
    }


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def evaluate(manifest_path: Path, annotation_path: Path, output_path: Path, engines: list[str]) -> dict:
    benchmark, manifest_bytes, annotation_bytes = load_benchmark(
        manifest_path,
        annotation_path,
        require_reviewed=True,
    )
    engine_results = {}
    for name in engines:
        initialized = time.perf_counter()
        recognizer = spacy_recognizer() if name == "spacy" else bert_recognizer()
        initialization_seconds = time.perf_counter() - initialized
        engine_results[name] = {
            "initialization_seconds": initialization_seconds,
            **evaluate_engine(name, recognizer, benchmark["passages"]),
        }

    label_counts = Counter(
        entity["label"] for passage in benchmark["passages"] for entity in passage["entities"]
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "split": benchmark["split"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "annotations_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "corpus": {
            "passages": len(benchmark["passages"]),
            "mentions": sum(label_counts.values()),
            "mentions_by_label": dict(sorted(label_counts.items())),
        },
        "protocol": {
            "primary": "Strict exact-span-and-label micro precision, recall, and F1",
            "secondary": "Same-label overlap precision, recall, and F1",
            "latency": "Warm model, sequential CPU inference; initialization and warm-up excluded",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                "en-core-web-trf": package_version("en-core-web-trf"),
                "spacy": package_version("spacy"),
                "spacy-curated-transformers": package_version("spacy-curated-transformers"),
                "torch": package_version("torch"),
                "transformers": package_version("transformers"),
            },
        },
        "models": {
            "spacy": {"package": SPACY_MODEL, "version": SPACY_MODEL_VERSION},
            "bert": {"repository": BERT_MODEL, "revision": BERT_REVISION},
        },
        "engines": engine_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare NER candidates on reviewed development annotations.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=("both", "spacy", "bert"), default="both")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        benchmark, _, _ = load_benchmark(args.manifest, args.annotations, require_reviewed=False)
        counts = Counter(entity["label"] for passage in benchmark["passages"] for entity in passage["entities"])
        print(json.dumps({"annotation": benchmark["annotation"], "mentions": counts}, indent=2))
        return 0

    engines = ["spacy", "bert"] if args.engine == "both" else [args.engine]
    result = evaluate(args.manifest, args.annotations, args.output, engines)
    print(json.dumps({name: value["metrics"] for name, value in result["engines"].items()}, indent=2))
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

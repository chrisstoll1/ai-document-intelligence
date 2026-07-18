from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytesseract
from docintel.extraction import TesseractOcrEngine
from docintel.ocr_evaluation import aggregate_error_counts, text_error_counts
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "ocr" / "development_manifest.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "ocr_development_candidate_comparison.json"
DEFAULT_EASYOCR_MODEL_DIR = ROOT / "data" / "evaluation" / "models" / "easyocr"


def tesseract_recognizer():
    engine = TesseractOcrEngine()

    def recognize(image: Image.Image) -> str:
        lines = {}
        for word in engine.recognize(image):
            lines.setdefault(word.line_key, []).append(word)
        return " ".join(
            word.text
            for line_key in sorted(lines)
            for word in sorted(lines[line_key], key=lambda item: item.left)
        )

    return recognize


def easyocr_recognizer(model_dir: Path):
    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError('EasyOCR evaluation requires: pip install -e ".[ocr-eval]"') from error

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(model_dir),
        download_enabled=True,
        verbose=False,
    )

    def recognize(image: Image.Image) -> str:
        detections = reader.readtext(np.asarray(image), detail=1, paragraph=False, decoder="greedy")
        ordered = sorted(
            detections,
            key=lambda item: min(point[0] for point in item[0]),
        )
        return " ".join(str(item[1]) for item in ordered)

    return recognize


def summarize(samples: list[dict]) -> dict:
    summary = {"overall": aggregate_error_counts(samples), "by_variant": {}, "by_category": {}}
    for field, destination in (("variant", "by_variant"), ("category", "by_category")):
        for value in sorted({sample[field] for sample in samples}):
            selected = [sample for sample in samples if sample[field] == value]
            summary[destination][value] = aggregate_error_counts(selected)
    return summary


def evaluate_engine(name: str, recognizer, samples: list[dict]) -> dict:
    warmup_image = Image.open(ROOT / samples[0]["image_file"]).convert("RGB")
    warmup_started = time.perf_counter()
    recognizer(warmup_image)
    warmup_seconds = time.perf_counter() - warmup_started

    results = []
    for index, sample in enumerate(samples, start=1):
        image = Image.open(ROOT / sample["image_file"]).convert("RGB")
        started = time.perf_counter()
        error = None
        try:
            prediction = recognizer(image)
        except Exception as exception:
            prediction = ""
            error = f"{type(exception).__name__}: {exception}"
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "id": sample["id"],
            "base_id": sample["base_id"],
            "category": sample["category"],
            "variant": sample["variant"],
            "latency_ms": latency_ms,
            **text_error_counts(sample["reference"], prediction),
        }
        if error:
            result["error"] = error
        results.append(result)
        print(f"{name}: evaluated {index}/{len(samples)} ({sample['id']})")
    return {"warmup_seconds": warmup_seconds, "metrics": summarize(results), "samples": results}


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def evaluate(manifest_path: Path, output_path: Path, engines: list[str], easyocr_model_dir: Path) -> dict:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = manifest["samples"]
    engine_results = {}
    for name in engines:
        initialized = time.perf_counter()
        recognizer = tesseract_recognizer() if name == "tesseract" else easyocr_recognizer(easyocr_model_dir)
        initialization_seconds = time.perf_counter() - initialized
        engine_results[name] = {
            "initialization_seconds": initialization_seconds,
            **evaluate_engine(name, recognizer, samples),
        }

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "split": manifest["split"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "protocol": {
            "normalization": "Unicode NFKC and whitespace collapsing; case and punctuation preserved",
            "accuracy": "Micro-averaged Levenshtein character error rate and word error rate",
            "latency": "Warm model, one sequential CPU inference per image; image loading excluded",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                "easyocr": package_version("easyocr"),
                "numpy": package_version("numpy"),
                "opencv-python-headless": package_version("opencv-python-headless"),
                "pillow": package_version("pillow"),
                "pytesseract": package_version("pytesseract"),
                "torch": package_version("torch"),
                "torchvision": package_version("torchvision"),
            },
            "tesseract": str(pytesseract.get_tesseract_version()),
        },
        "models": {
            "tesseract": {"language": "eng", "page_segmentation_mode": 3},
            "easyocr": {"languages": ["en"], "detector": "CRAFT", "recognizer": "english_g2", "gpu": False},
        },
        "engines": engine_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare OCR candidates on the controlled development benchmark.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engine", choices=("both", "tesseract", "easyocr"), default="both")
    parser.add_argument("--easyocr-model-dir", type=Path, default=DEFAULT_EASYOCR_MODEL_DIR)
    args = parser.parse_args()
    engines = ["tesseract", "easyocr"] if args.engine == "both" else [args.engine]
    result = evaluate(args.manifest, args.output, engines, args.easyocr_model_dir)
    print(json.dumps({name: value["metrics"] for name, value in result["engines"].items()}, indent=2))
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from docintel.generation import GroundedGenerationService
from docintel.generation_evaluation import (
    aggregate_metrics,
    evidence_page_citation_relevance,
    reference_coverage,
    token_f1,
)
from docintel.generators import HuggingFaceStructuredGenerator
from docintel.search import PersistentSearchResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "evaluation" / "generation" / "development_inputs.json"
DEFAULT_RESULT_DIR = ROOT / "evaluation" / "results"
MODEL_CONFIGS = {
    "qwen": {
        "repository": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "license": "Apache-2.0",
    },
    "mistral": {
        "repository": "mistralai/Mistral-7B-Instruct-v0.3",
        "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        "license": "Apache-2.0",
    },
}


class FixedSearch:
    def __init__(self, contexts: list[PersistentSearchResult]) -> None:
        self.contexts = contexts

    def search(self, query: str, *, limit: int = 5) -> list[PersistentSearchResult]:
        return self.contexts[:limit]


def persistent_result(value: dict) -> PersistentSearchResult:
    return PersistentSearchResult(**value)


def serialize_answer(answer) -> dict:
    return {
        "status": answer.status,
        "answer": answer.answer,
        "claims": [{"text": claim.text, "citation_ids": list(claim.citation_ids)} for claim in answer.claims],
        "failure_reason": answer.failure_reason,
    }


def evaluate(
    inputs_path: Path,
    output_path: Path,
    model_key: str,
    *,
    max_new_tokens: int = 384,
    overwrite: bool = False,
) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing generation results: {output_path}")
    input_bytes = inputs_path.read_bytes()
    inputs = json.loads(input_bytes)
    initialized = time.perf_counter()
    config = MODEL_CONFIGS[model_key]
    generator = HuggingFaceStructuredGenerator(
        config["repository"],
        config["revision"],
        max_new_tokens=max_new_tokens,
    )
    generator.torch
    initialization_seconds = time.perf_counter() - initialized
    results = []
    for index, question in enumerate(inputs["questions"], start=1):
        contexts = [persistent_result(context) for context in question["contexts"]]
        service = GroundedGenerationService(FixedSearch(contexts), generator)
        generator.last_raw_output = None
        started = time.perf_counter()
        answer = service.answer(question["question"], limit=5)
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            {
                "uid": question["uid"],
                "question": question["question"],
                "expected_status": question["expected_status"],
                "context_expected_status": question["context_expected_status"],
                **serialize_answer(answer),
                "raw_output": generator.last_raw_output,
                "latency_ms": latency_ms,
                "reference_coverage": reference_coverage(answer.answer, question["answer"])
                if question["expected_status"] == "answered"
                else None,
                "token_f1": token_f1(answer.answer, question["answer"])
                if question["expected_status"] == "answered"
                else None,
                "evidence_page_citation_relevance": evidence_page_citation_relevance(answer, question),
                "retrieval_evidence_available": question["retrieval_evidence_available"],
            }
        )
        print(f"{model_key}: evaluated {index}/{len(inputs['questions'])} ({question['uid']})")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "split": inputs.get("split", "development"),
        "inputs_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "model": {**config, "key": model_key},
        "configuration": {
            "temperature": 0,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "dtype": "bfloat16",
            "structured_decoding": "lm-format-enforcer JSON Schema",
            "context_limit": 5,
            "initialization_seconds": initialization_seconds,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                "accelerate": version("accelerate"),
                "lm-format-enforcer": version("lm-format-enforcer"),
                "torch": version("torch"),
                "transformers": version("transformers"),
            },
            "gpu": generator.torch.cuda.get_device_name(0),
        },
        "metrics": aggregate_metrics(results),
        "questions": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a grounded-generation candidate on frozen retrieval inputs.")
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS), required=True)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or DEFAULT_RESULT_DIR / f"generation_development_{args.model}.json"
    result = evaluate(
        args.inputs,
        output,
        args.model,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

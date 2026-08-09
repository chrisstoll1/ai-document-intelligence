from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docintel.api import build_services
from docintel.config import Settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "evaluation" / "generation" / "development_manifest.json"
DEFAULT_DATA_DIR = ROOT / "data" / "evaluation" / "indexes" / "tat-dqa-development"
DEFAULT_OUTPUT = ROOT / "evaluation" / "generation" / "development_inputs.json"


def context_dict(result) -> dict:
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_name": result.document_name,
        "text": result.text,
        "page_start": result.page_start,
        "page_end": result.page_end,
        "score": result.score,
        "keyword_rank": result.keyword_rank,
        "semantic_rank": result.semantic_rank,
    }


def has_retrieved_evidence(question: dict, contexts: list[dict]) -> bool:
    relevant_pages = set(question["relevant_pages"])
    return any(
        context["document_id"] == question["document_sha256"]
        and any(context["page_start"] <= page <= context["page_end"] for page in relevant_pages)
        for context in contexts
    )


def prepare(benchmark_path: Path, data_dir: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing generation inputs: {output_path}")
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    if benchmark.get("split") not in {"development", "locked_test"}:
        raise RuntimeError("Generation inputs require a development or locked-test benchmark")

    services = build_services(Settings(data_dir=data_dir))
    try:
        missing = sorted(
            {
                question["document_sha256"]
                for question in benchmark["questions"]
                if question["document_sha256"] and services.documents.get(question["document_sha256"]) is None
            }
        )
        if missing:
            raise RuntimeError(
                "Frozen development index is missing benchmark documents; run retrieval evaluation first"
            )
        questions = []
        for index, question in enumerate(benchmark["questions"], start=1):
            contexts = [context_dict(result) for result in services.search.search(question["question"], limit=5)]
            evidence_available = has_retrieved_evidence(question, contexts)
            questions.append(
                {
                    **question,
                    "retrieval_evidence_available": evidence_available,
                    "context_expected_status": "answered" if evidence_available else "insufficient_evidence",
                    "contexts": contexts,
                }
            )
            print(f"Retrieved {index}/{len(benchmark['questions'])}: {question['uid']}")
        result = {
            "schema_version": 1,
            "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
            "retrieval_configuration": {
                "version": "retrieval-v1",
                "embedding_model": services.semantic_index.base_model_name,
                "embedding_revision": services.semantic_index.model_revision,
                "chunker_version": services.search.chunks.chunker.version,
                "limit": 5,
                "fusion": "weighted_reciprocal_rank_fusion",
                "keyword_weight": services.search.keyword_weight,
                "semantic_weight": services.search.semantic_weight,
                "rrf_k": services.search.rrf_k,
            },
            "questions": questions,
        }
        if benchmark["split"] == "locked_test":
            result["split"] = "locked_test"
    finally:
        services.semantic_index.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache frozen retrieval-v1 inputs for generation candidates.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare(args.benchmark, args.data_dir, args.output, overwrite=args.overwrite)
    available = sum(question["retrieval_evidence_available"] for question in result["questions"])
    print(f"Saved {len(result['questions'])} generation inputs to {args.output}; evidence available for {available}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from docintel.api import build_services
from docintel.config import Settings
from docintel.retrieval_evaluation import QueryJudgment, RetrievalEvaluator, aggregate_metrics
from docintel.search import KEYWORD_WEIGHT, RRF_K, SEMANTIC_WEIGHT

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "evaluation" / "tat_dqa"
DEFAULT_RESULT_DIR = ROOT / "evaluation" / "results"
DEFAULT_DATA_ROOT = ROOT / "data" / "evaluation" / "indexes"


def manifest_path(split: str) -> Path:
    return MANIFEST_DIR / f"{split}_manifest.json"


def evaluate(split: str, data_dir: Path, output_path: Path) -> dict:
    source_manifest_path = manifest_path(split)
    manifest_bytes = source_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    documents = {document["uid"]: document for document in manifest["documents"]}
    services = build_services(Settings(data_dir))
    if services.semantic_index is None:
        raise RuntimeError("Semantic index is required for retrieval evaluation")

    ingestion_started = time.perf_counter()
    try:
        ingestion = []
        for document in manifest["documents"]:
            pdf_path = ROOT / document["pdf_file"]
            with pdf_path.open("rb") as pdf:
                record = services.ingestion.ingest(pdf, pdf_path.name)
            if record.id != document["pdf_sha256"]:
                raise RuntimeError(f"Document hash mismatch for {document['uid']}")
            ingestion.append({"document_uid": document["uid"], "document_id": record.id, "status": record.status})
        ingestion_seconds = time.perf_counter() - ingestion_started

        evaluator = RetrievalEvaluator(
            Settings(data_dir).database_path,
            services.search.chunks,
            services.semantic_index,
            services.search,
        )
        query_results = []
        for index, query in enumerate(manifest["queries"], start=1):
            document = documents[query["document_uid"]]
            judgment = QueryJudgment(
                document_id=document["pdf_sha256"],
                relevant_pages=frozenset(query["relevant_pages"]),
                evidence_texts=tuple(evidence["text"] for evidence in query["evidence"]),
            )
            query_results.append(evaluator.evaluate_query(query["uid"], query["question"], judgment))
            print(f"Evaluated {index}/{len(manifest['queries'])}: {query['uid']}")

        result = {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "split": split,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "corpus": {
                "documents": len(manifest["documents"]),
                "queries": len(manifest["queries"]),
                "source_split": manifest["source_split"],
            },
            "configuration": {
                "embedding_model": services.semantic_index.model_name,
                "chunker_version": services.search.chunks.chunker.version,
                "fusion": {
                    "method": "weighted_reciprocal_rank_fusion",
                    "rrf_k": RRF_K,
                    "keyword_weight": KEYWORD_WEIGHT,
                    "semantic_weight": SEMANTIC_WEIGHT,
                },
                "cutoffs": [1, 3, 5, 10],
                "relevance": {
                    "binary": "matching document and overlapping judged page",
                    "grade_2": "binary relevance plus mapped evidence token sequence",
                },
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "ingestion": {
                "elapsed_seconds": ingestion_seconds,
                "documents": ingestion,
            },
            "metrics": aggregate_metrics(query_results, RetrievalEvaluator.MODES),
            "queries": query_results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return result
    finally:
        services.semantic_index.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate lexical, semantic, and hybrid retrieval on TAT-DQA.")
    parser.add_argument("--split", choices=("development", "locked_test"), default="development")
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required to score the locked test split after configuration freeze.",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.split == "locked_test" and not args.confirm_locked_test:
        parser.error("Locked-test evaluation requires --confirm-locked-test after configuration freeze.")
    data_dir = args.data_dir or DEFAULT_DATA_ROOT / f"tat-dqa-{args.split}"
    output = args.output or DEFAULT_RESULT_DIR / f"tat_dqa_{args.split}_baseline.json"
    result = evaluate(args.split, data_dir, output)
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

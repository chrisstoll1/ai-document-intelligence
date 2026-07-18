from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from docintel.api import build_services
from docintel.config import (
    DEFAULT_CHUNK_MAX_WORDS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    Settings,
)
from docintel.retrieval_evaluation import QueryJudgment, RetrievalEvaluator, aggregate_metrics
from docintel.search import KEYWORD_WEIGHT, SEMANTIC_WEIGHT, HybridSearchService

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "evaluation" / "tat_dqa"
DEFAULT_RESULT_DIR = ROOT / "evaluation" / "results"
DEFAULT_DATA_ROOT = ROOT / "data" / "evaluation" / "indexes"


def manifest_path(split: str) -> Path:
    return MANIFEST_DIR / f"{split}_manifest.json"


def evaluate(
    split: str,
    data_dir: Path,
    output_path: Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_revision: str | None = DEFAULT_EMBEDDING_REVISION,
    query_prompt: str | None = None,
    chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    keyword_weight: float = KEYWORD_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> dict:
    source_manifest_path = manifest_path(split)
    manifest_bytes = source_manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    documents = {document["uid"]: document for document in manifest["documents"]}
    settings = Settings(
        data_dir=data_dir,
        embedding_model=embedding_model,
        embedding_revision=embedding_revision,
        embedding_query_prompt=query_prompt,
        chunk_max_words=chunk_max_words,
        chunk_overlap=chunk_overlap,
    )
    services = build_services(settings)
    if services.semantic_index is None:
        raise RuntimeError("Semantic index is required for retrieval evaluation")
    hybrid_search = HybridSearchService(
        settings.database_path,
        services.search.chunks,
        services.semantic_index,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )

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

        warmup_started = time.perf_counter()
        services.semantic_index.query(manifest["queries"][0]["question"], limit=1)
        warmup_seconds = time.perf_counter() - warmup_started

        evaluator = RetrievalEvaluator(
            settings.database_path,
            services.search.chunks,
            services.semantic_index,
            hybrid_search,
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
                "embedding_model": services.semantic_index.base_model_name,
                "embedding_revision": services.semantic_index.model_revision,
                "query_prompt": services.semantic_index.query_prompt,
                "chunker_version": services.search.chunks.chunker.version,
                "chunk_max_words": services.search.chunks.chunker.max_words,
                "chunk_overlap": services.search.chunks.chunker.overlap,
                "fusion": {
                    "method": "weighted_reciprocal_rank_fusion",
                    "rrf_k": hybrid_search.rrf_k,
                    "keyword_weight": hybrid_search.keyword_weight,
                    "semantic_weight": hybrid_search.semantic_weight,
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
                "packages": {
                    "chromadb": version("chromadb"),
                    "sentence-transformers": version("sentence-transformers"),
                    "torch": version("torch"),
                    "transformers": version("transformers"),
                },
            },
            "ingestion": {
                "elapsed_seconds": ingestion_seconds,
                "query_warmup_seconds": warmup_seconds,
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
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--query-prompt")
    parser.add_argument("--chunk-max-words", type=int, default=DEFAULT_CHUNK_MAX_WORDS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--keyword-weight", type=float, default=KEYWORD_WEIGHT)
    parser.add_argument("--semantic-weight", type=float, default=SEMANTIC_WEIGHT)
    args = parser.parse_args()

    if args.split == "locked_test" and not args.confirm_locked_test:
        parser.error("Locked-test evaluation requires --confirm-locked-test after configuration freeze.")
    model_name = _model_name(args.embedding_model)
    default_model = args.embedding_model == DEFAULT_EMBEDDING_MODEL
    embedding_revision = args.embedding_revision or (DEFAULT_EMBEDDING_REVISION if default_model else None)
    default_chunking = (
        args.chunk_max_words == DEFAULT_CHUNK_MAX_WORDS and args.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    )
    model_suffix = "" if default_model else f"-{model_name}"
    chunk_suffix = "" if default_chunking else f"-chunks{args.chunk_max_words}-{args.chunk_overlap}"
    data_suffix = f"{model_suffix}{chunk_suffix}"
    data_dir = args.data_dir or DEFAULT_DATA_ROOT / f"tat-dqa-{args.split}{data_suffix}"
    default_configuration = (
        default_model
        and default_chunking
        and args.keyword_weight == KEYWORD_WEIGHT
        and args.semantic_weight == SEMANTIC_WEIGHT
    )
    if default_configuration:
        result_name = f"tat_dqa_{args.split}_selected.json"
    else:
        keyword = round(args.keyword_weight * 100)
        semantic = round(args.semantic_weight * 100)
        prompt_suffix = "_prompted" if args.query_prompt else ""
        result_name = (
            f"tat_dqa_{args.split}_{model_name}{prompt_suffix}_kw{keyword}_sem{semantic}"
            f"_chunks{args.chunk_max_words}-{args.chunk_overlap}.json"
        )
    output = args.output or DEFAULT_RESULT_DIR / result_name
    result = evaluate(
        args.split,
        data_dir,
        output,
        embedding_model=args.embedding_model,
        embedding_revision=embedding_revision,
        query_prompt=args.query_prompt,
        chunk_max_words=args.chunk_max_words,
        chunk_overlap=args.chunk_overlap,
        keyword_weight=args.keyword_weight,
        semantic_weight=args.semantic_weight,
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved results to {output}")
    return 0


def _model_name(model: str) -> str:
    name = model.rsplit("/", 1)[-1].casefold()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


if __name__ == "__main__":
    raise SystemExit(main())

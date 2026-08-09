from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pdfplumber
from docintel.api import build_services
from docintel.config import Settings
from docintel.db import database_connection
from docintel.ocr_evaluation import text_error_counts
from docintel.ocr_retrieval_evaluation import paired_retrieval_summary
from docintel.retrieval_evaluation import QueryJudgment, RetrievalEvaluator, aggregate_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "evaluation" / "indexes"
DEFAULT_RESULT_DIR = ROOT / "evaluation" / "results"


def source_page_texts(source_manifest: dict) -> dict[tuple[str, int], str]:
    references = {}
    for document in source_manifest["documents"]:
        with pdfplumber.open(ROOT / document["pdf_file"], unicode_norm="NFKC") as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                references[(document["uid"], page_number)] = page.extract_text() or ""
    return references


def extracted_pages(database_path: Path, document_id: str) -> list[dict]:
    with database_connection(database_path) as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT page_number, text, method FROM pages WHERE document_id = ? ORDER BY page_number",
                (document_id,),
            ).fetchall()
        ]


def aggregate_ocr_quality(samples: list[dict]) -> dict:
    reference_characters = sum(int(sample["reference_characters"]) for sample in samples)
    reference_words = sum(int(sample["reference_words"]) for sample in samples)
    character_edits = sum(int(sample["character_edits"]) for sample in samples)
    word_edits = sum(int(sample["word_edits"]) for sample in samples)
    return {
        "pages": len(samples),
        "reference_characters": reference_characters,
        "character_edits": character_edits,
        "cer": character_edits / reference_characters,
        "reference_words": reference_words,
        "word_edits": word_edits,
        "wer": word_edits / reference_words,
    }


def evaluate_condition(condition: str, benchmark: dict, source: dict, data_dir: Path) -> dict:
    services = build_services(Settings(data_dir=data_dir))
    references = source_page_texts(source)
    document_by_uid = {document["uid"]: document for document in benchmark["documents"]}
    condition_ids = {}
    ocr_samples = []
    started = time.perf_counter()
    try:
        for document in benchmark["documents"]:
            pdf_path = ROOT / document["variants"][condition]["pdf_file"]
            with pdf_path.open("rb") as pdf:
                record = services.ingestion.ingest(pdf, pdf_path.name)
            condition_ids[document["uid"]] = record.id
            pages = extracted_pages(services.documents.database_path, record.id)
            if len(pages) != document["page_count"] or any(page["method"] != "ocr" for page in pages):
                raise RuntimeError(f"Not all generated pages used OCR for {document['uid']} ({condition})")
            for page in pages:
                counts = text_error_counts(references[(document["uid"], page["page_number"])], page["text"])
                ocr_samples.append(counts)
        ingestion_seconds = time.perf_counter() - started
        evaluator = RetrievalEvaluator(
            services.documents.database_path,
            services.search.chunks,
            services.semantic_index,
            services.search,
        )
        queries = []
        for query in source["queries"]:
            document = document_by_uid[query["document_uid"]]
            judgment = QueryJudgment(
                condition_ids[document["uid"]],
                frozenset(query["relevant_pages"]),
                tuple(evidence["text"] for evidence in query["evidence"]),
            )
            queries.append(evaluator.evaluate_query(query["uid"], query["question"], judgment))
        return {
            "data_dir": data_dir.relative_to(ROOT).as_posix(),
            "ingestion_seconds": ingestion_seconds,
            "ocr": aggregate_ocr_quality(ocr_samples),
            "metrics": aggregate_metrics(queries, RetrievalEvaluator.MODES),
            "queries": queries,
        }
    finally:
        services.semantic_index.close()


def evaluate(split: str, benchmark_path: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite OCR retrieval result: {output_path}")
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    source_path = ROOT / benchmark["source_manifest"]
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != benchmark["source_manifest_sha256"]:
        raise RuntimeError("OCR retrieval benchmark source manifest hash mismatch")
    source = json.loads(source_bytes)
    conditions = {
        condition: evaluate_condition(
            condition,
            benchmark,
            source,
            DEFAULT_DATA_ROOT / f"tat-dqa-{split}-ocr-{condition}",
        )
        for condition in ("clean", "degraded")
    }
    result = {
        "schema_version": 1,
        "split": split,
        "benchmark_sha256": hashlib.sha256(benchmark_bytes).hexdigest(),
        "primary": paired_retrieval_summary(
            conditions["clean"]["queries"], conditions["degraded"]["queries"], cutoff=5
        ),
        "conditions": conditions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate clean versus degraded full-page OCR retrieval.")
    parser.add_argument("--split", choices=("development", "locked_test"), default="development")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.split == "locked_test" and not args.confirm_locked_test:
        parser.error("Locked-test OCR retrieval evaluation requires --confirm-locked-test.")
    benchmark = args.benchmark or ROOT / "evaluation" / "ocr" / f"retrieval_{args.split}_manifest.json"
    output = args.output or DEFAULT_RESULT_DIR / f"ocr_retrieval_{args.split}.json"
    result = evaluate(args.split, benchmark, output, overwrite=args.overwrite)
    print(json.dumps(result["primary"], indent=2))
    print(f"Saved OCR retrieval results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

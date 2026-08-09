from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from docintel.api import build_services
from docintel.config import Settings
from docintel.metadata import PageText

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "tat_dqa" / "development_manifest.json"
DEFAULT_DATA_DIR = ROOT / "data" / "evaluation" / "indexes" / "tat-dqa-development"
DEFAULT_OUTPUT = ROOT / "evaluation" / "metadata" / "development_matches.json"


def query_match_record(query: dict, extractor, repository) -> dict:
    started = time.perf_counter()
    mentions = extractor.extract([PageText(1, query["question"])])
    extraction_ms = (time.perf_counter() - started) * 1000
    keys = sorted({(mention.label, mention.normalized_text) for mention in mentions})
    started = time.perf_counter()
    matches = repository.find_exact_pages(keys, model_version=extractor.version)
    match_ms = (time.perf_counter() - started) * 1000
    return {
        "uid": query["uid"],
        "question": query["question"],
        "entities": [
            {
                "label": mention.label,
                "text": mention.text,
                "normalized_text": mention.normalized_text,
                "char_start": mention.char_start,
                "char_end": mention.char_end,
            }
            for mention in mentions
        ],
        "matches": [match.__dict__ for match in matches],
        "latency_ms": {"query_ner": extraction_ms, "exact_match": match_ms},
    }


def prepare(manifest_path: Path, data_dir: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing metadata matches: {output_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("split") != "development":
        raise RuntimeError("Metadata policy preparation accepts only the development split")

    services = build_services(Settings(data_dir=data_dir))
    try:
        extractor = services.ingestion.metadata_extractor
        repository = services.metadata
        if extractor is None or repository is None:
            raise RuntimeError("Metadata extraction is unavailable")
        for index, document in enumerate(manifest["documents"], start=1):
            pdf_path = ROOT / document["pdf_file"]
            with pdf_path.open("rb") as pdf:
                record = services.ingestion.ingest(pdf, pdf_path.name)
            if record.metadata_status != "ready" or record.metadata_model != extractor.version:
                raise RuntimeError(f"Metadata enrichment failed for {document['uid']}")
            print(f"Enriched {index}/{len(manifest['documents'])}: {document['uid']}")

        queries = [query_match_record(query, extractor, repository) for query in manifest["queries"]]
        mention_counts = Counter(
            mention.label
            for document in manifest["documents"]
            for mention in repository.list_document(document["pdf_sha256"])
        )
        result = {
            "schema_version": 1,
            "split": "development",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "metadata_model": extractor.version,
            "policy": "Exact same-label normalized entity match with page-aware OR semantics",
            "corpus_mentions_by_label": dict(sorted(mention_counts.items())),
            "queries": queries,
        }
    finally:
        services.semantic_index.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare candidate-blind exact metadata matches.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare(args.manifest, args.data_dir, args.output, overwrite=args.overwrite)
    activated = sum(bool(query["matches"]) for query in result["queries"])
    print(f"Saved {len(result['queries'])} query match records to {args.output}; {activated} activated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

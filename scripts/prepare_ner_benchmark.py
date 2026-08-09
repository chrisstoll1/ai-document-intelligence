from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from docintel.ner_evaluation import ENTITY_LABELS, validate_annotations

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_MANIFEST = ROOT / "evaluation" / "tat_dqa" / "development_manifest.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "ner" / "development_manifest.json"
SELECTION_SEED = "docintel-ner-development-v1"
PASSAGES_PER_DOCUMENT = 2
MINIMUM_CHARACTERS = 40
MAXIMUM_CHARACTERS = 1500
MINIMUM_WORDS = 6
ENTITY_LIKELY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,})){1,}\b"
)


@dataclass(frozen=True)
class CandidatePassage:
    document_uid: str
    source_json: str
    source_json_sha256: str
    page: int
    block_id: str
    text: str


@dataclass(frozen=True)
class SelectedPassage:
    candidate: CandidatePassage
    stratum: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_passage(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def stable_key(candidate: CandidatePassage) -> str:
    identity = f"{candidate.document_uid}\0{candidate.page}\0{candidate.block_id}\0{candidate.text}"
    return sha256_bytes(f"{SELECTION_SEED}\0{identity}".encode())


def collect_candidates(document: dict) -> list[CandidatePassage]:
    source_json_path = (ROOT / document["pdf_file"]).with_suffix(".json")
    source_bytes = source_json_path.read_bytes()
    converted = json.loads(source_bytes)
    candidates = []
    for page_number, page in enumerate(converted["pages"], start=1):
        for block in page["blocks"]:
            text = normalize_passage(str(block["text"]))
            if not _eligible(text):
                continue
            candidates.append(
                CandidatePassage(
                    document_uid=document["uid"],
                    source_json=source_json_path.relative_to(ROOT).as_posix(),
                    source_json_sha256=sha256_bytes(source_bytes),
                    page=page_number,
                    block_id=str(block["uuid"]),
                    text=text,
                )
            )
    return candidates


def _eligible(text: str) -> bool:
    return (
        MINIMUM_CHARACTERS <= len(text) <= MAXIMUM_CHARACTERS
        and len(text.split()) >= MINIMUM_WORDS
    )


def select_candidates(
    candidates: list[CandidatePassage],
    *,
    count_per_document: int = PASSAGES_PER_DOCUMENT,
) -> list[SelectedPassage]:
    if count_per_document != 2:
        raise ValueError("NER selection currently requires one challenge and one general passage per document")
    by_document: dict[str, list[CandidatePassage]] = {}
    for candidate in candidates:
        by_document.setdefault(candidate.document_uid, []).append(candidate)

    selected = []
    for document_uid in sorted(by_document):
        document_candidates = sorted(by_document[document_uid], key=stable_key)
        if len(document_candidates) < count_per_document:
            raise RuntimeError(f"Document {document_uid} has too few eligible NER passages")
        challenge_candidates = [item for item in document_candidates if ENTITY_LIKELY_PATTERN.search(item.text)]
        if not challenge_candidates:
            raise RuntimeError(f"Document {document_uid} has no proper-name challenge passage")
        challenge = challenge_candidates[0]
        general = next(item for item in document_candidates if item != challenge)
        selected.extend((SelectedPassage(challenge, "proper_name_challenge"), SelectedPassage(general, "general")))
    return selected


def prepare(source_manifest_path: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing annotations: {output_path}")
    source_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_bytes)
    if source_manifest.get("split") != "development":
        raise RuntimeError("NER benchmark preparation accepts only the development split")

    candidates = [candidate for document in source_manifest["documents"] for candidate in collect_candidates(document)]
    selected = select_candidates(candidates)
    passages = []
    for item in selected:
        candidate = item.candidate
        passages.append(
            {
                "id": stable_key(candidate)[:16],
                "document_uid": candidate.document_uid,
                "source_json": candidate.source_json,
                "source_json_sha256": candidate.source_json_sha256,
                "page": candidate.page,
                "block_id": candidate.block_id,
                "text_sha256": sha256_bytes(candidate.text.encode()),
                "text": candidate.text,
                "stratum": item.stratum,
                "annotation_status": "pending",
                "entities": [],
            }
        )

    manifest = {
        "schema_version": 1,
        "name": "TAT-DQA NER development benchmark",
        "split": "development",
        "source_dataset": "TAT-DQA",
        "source_dataset_url": source_manifest["dataset_url"],
        "license": source_manifest["license"],
        "source_manifest": source_manifest_path.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": sha256_bytes(source_bytes),
        "labels": list(ENTITY_LABELS),
        "selection": {
            "seed": SELECTION_SEED,
            "method": (
                "SHA-256 ordering of eligible official converted blocks within a proper-name challenge stratum "
                "and a general stratum for each development document"
            ),
            "strata": {
                "proper_name_challenge": "Contains a model-independent sequence of two capitalized words or acronyms.",
                "general": "No entity-likelihood requirement; excludes the selected challenge passage.",
            },
            "passages_per_document": PASSAGES_PER_DOCUMENT,
            "document_count": len(source_manifest["documents"]),
            "passage_count": len(passages),
            "minimum_characters": MINIMUM_CHARACTERS,
            "maximum_characters": MAXIMUM_CHARACTERS,
            "minimum_words": MINIMUM_WORDS,
        },
        "annotation": {
            "status": "pending_review",
            "candidate_outputs_viewed": False,
            "normalization": "Unicode NFKC and whitespace collapsing before annotation",
            "guidelines": {
                "PERSON": "A specifically named person; exclude titles and unnamed roles.",
                "ORGANIZATION": (
                    "A named company, institution, agency, or formal group; include a legal suffix when present."
                ),
                "LOCATION": (
                    "A named geopolitical area, geographic place, or facility; include the complete proper name."
                ),
                "boundaries": "Use exact, non-overlapping character spans and annotate every repeated mention.",
            },
        },
        "passages": passages,
    }
    validate_annotations(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the candidate-blind TAT-DQA NER annotation set.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing annotation file.")
    args = parser.parse_args()
    manifest = prepare(args.source_manifest, args.output, overwrite=args.overwrite)
    print(f"Prepared {len(manifest['passages'])} NER passages at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

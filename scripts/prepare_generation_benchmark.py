from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL_MANIFEST = ROOT / "evaluation" / "tat_dqa" / "development_manifest.json"
DEFAULT_RAW_GOLD = ROOT / "data" / "evaluation" / "tat-dqa" / "tatdqa_dataset_dev.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "generation" / "development_manifest.json"
SELECTION_SEED = "docintel-generation-development-v1"
ANSWERABLE_QUOTAS = {"span": 5, "arithmetic": 5, "multi-span": 2}

UNANSWERABLE_QUESTIONS = (
    ("unanswerable-rainfall", "What was the average rainfall in Tokyo during 2019?"),
    ("unanswerable-spacecraft", "Which spacecraft first landed on Mars?"),
    ("unanswerable-graduates", "How many students graduated from Oxford University in 2018?"),
    ("unanswerable-dosage", "What dosage was administered in the clinical trial?"),
    ("unanswerable-election", "Who won the 2019 federal election in Canada?"),
    ("unanswerable-electricity", "What was the building's annual electricity consumption?"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(query: dict) -> str:
    return sha256_bytes(f"{SELECTION_SEED}\0{query['answer_type']}\0{query['uid']}".encode())


def select_answerable_queries(queries: list[dict]) -> list[dict]:
    selected = []
    for answer_type, count in ANSWERABLE_QUOTAS.items():
        candidates = sorted(
            (query for query in queries if query["answer_type"] == answer_type),
            key=stable_key,
        )
        if len(candidates) < count:
            raise RuntimeError(f"Not enough {answer_type} development questions")
        selected.extend(candidates[:count])
    return sorted(selected, key=lambda query: query["uid"])


def index_gold_questions(records: list[dict]) -> dict[str, dict]:
    return {question["uid"]: question for record in records for question in record["questions"]}


def prepare(
    retrieval_manifest_path: Path,
    raw_gold_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    retrieval_manifest_path = retrieval_manifest_path.resolve()
    raw_gold_path = raw_gold_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing generation benchmark: {output_path}")
    retrieval_bytes = retrieval_manifest_path.read_bytes()
    retrieval = json.loads(retrieval_bytes)
    split = retrieval.get("split")
    if split not in {"development", "locked_test"}:
        raise RuntimeError("Generation benchmark requires a development or locked-test retrieval split")
    raw_bytes = raw_gold_path.read_bytes()
    gold_by_uid = index_gold_questions(json.loads(raw_bytes))
    documents = {document["uid"]: document for document in retrieval["documents"]}

    questions = []
    selected_queries = (
        select_answerable_queries(retrieval["queries"]) if split == "development" else retrieval["queries"]
    )
    for query in selected_queries:
        gold = gold_by_uid[query["uid"]]
        if gold["question"] != query["question"]:
            raise RuntimeError(f"Generation gold mismatch for {query['uid']}")
        document = documents[query["document_uid"]]
        questions.append(
            {
                "uid": query["uid"],
                "expected_status": "answered",
                "question": query["question"],
                "answer_type": query["answer_type"],
                "answer": gold["answer"],
                "scale": gold["scale"],
                "derivation": gold["derivation"],
                "document_uid": query["document_uid"],
                "document_sha256": document["pdf_sha256"],
                "relevant_pages": query["relevant_pages"],
                "evidence": query["evidence"],
            }
        )
    questions.extend(
        {
            "uid": uid,
            "expected_status": "insufficient_evidence",
            "question": question,
            "answer_type": "unanswerable",
            "answer": None,
            "scale": "",
            "derivation": "",
            "document_uid": None,
            "document_sha256": None,
            "relevant_pages": [],
            "evidence": [],
        }
        for uid, question in UNANSWERABLE_QUESTIONS
    )

    if split == "development":
        selection = {
            "seed": SELECTION_SEED,
            "method": "SHA-256 ordering within official answer types plus fixed candidate-blind unanswerable questions",
            "answerable_quotas": ANSWERABLE_QUOTAS,
            "answerable_count": sum(ANSWERABLE_QUOTAS.values()),
            "unanswerable_count": len(UNANSWERABLE_QUESTIONS),
        }
    else:
        selection = {
            "method": "All previously locked retrieval questions plus fixed candidate-blind unanswerable questions",
            "answerable_count": len(selected_queries),
            "unanswerable_count": len(UNANSWERABLE_QUESTIONS),
        }
    manifest = {
        "schema_version": 1,
        "name": f"TAT-DQA grounded generation {split.replace('_', ' ')} benchmark",
        "split": split,
        "source_dataset": retrieval["dataset"],
        "source_dataset_url": retrieval["dataset_url"],
        "license": retrieval["license"],
        "retrieval_manifest": retrieval_manifest_path.relative_to(ROOT).as_posix(),
        "retrieval_manifest_sha256": sha256_bytes(retrieval_bytes),
        f"raw_{split}_gold_sha256": sha256_bytes(raw_bytes),
        "selection": selection,
        "questions": questions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the grounded-generation development benchmark.")
    parser.add_argument("--retrieval-manifest", type=Path, default=DEFAULT_RETRIEVAL_MANIFEST)
    parser.add_argument("--raw-gold", type=Path, default=DEFAULT_RAW_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.retrieval_manifest, args.raw_gold, args.output, overwrite=args.overwrite)
    print(f"Prepared {len(manifest['questions'])} generation questions at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

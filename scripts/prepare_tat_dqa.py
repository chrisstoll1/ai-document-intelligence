from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "evaluation" / "tat-dqa"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "tat_dqa"
DATASET_URL = "https://nextplusplus.github.io/TAT-DQA/"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2"
SELECTION_SEED = "docintel-tat-dqa-v1"


@dataclass(frozen=True)
class SourceFile:
    name: str
    drive_id: str
    size_bytes: int
    sha256: str


SOURCE_FILES = (
    SourceFile(
        "tatdqa_dataset_dev.json",
        "1dmXKledPa6ptXuFibUCtoJMEP6oCD60N",
        1_210_555,
        "07395642f317754ed22bd8330525af7c1e15b7189efdf046cd0a257fc3178fcb",
    ),
    SourceFile(
        "tatdqa_dataset_test.json",
        "1Dqvlu83R4t5odayhtt65qwFkxHWmieOr",
        351_652,
        "92e20a746efc203936fa38d76e3f3762df4f7a15b815bb6c247a5a1049d27982",
    ),
    SourceFile(
        "tatdqa_dataset_test_gold.json",
        "1ZQjjIC0BB14l6t9b1Ryq0t-CNAP6iC2J",
        1_229_595,
        "e6758315b839203a72e05f8849ce4d56fc8d6d7c34dc0240352769014807f5da",
    ),
    SourceFile(
        "tatdqa_docs_dev.zip",
        "1M1CtHAS4SzFFnNVsFVGgpaGq84mEzPcR",
        155_773_433,
        "0e463045f3f89dcfd0e7238480fc931bd5f301883d9a022d4bd54d3f9c3ee080",
    ),
    SourceFile(
        "tatdqa_docs_test.zip",
        "1iqe5r-qgQZLhGtM4G6LkNp9S6OCwOF2L",
        173_931_848,
        "83c3baf38102fe5b17041efe41ea770912e6afa32eb603fece5337318259ffc8",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(raw_dir: Path) -> None:
    import gdown

    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCE_FILES:
        destination = raw_dir / source.name
        if destination.is_file() and sha256_file(destination) == source.sha256:
            continue
        destination.unlink(missing_ok=True)
        result = gdown.download(id=source.drive_id, output=str(destination), quiet=False)
        if result is None:
            raise RuntimeError(f"Could not download {source.name}")


def verify_sources(raw_dir: Path) -> None:
    for source in SOURCE_FILES:
        path = raw_dir / source.name
        if not path.is_file():
            raise RuntimeError(f"Missing TAT-DQA source file: {path}")
        if path.stat().st_size != source.size_bytes:
            raise RuntimeError(f"Unexpected size for {source.name}")
        if sha256_file(path) != source.sha256:
            raise RuntimeError(f"Checksum mismatch for {source.name}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise RuntimeError(f"Corrupt ZIP member in {source.name}: {corrupt_member}")


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\0{namespace}\0{value}".encode()).hexdigest()


def select_documents(
    records: list[dict[str, Any]],
    page_counts: dict[str, int],
    *,
    split: str,
    count: int,
    multipage_count: int,
) -> list[dict[str, Any]]:
    multipage = [record for record in records if page_counts[record["doc"]["uid"]] > 1]
    single_page = [record for record in records if page_counts[record["doc"]["uid"]] == 1]
    multipage.sort(key=lambda record: stable_key(f"{split}-document", record["doc"]["uid"]))
    single_page.sort(key=lambda record: stable_key(f"{split}-document", record["doc"]["uid"]))
    if len(multipage) < multipage_count or len(single_page) < count - multipage_count:
        raise RuntimeError(f"Not enough documents to build the {split} subset")
    selected = multipage[:multipage_count] + single_page[: count - multipage_count]
    return sorted(selected, key=lambda record: record["doc"]["uid"])


def select_questions(record: dict[str, Any], *, split: str, count: int = 2) -> list[dict[str, Any]]:
    questions = sorted(
        record["questions"],
        key=lambda question: stable_key(f"{split}-question", question["uid"]),
    )
    if len(questions) < count:
        raise RuntimeError(f"Document {record['doc']['uid']} has fewer than {count} questions")
    return questions[:count]


def index_blocks(converted: dict[str, Any]) -> dict[str, tuple[int, dict[str, Any]]]:
    return {
        block["uuid"]: (page_number, block)
        for page_number, page in enumerate(converted["pages"], start=1)
        for block in page["blocks"]
    }


def build_evidence(question: dict[str, Any], converted: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = index_blocks(converted)
    evidence = []
    for mapping in question.get("block_mapping", []):
        for block_id, span in mapping.items():
            page, block = blocks[block_id]
            start, end = (int(value) for value in span)
            evidence.append(
                {
                    "block_id": block_id,
                    "page": page,
                    "char_start": start,
                    "char_end": end,
                    "text": block["text"][start:end],
                    "block_text": block["text"],
                }
            )

    if not evidence:
        for fact in question.get("facts", []):
            fact_text = str(fact).strip()
            for block_id, (page, block) in blocks.items():
                start = block["text"].casefold().find(fact_text.casefold())
                if start >= 0:
                    evidence.append(
                        {
                            "block_id": block_id,
                            "page": page,
                            "char_start": start,
                            "char_end": start + len(fact_text),
                            "text": block["text"][start : start + len(fact_text)],
                            "block_text": block["text"],
                        }
                    )
                    break

    unique = {}
    for item in evidence:
        key = (item["block_id"], item["char_start"], item["char_end"])
        unique[key] = item
    result = list(unique.values())
    if not result:
        raise RuntimeError(f"Question {question['uid']} has no resolvable evidence")
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_converted(archive: zipfile.ZipFile, split: str, uid: str) -> dict[str, Any]:
    return json.loads(archive.read(f"{split}/{uid}.json"))


def page_counts(archive: zipfile.ZipFile, split: str, records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        record["doc"]["uid"]: len(load_converted(archive, split, record["doc"]["uid"])["pages"])
        for record in records
    }


def extract_document(archive: zipfile.ZipFile, split: str, uid: str, destination: Path) -> tuple[bytes, dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    pdf = archive.read(f"{split}/{uid}.pdf")
    converted_bytes = archive.read(f"{split}/{uid}.json")
    (destination / f"{uid}.pdf").write_bytes(pdf)
    (destination / f"{uid}.json").write_bytes(converted_bytes)
    return pdf, json.loads(converted_bytes)


def prepare_split(
    raw_dir: Path,
    output_dir: Path,
    *,
    source_split: str,
    manifest_name: str,
    public_annotations: str,
    gold_annotations: str,
    document_count: int,
    multipage_count: int,
) -> dict[str, Any]:
    public_records = load_json(raw_dir / public_annotations)
    gold_records = load_json(raw_dir / gold_annotations)
    gold_by_document = {record["doc"]["uid"]: record for record in gold_records}
    archive_path = raw_dir / f"tatdqa_docs_{source_split}.zip"

    with zipfile.ZipFile(archive_path) as archive:
        counts = page_counts(archive, source_split, public_records)
        selected = select_documents(
            public_records,
            counts,
            split=manifest_name,
            count=document_count,
            multipage_count=multipage_count,
        )
        documents = []
        queries = []
        subset_dir = raw_dir / "subset" / source_split
        for public_record in selected:
            uid = public_record["doc"]["uid"]
            gold_record = gold_by_document[uid]
            gold_questions = {question["uid"]: question for question in gold_record["questions"]}
            pdf, converted = extract_document(archive, source_split, uid, subset_dir)
            documents.append(
                {
                    "uid": uid,
                    "source": public_record["doc"]["source"],
                    "source_page": public_record["doc"]["page"],
                    "page_count": counts[uid],
                    "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
                    "pdf_file": f"data/evaluation/tat-dqa/subset/{source_split}/{uid}.pdf",
                }
            )
            for public_question in select_questions(public_record, split=manifest_name):
                gold_question = gold_questions[public_question["uid"]]
                if public_question["question"] != gold_question["question"]:
                    raise RuntimeError(f"Public and gold question mismatch for {public_question['uid']}")
                evidence = build_evidence(gold_question, converted)
                queries.append(
                    {
                        "uid": public_question["uid"],
                        "document_uid": uid,
                        "question": public_question["question"],
                        "answer_type": gold_question["answer_type"],
                        "scale": gold_question["scale"],
                        "relevant_pages": sorted({item["page"] for item in evidence}),
                        "evidence": evidence,
                    }
                )

    manifest = {
        "schema_version": 1,
        "dataset": "TAT-DQA",
        "dataset_url": DATASET_URL,
        "license": "CC BY 4.0",
        "split": manifest_name,
        "source_split": source_split,
        "selection": {
            "seed": SELECTION_SEED,
            "method": "SHA-256 ordering over public IDs with a fixed multipage quota",
            "document_count": document_count,
            "multipage_document_count": multipage_count,
            "questions_per_document": 2,
        },
        "documents": documents,
        "queries": queries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{manifest_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare(raw_dir: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    development = prepare_split(
        raw_dir,
        output_dir,
        source_split="dev",
        manifest_name="development",
        public_annotations="tatdqa_dataset_dev.json",
        gold_annotations="tatdqa_dataset_dev.json",
        document_count=15,
        multipage_count=3,
    )
    locked_test = prepare_split(
        raw_dir,
        output_dir,
        source_split="test",
        manifest_name="locked_test",
        public_annotations="tatdqa_dataset_test.json",
        gold_annotations="tatdqa_dataset_test_gold.json",
        document_count=10,
        multipage_count=2,
    )
    return development, locked_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Download, verify, and prepare the TAT-DQA evaluation subset.")
    parser.add_argument("--download", action="store_true", help="Download missing or invalid official source files.")
    parser.add_argument("--verify-only", action="store_true", help="Verify official files without preparing manifests.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.download:
        download_sources(args.raw_dir)
    verify_sources(args.raw_dir)
    if args.verify_only:
        print(f"Verified {len(SOURCE_FILES)} official TAT-DQA files.")
        return 0

    development, locked_test = prepare(args.raw_dir, args.output_dir)
    print(
        "Prepared TAT-DQA subsets: "
        f"development={len(development['documents'])} documents/{len(development['queries'])} queries, "
        f"locked_test={len(locked_test['documents'])} documents/{len(locked_test['queries'])} queries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

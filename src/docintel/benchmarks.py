from __future__ import annotations

import json
import math
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import Document, SearchResult
from .pipeline import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = ROOT / "data" / "benchmarks"

BEIR_DATASETS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}


@dataclass(frozen=True)
class BenchmarkCase:
    query_id: str
    query: str
    relevant_documents: dict[str, int]


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    documents: list[Document]
    cases: list[BenchmarkCase]


def load_beir_dataset(name: str = "scifact", data_dir: Path = DEFAULT_BENCHMARK_DIR) -> BenchmarkDataset:
    if name not in BEIR_DATASETS:
        supported = ", ".join(sorted(BEIR_DATASETS))
        raise ValueError(f"Unsupported benchmark dataset '{name}'. Supported datasets: {supported}")

    dataset_dir = _ensure_beir_dataset(name, data_dir)
    documents = _load_corpus(dataset_dir, name)
    queries = _load_queries(dataset_dir)
    qrels = _load_qrels(dataset_dir)
    cases = [
        BenchmarkCase(query_id=query_id, query=query, relevant_documents=qrels[query_id])
        for query_id, query in queries.items()
        if query_id in qrels
    ]
    return BenchmarkDataset(name=name, documents=documents, cases=cases)


def run_benchmark(dataset: BenchmarkDataset, modes: list[str], limit: int = 10, query_limit: Optional[int] = None) -> dict[str, dict[str, float]]:
    pipeline = RetrievalPipeline.from_documents(dataset.documents, max_words=220)
    cases = dataset.cases[:query_limit] if query_limit else dataset.cases
    output: dict[str, dict[str, float]] = {}

    for mode in modes:
        rankings: dict[str, list[str]] = {}
        qrels: dict[str, dict[str, int]] = {}
        for case in cases:
            results = pipeline.search(case.query, mode=mode, limit=limit * 3)
            rankings[case.query_id] = _dedupe_doc_ids(results)[:limit]
            qrels[case.query_id] = case.relevant_documents
        output[mode] = evaluate_rankings(qrels, rankings, k=limit)

    output["_meta"] = {
        "documents": float(len(dataset.documents)),
        "queries": float(len(cases)),
        "k": float(limit),
        "chunks": float(len(pipeline.chunks)),
    }
    return output


def evaluate_rankings(qrels: dict[str, dict[str, int]], rankings: dict[str, list[str]], k: int = 10) -> dict[str, float]:
    if not qrels:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}

    recall = precision = mrr = ndcg = 0.0
    for query_id, relevant in qrels.items():
        ranked = rankings.get(query_id, [])[:k]
        relevant_ids = {doc_id for doc_id, score in relevant.items() if score > 0}
        hits = [doc_id for doc_id in ranked if doc_id in relevant_ids]
        recall += len(hits) / max(1, len(relevant_ids))
        precision += len(hits) / max(1, k)
        mrr += _reciprocal_rank(ranked, relevant_ids)
        ndcg += _ndcg(ranked, relevant, k)

    count = len(qrels)
    return {
        "recall": recall / count,
        "precision": precision / count,
        "mrr": mrr / count,
        "ndcg": ndcg / count,
    }


def _ensure_beir_dataset(name: str, data_dir: Path) -> Path:
    dataset_dir = data_dir / name
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        _download_file(BEIR_DATASETS[name], zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    return dataset_dir


def _download_file(url: str, target: Path) -> None:
    try:
        urllib.request.urlretrieve(url, target)
        return
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise

    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response:
        target.write_bytes(response.read())


def _load_corpus(dataset_dir: Path, dataset_name: str) -> list[Document]:
    documents = []
    with (dataset_dir / "corpus.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row.get("_id", ""))
            title = row.get("title") or doc_id
            text = row.get("text") or ""
            full_text = f"{title}\n\n{text}" if title else text
            documents.append(
                Document(
                    id=doc_id,
                    title=title,
                    text=full_text,
                    source_path=f"beir/{dataset_name}/{doc_id}",
                )
            )
    return documents


def _load_queries(dataset_dir: Path) -> dict[str, str]:
    queries = {}
    with (dataset_dir / "queries.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def _load_qrels(dataset_dir: Path) -> dict[str, dict[str, int]]:
    qrels_dir = dataset_dir / "qrels"
    qrel_path = qrels_dir / "test.tsv"
    if not qrel_path.exists():
        qrel_path = qrels_dir / "dev.tsv"
    if not qrel_path.exists():
        qrel_path = next(qrels_dir.glob("*.tsv"))

    qrels: dict[str, dict[str, int]] = {}
    with qrel_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().lower().split("\t")
        has_header = "query-id" in header or "corpus-id" in header or "score" in header
        if not has_header:
            handle.seek(0)
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def _dedupe_doc_ids(results: list[SearchResult]) -> list[str]:
    seen = set()
    doc_ids = []
    for result in results:
        doc_id = result.chunk.document_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def _reciprocal_rank(ranked: list[str], relevant_ids: set[str]) -> float:
    for index, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def _ndcg(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked[:k], start=1):
        rel = relevant.get(doc_id, 0)
        dcg += rel / math.log2(index + 1)

    ideal_rels = sorted((score for score in relevant.values() if score > 0), reverse=True)[:k]
    idcg = sum(rel / math.log2(index + 1) for index, rel in enumerate(ideal_rels, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg

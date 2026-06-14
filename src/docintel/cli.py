from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmarks import DEFAULT_BENCHMARK_DIR, load_beir_dataset, run_benchmark
from .pipeline import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "data" / "sample_docs"
DEFAULT_QUERIES = ROOT / "data" / "queries" / "test_queries.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidence-based document retrieval prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the sample document collection")
    search_parser.add_argument("query")
    search_parser.add_argument("--mode", choices=["keyword", "vector", "combined", "both"], default="combined")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)

    eval_parser = subparsers.add_parser("evaluate", help="Run known-answer evaluation queries")
    eval_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    eval_parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    eval_parser.add_argument("--mode", choices=["keyword", "vector", "combined", "both", "all"], default="all")
    eval_parser.add_argument("--limit", type=int, default=3)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a public BEIR benchmark evaluation")
    benchmark_parser.add_argument("--dataset", choices=["scifact"], default="scifact")
    benchmark_parser.add_argument("--data-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    benchmark_parser.add_argument("--mode", choices=["keyword", "vector", "combined", "all"], default="all")
    benchmark_parser.add_argument("--limit", type=int, default=10)
    benchmark_parser.add_argument("--query-limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "search":
        run_search(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "benchmark":
        run_public_benchmark(args)


def run_search(args: argparse.Namespace) -> None:
    pipeline = RetrievalPipeline(args.docs)
    results = pipeline.search(args.query, mode=args.mode, limit=args.limit)
    print(f"Query: {args.query}")
    print(
        f"Documents: {len(pipeline.documents)} | Chunks: {len(pipeline.chunks)} | "
        f"Mode: {args.mode} | Vector backend: {pipeline.vector_index.backend}"
    )
    print()
    for index, result in enumerate(results, start=1):
        print(f"{index}. [{result.method}] {result.chunk.document_title} score={result.score:.3f}")
        if result.components:
            components = ", ".join(f"{key}={value:.2f}" for key, value in result.components.items())
            print(f"   components: {components}")
        if result.chunk.metadata:
            metadata = ", ".join(f"{key}={value}" for key, value in result.chunk.metadata.items() if value)
            print(f"   metadata: {metadata}")
        print(f"   {result.chunk.text[:350]}...")
        print(f"   Source: {result.chunk.source_path}")
        print()


def run_evaluate(args: argparse.Namespace) -> None:
    pipeline = RetrievalPipeline(args.docs)
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    modes = ["keyword", "vector", "combined"] if args.mode == "all" else [args.mode]

    for mode in modes:
        hits = 0
        print(f"Mode: {mode} | Vector backend: {pipeline.vector_index.backend}")
        print("| Query | Expected | Returned | Result |")
        print("|---|---|---|---|")
        for case in cases:
            results = pipeline.search(case["query"], mode=mode, limit=args.limit)
            expected = set(case["expected_documents"])
            returned = {result.chunk.document_id for result in results}
            success = bool(expected & returned)
            hits += int(success)
            status = "PASS" if success else "FAIL"
            print(
                f"| {case['query']} | {', '.join(sorted(expected))} | "
                f"{', '.join(sorted(returned)) or 'none'} | {status} |"
            )

        total = max(1, len(cases))
        print(f"Top-{args.limit} success: {hits}/{len(cases)} ({hits / total:.0%})")
        top1_hits = 0
        for case in cases:
            results = pipeline.search(case["query"], mode=mode, limit=1)
            expected = set(case["expected_documents"])
            returned = {result.chunk.document_id for result in results}
            top1_hits += int(bool(expected & returned))
        print(f"Top-1 success: {top1_hits}/{len(cases)} ({top1_hits / total:.0%})")
        print()


def run_public_benchmark(args: argparse.Namespace) -> None:
    dataset = load_beir_dataset(args.dataset, args.data_dir)
    modes = ["keyword", "vector", "combined"] if args.mode == "all" else [args.mode]
    results = run_benchmark(dataset, modes=modes, limit=args.limit, query_limit=args.query_limit)
    meta = results.pop("_meta")

    print(f"Dataset: BEIR/{dataset.name}")
    print(
        f"Documents: {int(meta['documents'])} | Chunks: {int(meta['chunks'])} | "
        f"Queries: {int(meta['queries'])} | k={int(meta['k'])}"
    )
    print("| Mode | Recall@k | Precision@k | MRR@k | nDCG@k |")
    print("|---|---:|---:|---:|---:|")
    for mode, metrics in results.items():
        print(
            f"| {mode} | {metrics['recall']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['ndcg']:.3f} |"
        )


if __name__ == "__main__":
    main()

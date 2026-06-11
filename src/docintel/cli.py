from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RetrievalPipeline


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "data" / "sample_docs"
DEFAULT_QUERIES = ROOT / "data" / "queries" / "test_queries.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidence-based document retrieval prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the sample document collection")
    search_parser.add_argument("query")
    search_parser.add_argument("--mode", choices=["keyword", "vector", "both"], default="both")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)

    eval_parser = subparsers.add_parser("evaluate", help="Run known-answer evaluation queries")
    eval_parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    eval_parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    eval_parser.add_argument("--mode", choices=["keyword", "vector", "both"], default="both")
    eval_parser.add_argument("--limit", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "search":
        run_search(args)
    elif args.command == "evaluate":
        run_evaluate(args)


def run_search(args: argparse.Namespace) -> None:
    pipeline = RetrievalPipeline(args.docs)
    results = pipeline.search(args.query, mode=args.mode, limit=args.limit)
    print(f"Query: {args.query}")
    print(f"Documents: {len(pipeline.documents)} | Chunks: {len(pipeline.chunks)} | Mode: {args.mode}")
    print()
    for index, result in enumerate(results, start=1):
        print(f"{index}. [{result.method}] {result.chunk.document_title} score={result.score:.3f}")
        print(f"   {result.chunk.text[:350]}...")
        print(f"   Source: {result.chunk.source_path}")
        print()


def run_evaluate(args: argparse.Namespace) -> None:
    pipeline = RetrievalPipeline(args.docs)
    cases = json.loads(args.queries.read_text(encoding="utf-8"))
    hits = 0

    for case in cases:
        results = pipeline.search(case["query"], mode=args.mode, limit=args.limit)
        expected = set(case["expected_documents"])
        returned = {result.chunk.document_id for result in results}
        success = bool(expected & returned)
        hits += int(success)
        status = "PASS" if success else "FAIL"
        print(f"{status}: {case['query']}")
        print(f"  expected: {', '.join(sorted(expected))}")
        print(f"  returned: {', '.join(sorted(returned)) or 'none'}")

    total = max(1, len(cases))
    print()
    print(f"Top-{args.limit} success: {hits}/{len(cases)} ({hits / total:.0%})")


if __name__ == "__main__":
    main()

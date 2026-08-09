from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from docintel.metadata_reranking import aggregate_query_metrics, metrics_for_ranking, stable_page_match_rerank

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHES = ROOT / "evaluation" / "metadata" / "development_matches.json"
DEFAULT_BASELINE = ROOT / "evaluation" / "results" / "tat_dqa_development_selected.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "tat_dqa_development_metadata_rerank.json"


def evaluate(matches_path: Path, baseline_path: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing metadata evaluation: {output_path}")
    match_bytes = matches_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()
    matches = json.loads(match_bytes)
    baseline = json.loads(baseline_bytes)
    if matches["manifest_sha256"] != baseline["manifest_sha256"]:
        raise RuntimeError("Metadata matches and retrieval baseline use different manifests")
    match_by_uid = {query["uid"]: query for query in matches["queries"]}
    results = []
    for query in baseline["queries"]:
        match = match_by_uid[query["query_id"]]
        original = query["modes"]["hybrid"]["ranking"]
        reranked = stable_page_match_rerank(original, match["matches"])
        judgment = query["judgment"]
        control = metrics_for_ranking(
            original,
            relevant_count=judgment["relevant_chunk_count"],
            evidence_count=judgment["evidence_chunk_count"],
        )
        treatment = metrics_for_ranking(
            reranked,
            relevant_count=judgment["relevant_chunk_count"],
            evidence_count=judgment["evidence_chunk_count"],
        )
        results.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "entities": match["entities"],
                "matches": match["matches"],
                "activated": [item["chunk_id"] for item in original] != [item["chunk_id"] for item in reranked],
                "control_metrics": control,
                "metadata_metrics": treatment,
                "control_ranking": [item["chunk_id"] for item in original],
                "metadata_ranking": [item["chunk_id"] for item in reranked],
            }
        )
    control = aggregate_query_metrics(results, "control_metrics")
    treatment = aggregate_query_metrics(results, "metadata_metrics")
    for name, value in control.items():
        expected = baseline["metrics"]["hybrid"][name]
        if abs(value - expected) > 1e-12:
            raise RuntimeError(f"Frozen hybrid baseline did not reproduce for {name}")
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "split": "development",
        "matches_sha256": hashlib.sha256(match_bytes).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "policy": matches["policy"],
        "query_coverage": {
            "queries": len(results),
            "with_entities": sum(bool(result["entities"]) for result in results),
            "with_global_matches": sum(bool(result["matches"]) for result in results),
            "rankings_changed": sum(result["activated"] for result in results),
        },
        "metrics": {
            "control": control,
            "metadata_rerank": treatment,
            "delta": {name: treatment[name] - control[name] for name in control},
        },
        "queries": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exact metadata reranking over frozen hybrid rankings.")
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.matches, args.baseline, args.output, overwrite=args.overwrite)
    print(json.dumps({"coverage": result["query_coverage"], "metrics": result["metrics"]}, indent=2))
    print(f"Saved metadata reranking results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

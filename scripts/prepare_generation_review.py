from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "evaluation" / "generation" / "development_inputs.json"
DEFAULT_RESULTS = (
    ROOT / "evaluation" / "results" / "generation_development_qwen.json",
    ROOT / "evaluation" / "results" / "generation_development_mistral.json",
)
DEFAULT_OUTPUT = ROOT / "evaluation" / "generation" / "development_review.json"
BLINDING_SEED = "docintel-generation-review-v1"


def candidate_aliases(results: list[dict]) -> dict[str, str]:
    ordered = sorted(
        (result["model"]["key"] for result in results),
        key=lambda key: hashlib.sha256(f"{BLINDING_SEED}\0{key}".encode()).hexdigest(),
    )
    return {key: f"Candidate {chr(65 + index)}" for index, key in enumerate(ordered)}


def prepare(inputs_path: Path, result_paths: list[Path], output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite generation review: {output_path}")
    input_bytes = inputs_path.read_bytes()
    inputs = json.loads(input_bytes)
    input_by_uid = {question["uid"]: question for question in inputs["questions"]}
    results = [json.loads(path.read_bytes()) for path in result_paths]
    if any(result["inputs_sha256"] != hashlib.sha256(input_bytes).hexdigest() for result in results):
        raise RuntimeError("Generation results do not match review inputs")
    aliases = candidate_aliases(results)
    items = []
    for result in results:
        alias = aliases[result["model"]["key"]]
        for question in result["questions"]:
            source = input_by_uid[question["uid"]]
            context_by_id = {
                f"C{index}": context for index, context in enumerate(source["contexts"], start=1)
            }
            claims = []
            for claim in question["claims"]:
                claims.append(
                    {
                        **claim,
                        "cited_contexts": [context_by_id[citation_id] for citation_id in claim["citation_ids"]],
                        "support_rating": None,
                        "support_notes": "",
                    }
                )
            review_id = hashlib.sha256(f"{BLINDING_SEED}\0{alias}\0{question['uid']}".encode()).hexdigest()[:16]
            items.append(
                {
                    "review_id": review_id,
                    "candidate": alias,
                    "question_id": question["uid"],
                    "question": question["question"],
                    "status": question["status"],
                    "answer": question["answer"],
                    "claims": claims,
                    "all_contexts": source["contexts"],
                    "answer_relevance": None,
                    "citation_completeness": None,
                    "review_notes": "",
                }
            )
    items.sort(key=lambda item: item["review_id"])
    review = {
        "schema_version": 1,
        "status": "awaiting_human_review",
        "blinding_seed": BLINDING_SEED,
        "rubric": {
            "support_rating": ["supported", "partially_supported", "unsupported"],
            "answer_relevance": ["direct", "partially_direct", "not_direct", "not_applicable_refusal"],
            "citation_completeness": ["complete", "incomplete", "not_applicable_refusal"],
            "instructions": (
                "Judge only against supplied contexts. Do not infer candidate identity or use outside knowledge."
            ),
        },
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(review, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return review


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a candidate-blind grounded-generation review file.")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--results", type=Path, nargs="+", default=list(DEFAULT_RESULTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    review = prepare(args.inputs, args.results, args.output, overwrite=args.overwrite)
    print(f"Prepared {len(review['items'])} blinded review items at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

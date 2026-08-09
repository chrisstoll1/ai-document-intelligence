from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import psutil
from docintel.config import DEFAULT_GENERATION_MAX_NEW_TOKENS, DEFAULT_GENERATION_MODEL, DEFAULT_GENERATION_REVISION
from docintel.generation import GroundedGenerationService
from docintel.generators import HuggingFaceStructuredGenerator
from docintel.search import PersistentSearchResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "evaluation" / "generation" / "development_inputs.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "generation_qwen_resource_profile.json"


class FixedSearch:
    def __init__(self, contexts: list[PersistentSearchResult]) -> None:
        self.contexts = contexts

    def search(self, query: str, *, limit: int = 5) -> list[PersistentSearchResult]:
        return self.contexts[:limit]


def profile(inputs_path: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite generation profile: {output_path}")
    inputs = json.loads(inputs_path.read_bytes())
    process = psutil.Process()
    rss_before = process.memory_info().rss
    generator = HuggingFaceStructuredGenerator(
        DEFAULT_GENERATION_MODEL,
        DEFAULT_GENERATION_REVISION,
        max_new_tokens=DEFAULT_GENERATION_MAX_NEW_TOKENS,
    )
    initialized = time.perf_counter()
    torch = generator.torch
    initialization_seconds = time.perf_counter() - initialized
    torch.cuda.reset_peak_memory_stats()
    loaded_allocated = torch.cuda.memory_allocated()
    loaded_reserved = torch.cuda.memory_reserved()
    rss_loaded = process.memory_info().rss
    latencies = []
    statuses = []
    for question in inputs["questions"]:
        contexts = [PersistentSearchResult(**context) for context in question["contexts"]]
        service = GroundedGenerationService(FixedSearch(contexts), generator)
        started = time.perf_counter()
        answer = service.answer(question["question"], limit=5)
        latencies.append((time.perf_counter() - started) * 1000)
        statuses.append(answer.status)
    profile_result = {
        "schema_version": 1,
        "model": generator.version,
        "device": torch.cuda.get_device_name(0),
        "questions": len(inputs["questions"]),
        "initialization_seconds": initialization_seconds,
        "latency_ms": {
            "mean": mean(latencies),
            "minimum": min(latencies),
            "maximum": max(latencies),
        },
        "memory_bytes": {
            "process_rss_before_load": rss_before,
            "process_rss_after_load": rss_loaded,
            "process_rss_increase": rss_loaded - rss_before,
            "gpu_allocated_after_load": loaded_allocated,
            "gpu_reserved_after_load": loaded_reserved,
            "gpu_peak_allocated_during_inference": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_during_inference": torch.cuda.max_memory_reserved(),
        },
        "statuses": {status: statuses.count(status) for status in sorted(set(statuses))},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile_result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return profile_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile selected grounded-generation resource use.")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = profile(args.inputs, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

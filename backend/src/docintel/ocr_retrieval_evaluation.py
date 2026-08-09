from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def reciprocal_rank_at(ranking: Sequence[dict], cutoff: int) -> float:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    for rank, candidate in enumerate(ranking[:cutoff], start=1):
        if int(candidate["relevance_grade"]) > 0:
            return 1.0 / rank
    return 0.0


def paired_retrieval_summary(clean: Sequence[dict], degraded: Sequence[dict], *, cutoff: int = 5) -> dict:
    clean_by_id = {query["query_id"]: query for query in clean}
    degraded_by_id = {query["query_id"]: query for query in degraded}
    if clean_by_id.keys() != degraded_by_id.keys():
        raise ValueError("Clean and degraded query sets must match")
    pairs = []
    for query_id in sorted(clean_by_id):
        clean_rr = reciprocal_rank_at(clean_by_id[query_id]["modes"]["hybrid"]["ranking"], cutoff)
        degraded_rr = reciprocal_rank_at(degraded_by_id[query_id]["modes"]["hybrid"]["ranking"], cutoff)
        pairs.append(
            {
                "query_id": query_id,
                "clean_rr": clean_rr,
                "degraded_rr": degraded_rr,
                "delta": degraded_rr - clean_rr,
            }
        )
    return {
        "cutoff": cutoff,
        "clean_mrr": mean(pair["clean_rr"] for pair in pairs),
        "degraded_mrr": mean(pair["degraded_rr"] for pair in pairs),
        "delta": mean(pair["delta"] for pair in pairs),
        "wins": sum(pair["delta"] > 0 for pair in pairs),
        "ties": sum(pair["delta"] == 0 for pair in pairs),
        "losses": sum(pair["delta"] < 0 for pair in pairs),
        "queries": pairs,
    }

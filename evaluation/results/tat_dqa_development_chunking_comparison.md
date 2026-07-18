# TAT-DQA Development Chunking Comparison

Run date: July 18, 2026

These are development results. The locked-test split was not scored. Every run used the selected MiniLM revision, equal keyword/semantic RRF weights, `k=60`, the same 15 documents, and the same 30 queries and judgments.

## Candidate Structure

| Maximum / overlap words | Chunks | Mean words | Cross-page chunks | Queries with exact evidence chunks |
|---|---:|---:|---:|---:|
| 80 / 15 | 119 | 67.8 | 3 | 29/30 |
| **120 / 20** | 79 | 102.2 | **2** | **29/30** |
| 180 / 30 | 54 | 149.5 | 3 | 28/30 |

The chunker preserves extracted blocks and only windows blocks that exceed the maximum. Adjacent shorter blocks may be combined up to the limit. Exact character spans link chunk text back to source blocks.

## Hybrid Results

| Maximum / overlap | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Evidence Hit@1 | Evidence Hit@5 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 / 15 | 0.700 | 0.833 | **0.867** | **0.867** | 0.517 | 0.793 | 0.769 | 0.538 | 7.22 ms |
| **120 / 20** | **0.767** | 0.833 | 0.833 | 0.833 | **0.690** | **0.828** | **0.794** | 0.625 | 7.35 ms |
| 180 / 30 | 0.700 | **0.867** | **0.867** | **0.867** | 0.679 | 0.821 | 0.772 | **0.662** | **7.00 ms** |

Evidence Hit is averaged only over queries for which the application's extracted chunks contain an exact official evidence token sequence. Ordinary Hit and MRR use the official document/page judgment.

Page-level Recall@k is retained in the machine-readable results but is not used to choose chunk size. Smaller chunks create more relevant chunks on the same judged page, changing the recall denominator and making direct cross-configuration comparison misleading.

## Selection

The 120/20 configuration is retained. It provides the strongest first-result success, MRR, exact-evidence Hit@1 and Hit@5, the fewest cross-page chunks, and exact evidence resolution for 29 of 30 queries. These properties match the evidence-first interface, where users inspect five passages and citations should resolve to focused source text.

The 180/30 candidate is a credible alternative: it produces the strongest nDCG and finds one additional relevant page by ranks three, five, and ten. It was not selected because its average 150-word chunks reduce citation granularity, its MRR is lower, and one additional query loses exact evidence-token resolution. The 80/15 candidate finds one additional page by rank five but substantially reduces MRR, nDCG, and exact-evidence ranking.

The selected retrieval configuration is frozen in `evaluation/config/retrieval_v1.json`. No model, chunking, lexical, semantic, or fusion parameter may now be changed before locked retrieval testing unless a documented implementation defect invalidates the development evidence.

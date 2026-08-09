# TAT-DQA Development Metadata Reranking

The selected spaCy model enriched the frozen 15-document development index with 190 mentions. Candidate-blind preparation extracted supported entity types from query text and resolved exact same-label normalized document/page matches before retrieval rankings were opened.

| Measure | Frozen hybrid | Metadata rerank | Delta |
|---|---:|---:|---:|
| Hit@1 | 0.7667 | 0.7667 | 0.0000 |
| Evidence-Hit@1 | 0.6897 | 0.6897 | 0.0000 |
| Recall@5 | 0.4153 | 0.4219 | +0.0067 |
| MRR@10 | 0.7944 | 0.7944 | 0.0000 |
| nDCG@10 | 0.6247 | 0.6274 | +0.0027 |

Only eight of 30 queries contained a supported spaCy entity type, six had a global exact match, and two top-ten rankings changed. Exact metadata slightly improved ordering among already retrieved relevant chunks but never improved the first relevant or first exact-evidence result. It is therefore not selected for default ranking.

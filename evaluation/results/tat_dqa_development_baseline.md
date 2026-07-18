# TAT-DQA Development Retrieval Baseline

Run date: July 18, 2026

This is development evidence, not a final test result. The locked-test split was not scored.

## Configuration

- Corpus: 15 TAT-DQA development documents, 18 pages, and 30 queries.
- Indexed chunks: 79 using `blocks-v1-120-20`.
- Lexical retrieval: SQLite FTS5 with Porter Unicode tokenization and BM25 ranking.
- Semantic retrieval: `sentence-transformers/all-MiniLM-L6-v2` with normalized embeddings and Chroma cosine distance.
- Hybrid retrieval: weighted reciprocal-rank fusion, `k=60`, keyword weight 0.33, semantic weight 0.67.
- Binary relevance: matching document and overlap with an official evidence page.
- Graded relevance: grade 2 when a relevant chunk also contains an official mapped evidence token sequence; otherwise grade 1.

All 15 documents reached `ready`. Fresh extraction and indexing completed in 7.38 seconds with the embedding model already cached locally. All 18 pages used direct extraction.

## Results

| Mode | Hit@1 | Hit@10 | Recall@1 | Recall@10 | MRR@10 | nDCG@10 | Mean query latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keyword | 0.767 | **0.867** | 0.188 | **0.485** | **0.794** | **0.617** | **1.32 ms** |
| Semantic | 0.667 | 0.833 | 0.175 | 0.484 | 0.720 | 0.573 | 5.35 ms |
| Hybrid | **0.767** | 0.833 | **0.193** | 0.472 | 0.788 | 0.602 | 7.54 ms |

Keyword retrieval returned a relevant chunk within the first ten results for 26 of 30 queries. Semantic and hybrid retrieval each succeeded for 25 of 30. Hybrid retrieval slightly improved mean Recall@1 over keyword retrieval, but keyword retrieval produced the strongest Hit@10, Recall@10, MRR@10, and nDCG@10.

## Interpretation

The provisional 0.67 semantic weight is not supported as a final choice by this run. It does not produce a broad hybrid advantage and loses one query that keyword retrieval resolves. The next development experiment should compare alternative fusion weights and equal-weight RRF using the same saved queries and judgments. The final configuration must then be frozen before any locked-test run.

One query had a valid page-level judgment but no exact mapped evidence token sequence in the extracted chunks, showing a difference between TAT-DQA's converted text and the application's PDF extraction. Its chunks remain binary-relevant for Recall, Hit, and MRR but cannot receive grade 2 for nDCG. The corpus is small and financial-domain specific, so these results do not establish statistical significance or general performance.

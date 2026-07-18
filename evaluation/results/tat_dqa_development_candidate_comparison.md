# TAT-DQA Development Candidate Comparison

Run date: July 18, 2026

These are development results. The locked-test split was not scored. Every run used the same 15 documents, 30 queries, 79 chunks, evidence judgments, `blocks-v1-120-20` chunking, and RRF constant `k=60`. One unmeasured semantic query warmed the model before latency measurement.

## Candidates

| Model | Parameters | Dimensions | Maximum tokens | Licence | Query treatment |
|---|---:|---:|---:|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22,713,216 | 384 | 256 | Apache 2.0 | Symmetric empty query/document prompts |
| `BAAI/bge-small-en-v1.5` | 33,360,000 | 384 | 512 | MIT | Recommended retrieval query instruction |

The evaluated Hugging Face revisions were `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` for MiniLM and `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` for BGE-small. The environment used Sentence Transformers 5.5.1, Transformers 5.12.0, PyTorch 2.12.0, and Chroma 1.5.9.

The BGE query instruction was:

```text
Represent this sentence for searching relevant passages:
```

## Hybrid Results

| Model | Keyword / semantic weight | Hit@1 | Hit@10 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | 0.33 / 0.67 | **0.767** | 0.833 | 0.408 | 0.472 | 0.788 | 0.602 | 7.54 ms |
| MiniLM | 0.50 / 0.50 | **0.767** | 0.833 | **0.415** | **0.489** | **0.794** | 0.625 | **6.73 ms** |
| MiniLM | 0.67 / 0.33 | **0.767** | 0.833 | 0.404 | 0.485 | **0.794** | **0.626** | 6.80 ms |
| BGE-small prompted | 0.50 / 0.50 | 0.700 | **0.900** | **0.421** | **0.547** | **0.749** | **0.612** | **11.43 ms** |
| BGE-small prompted | 0.67 / 0.33 | 0.667 | 0.867 | 0.403 | 0.505 | 0.740 | 0.600 | 11.61 ms |

Bold values identify the best result within each model where there is a weight comparison. BGE-small's equal-weight configuration produces the strongest top-ten coverage. MiniLM produces better early ranking, reciprocal rank, graded gain, and latency.

## Semantic-Only Results

| Model | Hit@10 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM | **0.833** | 0.408 | 0.484 | **0.720** | **0.573** | **4.81 ms** |
| BGE-small prompted | 0.767 | **0.411** | **0.486** | 0.661 | 0.537 | 9.12 ms |

## Selection

MiniLM with equal keyword and semantic RRF weights is selected for the current application. The browser workflow returns five results by default, so early ranking, MRR, nDCG, and top-five behavior are more important than retrieving additional same-page chunks near rank ten. MiniLM equal weighting has the best MiniLM Recall@5 and Recall@10, ties the best MRR, and is within 0.001 nDCG of the keyword-heavy variant. Equal weighting is also simpler to justify than a near-tied asymmetric setting.

BGE-small is not rejected as ineffective. Its prompted equal-weight run raises Hit@10 from 0.833 to 0.900 and Recall@10 from 0.489 to 0.547. However, it lowers Hit@1 from 0.767 to 0.700, MRR from 0.794 to 0.749, and nDCG from 0.625 to 0.612 while increasing hybrid latency by approximately 70% and parameter count by approximately 47%. This is a coverage-versus-ranking tradeoff, not a universal model ordering.

The embedding model and fusion weights are selected using development data only. The complete retrieval configuration remains unfrozen until chunking candidates are compared. No result in this document uses the locked-test split.

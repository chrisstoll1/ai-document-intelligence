# Evaluation Results

## Local Sample Evaluation

Latest run used the Python 3.11 virtual environment and `sentence-transformers/all-MiniLM-L6-v2`.

```powershell
.\.venv\Scripts\python.exe -m src.docintel.cli evaluate --mode all
```

| Mode | Top-3 success | Top-1 success |
|---|---:|---:|
| Keyword | 8/8 | 7/8 |
| Vector | 8/8 | 7/8 |
| Combined | 8/8 | 7/8 |

## Notes

- The sample collection has seven documents: four relevant project documents and three distractors.
- All methods found the expected source within the top three results.
- The query set now has eight known-answer cases, including harder wording-mismatch questions.
- All three modes reached 7/8 top-1 success on this run.
- The remaining failures show that a related distractor can still appear above the best evidence.
- The next evaluation set should include more documents and passage-level expected answers, not only document-level labels.

## Public Benchmark Evaluation

The prototype was also evaluated on the public BEIR/SciFact test set. The benchmark has 5183 scientific documents, 300 test queries and public relevance judgements.

```powershell
.\.venv\Scripts\python.exe -m src.docintel.cli benchmark --dataset scifact --mode all --limit 10
```

| Mode | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Keyword | 0.775 | 0.085 | 0.625 | 0.656 |
| Vector | 0.794 | 0.089 | 0.587 | 0.635 |
| Combined | 0.838 | 0.094 | 0.673 | 0.710 |

The combined method performed best overall on this public benchmark. This supports keeping hybrid ranking in the design rather than relying only on keyword matching or only on vector similarity.

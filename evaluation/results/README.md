# Evaluation Results

Machine-readable development and final evaluation outputs are stored here. Development runs may guide model and fusion decisions. Locked-test results must only be generated after the configuration and evaluation implementation are frozen.

Run the current development baseline with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

The evaluator uses the production SQLite FTS5, Chroma, and weighted reciprocal-rank-fusion implementations. Recall and MRR treat a chunk as relevant when it belongs to the judged document and overlaps an official evidence page. nDCG assigns grade 2 when the chunk also contains an official mapped evidence token sequence and grade 1 for other chunks on the evidence page.

The first development baseline is recorded in:

- `tat_dqa_development_baseline.md`: readable method, results, and interpretation.
- `tat_dqa_development_baseline.json`: complete configuration, environment, aggregate metrics, per-query rankings, judgments, and timings.

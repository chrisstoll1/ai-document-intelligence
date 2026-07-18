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
- `tat_dqa_development_candidate_comparison.md`: MiniLM/BGE-small and RRF-weight comparison with the selection decision.
- `tat_dqa_development_*_kw*_sem*.json`: complete candidate runs used by the comparison.
- `tat_dqa_development_chunking_comparison.md`: 80/15, 120/20, and 180/30 chunk comparison and final selection.
- `../config/retrieval_v1.json`: machine-readable retrieval configuration frozen before locked testing.
- `ocr_development_candidate_comparison.md`: Tesseract/EasyOCR clean and degraded development comparison.
- `ocr_development_candidate_comparison.json`: complete OCR environment, metrics, predictions, and timings.
- `../config/ocr_v1.json`: selected Tesseract version, trained-data checksum, routing, and benchmark evidence.

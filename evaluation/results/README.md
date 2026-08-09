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
- `ner_development_candidate_comparison.md`: reviewed spaCy/BERT NER comparison and selection decision.
- `ner_development_candidate_comparison.json`: complete NER environment, metrics, predictions, and timings.
- `../config/ner_v1.json`: selected spaCy model, evaluated taxonomy, version identity, and evidence.
- `tat_dqa_locked_test_selected.json`: complete frozen direct-text locked retrieval rankings and metrics.
- `tat_dqa_development_metadata_rerank.json`: exact selected-NER metadata ablation over frozen rankings.
- `ocr_retrieval_development.json` and `ocr_retrieval_locked_test.json`: paired clean/degraded full-page OCR retrieval results.
- `generation_development_qwen.json` and `generation_development_mistral.json`: generation candidate-selection outputs.
- `generation_locked_test_qwen.json`: selected Qwen locked outputs, citations, refusals, and automatic metrics.
- `generation_qwen_resource_profile.json`: selected generator initialization, latency, RAM, and GPU measurements.
- `reproducibility_check.md`: clean Python/CUDA and frontend installation verification.

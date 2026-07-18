# Controlled OCR Development Benchmark

This benchmark compares OCR candidates without using the locked TAT-DQA test split. TAT-DQA has digital text layers, so the preparation step renders selected development-only line regions as images and uses their directly extracted text as a silver-standard reference.

## Prepare

First prepare the official TAT-DQA subset, then run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ocr_benchmark.py
```

The fixed seed `docintel-ocr-development-v1` selects six prose and six number-heavy line regions from distinct documents within each category. Each region produces a clean 300-DPI image and a deterministic degraded variant. Raw PNG files are written under the ignored `data/evaluation/ocr/` directory; `development_manifest.json` records references, source coordinates, image checksums, and the exact transformation.

## Evaluate

Install the evaluation-only EasyOCR candidate and run both engines:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr-eval]"
.\.venv\Scripts\python.exe scripts\evaluate_ocr.py
```

The evaluator preserves case and punctuation, applies Unicode NFKC and whitespace normalization, and reports micro-averaged Levenshtein character error rate (CER), word error rate (WER), failures, and warmed sequential CPU latency. Image loading, model initialization, and one warm-up inference are excluded from measured latency.

## Limitations

The reference is directly extracted PDF text rather than independently transcribed ground truth. The benchmark contains 12 base regions and controlled degradation rather than naturally scanned pages, so its results support a project-level candidate decision but do not estimate OCR performance for all document types.

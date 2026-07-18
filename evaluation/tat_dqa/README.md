# TAT-DQA Evaluation Subset

This directory records the reproducible corpus selection for retrieval and grounded-answer evaluation. Raw dataset files and extracted PDFs live under `data/evaluation/tat-dqa/` and are excluded from Git.

## Source

- Dataset: TAT-DQA, *Towards Complex Document Understanding By Discrete Reasoning*
- Official site: https://nextplusplus.github.io/TAT-DQA/
- Official download: https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Retrieved: July 18, 2026

The project uses only the official development and test annotations, released test ground truth, and matching document archives. The training split is intentionally not downloaded. Official file identifiers, byte sizes, and SHA-256 checksums are recorded in `source_files.json`.

## Protocol

Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_tat_dqa.py --download
```

The preparation script verifies every official source file before extracting the selected PDFs and converted JSON. Selection uses SHA-256 ordering over public document and question identifiers with the fixed seed `docintel-tat-dqa-v1`.

- Development: 15 documents, including 3 multipage documents, with 2 queries per document.
- Locked test: 10 documents, including 2 multipage documents, with 2 queries per document.
- Total: 25 documents and 50 queries.

The development split may be used for model, chunking, and fusion decisions. The locked-test split must not be used for tuning. Its first scored run should occur only after the retrieval configuration and evaluation code are frozen.

Each query records evidence blocks and local PDF page numbers derived from the official block mappings. Answers and derivations are deliberately omitted from the committed manifests because the first evaluation milestone concerns retrieval, not numerical question answering.

## Scope Limitation

All 551 official development and test PDFs were checked with direct PDF extraction and every page exceeded the application's OCR routing threshold. TAT-DQA therefore supplies visually rich digital PDFs but does not independently test OCR. OCR model comparison requires a separately declared benchmark, such as controlled image-only variants with text references or a licensed scanned-document corpus.

## Citation

Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. 2022. *Towards Complex Document Understanding by Discrete Reasoning*. Proceedings of the 30th ACM International Conference on Multimedia, 4857-4866.

# Locked-Test OCR Retrieval Impact

The frozen paired protocol was applied once to 12 pages from ten locked-test documents. Every page followed the production Tesseract route, and clean and degraded conditions used isolated indexes with identical queries and canonical page judgments.

| Condition | Page CER | Page WER | Hybrid Hit@1 | Hybrid Hit@5 | Hybrid MRR@5 | Hybrid nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Clean OCR | 8.93% | 14.97% | 0.800 | 1.000 | 0.892 | 0.731 |
| Degraded OCR | 11.04% | 17.01% | 0.800 | 1.000 | 0.870 | 0.716 |

Degradation reduced paired hybrid MRR@5 by `0.0217`. Two queries improved, 16 tied, and two fell from rank one to rank five. Hit@1 and Hit@5 were unchanged, but Recall@10 decreased from 0.640 to 0.602. The result indicates modest early-ranking sensitivity rather than catastrophic retrieval failure under this synthetic degradation.

These figures apply to a small financial-report sample and artificial full-page scans. Page-level relevance, changed OCR chunk boundaries, and synthetic rather than natural defects limit generalisation.

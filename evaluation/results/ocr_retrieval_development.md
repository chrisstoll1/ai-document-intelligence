# Development OCR Retrieval Impact

All 18 development pages were rendered into paired clean and deterministically degraded image-only PDFs. Every generated page followed the production Tesseract route before frozen `retrieval-v1` indexing.

| Condition | Page CER | Page WER | Hybrid Hit@1 | Hybrid Hit@5 | Hybrid MRR@5 |
|---|---:|---:|---:|---:|---:|
| Clean OCR | 7.71% | 28.29% | 0.800 | 0.900 | 0.840 |
| Degraded OCR | 8.15% | 29.01% | 0.833 | 0.900 | 0.858 |

The degraded condition produced two query-level MRR@5 improvements, 28 ties, and no losses (`+0.0183` mean). This does not imply degradation is beneficial: OCR changed chunk text and ranking interactions, and the small sample can produce favorable noise. The defensible conclusion is that the frozen synthetic degradation measurably increased OCR error but did not reduce early retrieval on development data. The protocol was frozen unchanged before locked evaluation.

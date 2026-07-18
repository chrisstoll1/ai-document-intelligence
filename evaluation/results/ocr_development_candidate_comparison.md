# OCR Development Candidate Comparison

## Protocol

Tesseract 5.4.0 and EasyOCR 1.7.2 were compared on 24 development-only images: six prose and six number-heavy TAT-DQA line regions, each rendered as a clean 300-DPI crop and a deterministic degraded variant. Directly extracted NFKC-normalized line text provides a silver-standard reference. Accuracy uses micro-averaged Levenshtein character error rate (CER) and word error rate (WER). Latency is warmed sequential CPU inference and excludes image loading and model initialization.

## Results

| Engine | Variant | Samples | CER | WER | Mean latency | Failures |
|---|---|---:|---:|---:|---:|---:|
| Tesseract | Clean | 12 | **0.08%** | **1.08%** | **93.11 ms** | 0 |
| Tesseract | Degraded | 12 | **1.42%** | **8.60%** | **99.45 ms** | 0 |
| **Tesseract** | **Overall** | **24** | **0.75%** | **4.84%** | **96.28 ms** | **0** |
| EasyOCR | Clean | 12 | 2.00% | 12.37% | 179.09 ms | 0 |
| EasyOCR | Degraded | 12 | 2.42% | 16.67% | 183.67 ms | 0 |
| EasyOCR | Overall | 24 | 2.21% | 14.52% | 181.38 ms | 0 |

Tesseract also performed better within both content categories. Its number-heavy CER/WER were 1.61%/10.53%, compared with EasyOCR's 3.72%/22.37%. Its prose CER/WER were 0.14%/0.91%, compared with 1.14%/9.09%.

## Decision

Tesseract is selected for `ocr-v1`. It made 66% fewer character edits and 67% fewer word edits than EasyOCR while its mean latency was 47% lower. It already supports the page-level production path without EasyOCR's additional PyTorch, CRAFT, and OpenCV runtime dependencies. The selected executable, English trained-data checksum, routing threshold, render DPI, and benchmark evidence are frozen in `../config/ocr_v1.json`.

## Limitations

The 12 base references are directly extracted text rather than independent human transcriptions. Controlled clean/degraded line crops do not represent naturally scanned full pages, handwriting, rotation extremes, or every layout. The comparison supports this project's candidate selection but does not establish general OCR accuracy. Downstream retrieval impact remains part of the final aggregate evaluation.

# NER Development Candidate Comparison

## Protocol

spaCy `en_core_web_trf` 3.8.0 and `dslim/bert-base-NER` at immutable revision `d1a3e8f...` were compared on 30 reviewed TAT-DQA development passages. One general and one model-independent proper-name challenge passage were selected from each of 15 documents before candidate output was viewed. The shared taxonomy contains `PERSON`, `ORGANIZATION`, and `LOCATION`. The reviewed references contain 33 mentions: 26 organisations, five locations, and two people.

Strict exact-span-and-label micro precision, recall, and F1 are primary. Same-label overlap is secondary. Latency is warmed sequential CPU inference and excludes model initialization and one warm-up call.

## Results

| Candidate | Strict precision | Strict recall | Strict F1 | Overlap F1 | Mean latency | Failures |
|---|---:|---:|---:|---:|---:|---:|
| **spaCy `en_core_web_trf`** | **0.700** | **0.848** | **0.767** | **0.849** | 35.33 ms | 0 |
| `dslim/bert-base-NER` | 0.458 | 0.818 | 0.587 | 0.652 | **24.50 ms** | 0 |

| Label | References | spaCy F1 | BERT F1 |
|---|---:|---:|---:|
| PERSON | 2 | 1.000 | 0.800 |
| ORGANIZATION | 26 | 0.733 | 0.557 |
| LOCATION | 5 | 0.889 | 0.750 |

## Decision

spaCy is selected for `ner-v1`. It improved strict F1 by 0.180 and reduced false positives from 32 to 12 while also slightly improving recall. BERT was approximately 31% faster, but spaCy's 35 ms mean passage latency remains acceptable for asynchronous document ingestion and its precision is more suitable for metadata that users may inspect or use as a filter.

The most common spaCy errors were over-classifying generic or document-specific capitalized phrases such as `Group`, `Board of Directors`, `FIAA`, and accounting-standard titles as organisations. Three otherwise correct organisations differed only by a leading article or boundary. BERT produced substantially more false organisations, including products, generic groups, standards, and fragmented acronyms such as `TCJA`.

## Limitations

This is a small development comparison, not a final general-performance estimate. PERSON and LOCATION counts are too low for strong per-label conclusions. The financial domain and capitalization-based challenge stratum may not represent other document collections. The models identify mention spans only; neither resolves aliases such as `Peel` and `Peel Group` to a canonical entity. Metadata filtering or reranking remains a separate development experiment and will not modify frozen `retrieval-v1` by default.

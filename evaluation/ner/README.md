# NER Development Benchmark

This benchmark supports candidate selection for the metadata model without using the locked TAT-DQA test split. It uses official converted blocks from the 15 development documents under the dataset's CC BY 4.0 licence.

## Taxonomy

- `PERSON`: a specifically named person; titles and unnamed roles are excluded.
- `ORGANIZATION`: a named company, institution, agency, or formal group; legal suffixes are included when present.
- `LOCATION`: a named geopolitical area, geographic place, or facility.

The shared taxonomy permits a fair comparison between spaCy `en_core_web_trf` and `dslim/bert-base-NER`. Dates, money, percentages, products, and miscellaneous names are outside this benchmark and must not be silently mapped into the three labels.

## Preparation

Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ner_benchmark.py
```

The script selects one general passage and one proper-name challenge passage per development document through SHA-256 ordering with seed `docintel-ner-development-v1`. The challenge stratum uses only a model-independent capitalization pattern; it does not use either candidate's output. This prevents a sample dominated by financial rows while retaining general and entity-negative examples. The script refuses to overwrite an existing manifest so completed annotations cannot be erased accidentally.

Annotations use exact character offsets into the committed normalized text. Every repeated mention is annotated, spans cannot overlap, and all passages require review before candidate evaluation. Candidate output must not be viewed while creating or reviewing the reference labels.

`development_preannotations.json` contains AI-assisted candidate-blind labels separated from the generated passage manifest. Its `awaiting_human_review` status is intentional: these spans are not evaluation ground truth until a human checks the proposed labels, confirms negative passages, changes every passage to `reviewed`, and changes the top-level status to `reviewed`.

Validate the manifest/preannotation link, offsets, and current review status without loading either model:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ner.py --validate-only
```

The full evaluator refuses to load either candidate until the top-level annotation status and every passage are marked `reviewed`. After review, install the isolated candidate dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ner-eval]"
.\.venv\Scripts\python.exe scripts\evaluate_ner.py
```

Strict exact-span-and-label precision, recall, and F1 are primary. Same-label overlap metrics are diagnostic only. The result also records per-label and per-stratum metrics, initialization, warm-up, sequential CPU latency, failures, package versions, model identities, and every prediction.

## Limitations

The benchmark is a small, single-project development set drawn from financial reports. It supports local candidate selection but cannot establish general NER accuracy. Annotation counts and sparse labels must be reported explicitly, and final evaluation still requires data that was not used for model selection.

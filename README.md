# Document Intelligence

Local-first PDF ingestion, entity enrichment, hybrid evidence retrieval, and grounded generation. The backend stores original PDFs and application records locally, indexes chunks with SQLite FTS5 and Chroma, and preserves page/block/entity/citation provenance.

The React frontend provides PDF collection management, grounded questions, claim-level citation navigation, evidence cards, and original-page links. FastAPI also exposes generated interactive API documentation.

## Setup

Requires Python 3.11-3.13, Node.js, and Tesseract 5. Selected Qwen generation requires a CUDA-capable NVIDIA GPU; the evaluated setup uses an RTX 4090 with 24 GB VRAM and PyTorch 2.12 CUDA 13.0. On Windows, the backend detects the standard `C:\Program Files\Tesseract-OCR` installation automatically. For another location, set `TESSERACT_CMD` to the full executable path.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm ci --prefix frontend
```

Runtime data defaults to `data/`, which is ignored by Git. Set `DOCINTEL_DATA_DIR` before launch to use a different location.
The embedding model defaults to the evaluated `sentence-transformers/all-MiniLM-L6-v2` revision recorded in `DOCINTEL_EMBEDDING_REVISION`; set `DOCINTEL_EMBEDDING_MODEL` and its revision to use another evaluated compatible model. Models that require an asymmetric retrieval instruction can use `DOCINTEL_EMBEDDING_QUERY_PROMPT`.
Chunking defaults to 120 words with 20-word overlap for oversized blocks. `DOCINTEL_CHUNK_MAX_WORDS` and `DOCINTEL_CHUNK_OVERLAP` select another evaluated configuration; incompatible persisted chunks and embeddings are rebuilt from the stored page extraction.
NER defaults to the evaluated spaCy `en_core_web_trf` 3.8.0 package. `DOCINTEL_NER_MODEL` and `DOCINTEL_NER_MODEL_VERSION` select another compatible model; changed NER settings rebuild only metadata from persisted page text.
Generation defaults to pinned `Qwen/Qwen2.5-7B-Instruct`. `DOCINTEL_GENERATION_MODEL`, `DOCINTEL_GENERATION_REVISION`, and `DOCINTEL_GENERATION_MAX_NEW_TOKENS` select another compatible evaluated configuration. Generation loads lazily on the first answer request.

## Run

Select **Full App** in PyCharm and press Run.

Or from a terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

- App: http://127.0.0.1:5173
- API docs: http://127.0.0.1:8000/docs

The first document upload may download `sentence-transformers/all-MiniLM-L6-v2` and lazily loads the installed spaCy pipeline. The first answer may download and load Qwen. Later runs reuse the local model cache and compatible persisted indexes.

## API

- `POST /api/documents`: validate, store, extract, chunk, enrich, and index a PDF.
- `GET /api/documents`: list locally stored documents and processing status.
- `GET /api/documents/{id}`: retrieve document status and metadata.
- `GET /api/documents/{id}/pdf`: retrieve the original PDF.
- `GET /api/documents/{id}/metadata`: retrieve enrichment status and page-relative entity mentions.
- `DELETE /api/documents/{id}`: remove one PDF and all relational, lexical, metadata, and vector records.
- `DELETE /api/documents`: reset the collection and recreate the active Chroma collection.
- `POST /api/search`: run weighted reciprocal-rank fusion over FTS5 and Chroma candidates.
- `POST /api/answer`: retrieve five passages and return schema-constrained claims with validated context IDs.
- `GET /api/health`: liveness check.

Digital pages use direct extraction. Pages with fewer than 20 alphanumeric characters are rendered at 300 DPI and routed through Tesseract OCR. OCR failures are retained as document processing errors rather than silently ignored.
Metadata enrichment runs after core indexing. Its status and errors are independent so an NER failure does not make lexical or semantic evidence unavailable.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
```

## Evaluation Data

The reproducible TAT-DQA development and locked-test subsets are defined under `evaluation/tat_dqa/`. Raw source archives and extracted PDFs are stored under the ignored `data/evaluation/tat-dqa/` directory.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_tat_dqa.py --download
```

The command downloads only the official development and test artifacts, verifies fixed SHA-256 checksums, and prepares 25 documents with 50 evidence-mapped queries. Development data may be used for tuning; the locked-test subset must not be scored until the retrieval configuration is frozen. TAT-DQA contains digitally extractable PDFs, so the OCR comparison uses a separately declared controlled benchmark.

The controlled development-only OCR comparison can then be reproduced with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ocr_benchmark.py
.\.venv\Scripts\python.exe -m pip install -e ".[ocr-eval]"
.\.venv\Scripts\python.exe scripts\evaluate_ocr.py
```

Tesseract 5.4 was selected over EasyOCR 1.7.2 for `ocr-v1` from clean and degraded prose/number-heavy evidence. The protocol, complete predictions, limitations, and frozen configuration are recorded under `evaluation/ocr/`, `evaluation/results/`, and `evaluation/config/`.

The candidate-blind NER development set can be regenerated and its current annotations validated without loading either candidate:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ner_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_ner.py --validate-only
.\.venv\Scripts\python.exe -m pip install -e ".[ner-eval]"
.\.venv\Scripts\python.exe scripts\evaluate_ner.py
```

The project owner reviewed the candidate-blind annotations before evaluation. spaCy `en_core_web_trf` was selected over pinned `dslim/bert-base-NER` for stronger strict precision, recall, and F1. The protocol, complete predictions, limitations, selected mapping, and frozen `ner-v1` configuration are recorded under `evaluation/ner/`, `evaluation/results/`, and `evaluation/config/`.

Metadata reranking, full-page OCR retrieval impact, direct locked retrieval, and selected Qwen generation have also been evaluated. Frozen policies and result hashes are stored in `evaluation/config/`; complete per-query results are stored in `evaluation/results/`. Evaluation scripts refuse to overwrite frozen artifacts unless `--overwrite` is explicitly supplied, and locked evaluators require `--confirm-locked-test`.

The selected generator's locked arithmetic coverage is poor despite valid citation structure. Do not treat a valid citation ID as proof that the associated numerical claim is correct; inspect cited source passages.

# Document Intelligence

Local-first PDF ingestion and hybrid evidence retrieval. The backend stores original PDFs and application records locally, indexes chunks with SQLite FTS5 and Chroma, and preserves page/block provenance.

The React frontend is still a minimal shell. Use the generated API documentation to exercise the implemented backend workflow.

## Setup

Requires Python 3.11-3.13 and Node.js. Scanned PDFs also require Tesseract 5 with the `tesseract` executable available on `PATH`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm install --prefix frontend
```

Runtime data defaults to `data/`, which is ignored by Git. Set `DOCINTEL_DATA_DIR` before launch to use a different location.

## Run

Select **Full App** in PyCharm and press Run.

Or from a terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

- App: http://127.0.0.1:5173
- API docs: http://127.0.0.1:8000/docs

The first document upload may download `sentence-transformers/all-MiniLM-L6-v2`. Later runs reuse the local model cache and persisted indexes.

## API

- `POST /api/documents`: validate, store, extract, chunk, and index a PDF.
- `GET /api/documents`: list locally stored documents and processing status.
- `GET /api/documents/{id}`: retrieve document status and metadata.
- `GET /api/documents/{id}/pdf`: retrieve the original PDF.
- `POST /api/search`: run weighted reciprocal-rank fusion over FTS5 and Chroma candidates.
- `GET /api/health`: liveness check.

Digital pages use direct extraction. Pages with fewer than 20 alphanumeric characters are rendered at 300 DPI and routed through Tesseract OCR. OCR failures are retained as document processing errors rather than silently ignored.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
npm --prefix frontend run lint
npm --prefix frontend run build
```

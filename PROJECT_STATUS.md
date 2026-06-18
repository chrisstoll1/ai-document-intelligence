# Prototype Status

This file tracks the feature prototype so the report and code do not drift apart.

## Goal

Build a small evidence-based retrieval prototype that proves the core project idea is feasible.

## Current Scope

- Load a small document collection.
- Split documents into source passages.
- Compare keyword, vector and combined retrieval.
- Store simple metadata with each chunk.
- Evaluate against known-answer queries.

## Implemented

- Python 3.11 virtual environment created at `.venv`.
- `sentence-transformers/all-MiniLM-L6-v2` installed and working.
- Paragraph-aware chunking implemented.
- Rule-based metadata extraction implemented.
- Hybrid ranking implemented with 0.6 semantic, 0.3 keyword and 0.1 metadata score.
- Evaluation compares keyword, vector and combined modes.
- Plain browser workbench implemented with Python's standard library HTTP server.
- Public BEIR/SciFact benchmark command implemented.
- Retrieval code organised around loader, chunker, metadata extractor, ranker and benchmark runner classes.

## Design Decisions

- First prototype uses local Markdown/text documents.
- Chunking should be paragraph-aware where possible.
- Metadata extraction starts with simple rules: filename type, dates and capitalised entity-like phrases.
- Hybrid ranking uses weighted keyword, vector and metadata scores.
- Real Sentence-BERT embeddings use `sentence-transformers/all-MiniLM-L6-v2` when the Python 3.11 virtual environment is used.
- The fallback hashed vector search remains available for machines without ML dependencies.

## Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe -m src.docintel.cli search "privacy risks" --mode combined
.\.venv\Scripts\python.exe -m src.docintel.cli evaluate --mode all
.\.venv\Scripts\python.exe -m src.docintel.cli benchmark --dataset scifact --mode all --limit 10
.\.venv\Scripts\python.exe -m src.docintel.web
```

## Later Extensions

- PDF extraction with PyMuPDF.
- OCR for scanned PDFs.
- Chroma or FAISS vector persistence.
- Metadata extraction with spaCy or Hugging Face.
- Expanded web interface with upload and source inspection.
- Optional cited summaries from retrieved passages.

## Latest Evaluation

Using `sentence-transformers/all-MiniLM-L6-v2` on seven sample documents and eight known-answer queries:

| Mode | Top-3 success | Top-1 success |
|---|---:|---:|
| Keyword | 8/8 | 7/8 |
| Vector | 8/8 | 7/8 |
| Combined | 8/8 | 7/8 |

Using the public BEIR/SciFact test set with 5183 documents and 300 queries:

| Mode | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Keyword | 0.775 | 0.085 | 0.625 | 0.656 |
| Vector | 0.794 | 0.089 | 0.587 | 0.635 |
| Combined | 0.838 | 0.094 | 0.673 | 0.710 |

# Evidence Retrieval Prototype

This is the feature prototype for the final project idea: **AI-Assisted Document Intelligence**.

The prototype focuses on the retrieval part of the system. It loads a small document collection, splits files into passages, indexes them, and compares keyword search, semantic vector search and a hybrid ranking method. The aim is to check whether evidence-based retrieval is feasible before adding upload handling, OCR and generated summaries.

## What It Demonstrates

- Class-based document loading from text or Markdown files.
- Paragraph-aware chunking into searchable passages.
- Keyword retrieval as a baseline.
- Vector retrieval using `sentence-transformers/all-MiniLM-L6-v2` when available.
- Rule-based metadata extraction for dates, document type and entity-like phrases.
- Hybrid ranking using semantic, keyword and metadata scores.
- A plain browser workbench that shows document name, scores and source passage.
- An eight-query known-answer evaluation set.
- Public BEIR/SciFact benchmark evaluation with Recall@k, Precision@k, MRR@k and nDCG@k.

The vector search uses `sentence-transformers/all-MiniLM-L6-v2` when the optional ML dependencies are installed. If they are not available, the code falls back to dependency-light hashed vectors so the prototype still runs for smoke tests.

## Project Structure

```text
ai-document-intelligence/
  data/
    sample_docs/       Example documents for testing
    queries/           Known-answer evaluation queries
  src/docintel/         Prototype package and browser workbench
  tests/                Smoke tests
  pyproject.toml        Project metadata
```

## Run It

From this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m src.docintel.cli search "privacy risks" --mode combined
```

Run the small evaluation set:

```powershell
.\.venv\Scripts\python.exe -m src.docintel.cli evaluate --mode all
```

Run the public SciFact benchmark. This downloads the BEIR SciFact dataset into `data/benchmarks`, which is ignored by git:

```powershell
.\.venv\Scripts\python.exe -m src.docintel.cli benchmark --dataset scifact --mode all --limit 10
```

Run the browser workbench:

```powershell
.\.venv\Scripts\python.exe -m src.docintel.web
```

Then open `http://127.0.0.1:8000`.

Run tests:

```powershell
python -m unittest discover tests
```

If the virtual environment is not available, the system can still run with the default `python` command, but it will use the fallback vector search instead of Sentence-BERT.

## Prototype Scope

This is not the final system. It includes a lightweight browser workbench, but intentionally leaves out the full upload interface, OCR and LLM summaries for now. The first goal is to prove the retrieval core works and can be evaluated against known queries.

## Next Steps

- Add Chroma or FAISS for persistent vector storage.
- Add PDF extraction using PyMuPDF or pdfplumber.
- Add OCR for scanned documents.
- Add metadata/entity extraction with spaCy or a Hugging Face model.
- Expand the browser UI with upload and source inspection.
- Add optional cited summaries from retrieved passages.

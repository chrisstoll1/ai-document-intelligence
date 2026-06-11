# AI Document Intelligence Prototype

This is the feature prototype for the final project idea: **AI-Assisted Document Intelligence**.

The prototype focuses on the core technical feature: evidence-based retrieval from a small document collection. It loads documents, splits them into passages, indexes them, and lets a user compare keyword search with a simple semantic-style vector search. The aim is to prove that the retrieval part of the project is feasible before building the full web app.

## What It Demonstrates

- Document loading from text or Markdown files.
- Chunking documents into searchable passages.
- Keyword retrieval as a baseline.
- Vector retrieval as a semantic-search placeholder.
- Result display with document name and source passage.
- A small query set for evaluation.

The current vector search is dependency-light and uses hashed bag-of-words vectors. This is enough for the prototype structure and smoke tests. Later, it can be replaced with Sentence-BERT, FAISS, Chroma, OCR and language-model summarisation.

## Project Structure

```text
ai-document-intelligence/
  data/
    sample_docs/       Example documents for testing
    queries/           Known-answer evaluation queries
  src/docintel/         Prototype package
  tests/                Smoke tests
  pyproject.toml        Project metadata
```

## Run It

From this folder:

```powershell
python -m src.docintel.cli search "privacy risks" --mode both
```

Run the small evaluation set:

```powershell
python -m src.docintel.cli evaluate
```

Run tests:

```powershell
python -m unittest discover tests
```

## Prototype Scope

This is not the final system. It intentionally leaves out the full web interface, OCR and LLM summaries for now. The first goal is to prove the retrieval core works and can be evaluated against known queries.

## Next Steps

- Replace the fallback vector search with Sentence-BERT embeddings.
- Add PDF extraction using PyMuPDF or pdfplumber.
- Add OCR for scanned documents.
- Add metadata/entity extraction with spaCy or a Hugging Face model.
- Add a small web UI for upload, search and source inspection.
- Add optional cited summaries from retrieved passages.

# OCR Notes

OCR converts scanned pages and image-based documents into machine-readable text. This is useful when a PDF does not contain selectable text. Without OCR, a scanned report may be invisible to a search system.

OCR quality depends on scan resolution, page layout, fonts, skew, tables and image noise. Errors in this stage can affect later retrieval because the indexed text may not match the original document.

For the prototype, OCR can be treated as a later extension. The first version can use clean text files while still keeping the ingestion pipeline designed for OCR support.

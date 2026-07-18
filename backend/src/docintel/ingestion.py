from __future__ import annotations

from collections.abc import Sequence
from typing import BinaryIO, Protocol

from docintel.chunking import ChunkRepository
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractedPage, ExtractionRepository


class PageExtractor(Protocol):
    def extract(self, path) -> Sequence[ExtractedPage]: ...


class SemanticIndexer(Protocol):
    model_name: str

    def replace_document(self, document_id: str, chunks) -> None: ...


class IngestionService:
    def __init__(
        self,
        catalog: DocumentCatalog,
        documents: DocumentRepository,
        extractor: PageExtractor,
        extractions: ExtractionRepository,
        chunks: ChunkRepository,
        semantic_index: SemanticIndexer | None = None,
    ) -> None:
        self.catalog = catalog
        self.documents = documents
        self.extractor = extractor
        self.extractions = extractions
        self.chunks = chunks
        self.semantic_index = semantic_index

    def ingest(self, source: BinaryIO, original_filename: str) -> DocumentRecord:
        document = self.catalog.add_pdf(source, original_filename)
        if self.semantic_index is None and document.status in {"indexed_lexical", "ready"}:
            return document
        if (
            self.semantic_index is not None
            and document.status == "ready"
            and document.embedding_model == self.semantic_index.model_name
        ):
            return document

        chunks = self.chunks.list_document(document.id)
        lexical_ready = document.status in {"indexed_lexical", "index_failed", "ready"} and bool(chunks)
        try:
            if not lexical_ready:
                self.documents.set_status(document.id, "processing")
                pages = self.extractor.extract(self.catalog.pdf_store.path_for(document.id))
                self.extractions.replace_pages(document.id, pages)
                chunks = self.chunks.rebuild(document.id)
                lexical_ready = True
            if self.semantic_index is not None:
                self.semantic_index.replace_document(document.id, chunks)
                self.documents.mark_ready(document.id, self.semantic_index.model_name)
        except Exception as error:
            failure_status = "index_failed" if lexical_ready and self.semantic_index is not None else "failed"
            self.documents.set_status(document.id, failure_status, str(error))
            raise

        completed = self.documents.get(document.id)
        if completed is None:
            raise RuntimeError("Ingested document could not be loaded")
        return completed

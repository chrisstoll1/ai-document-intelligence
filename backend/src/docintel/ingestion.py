from __future__ import annotations

from collections.abc import Sequence
from typing import BinaryIO, Protocol

from docintel.chunking import ChunkRepository
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractedPage, ExtractionRepository


class PageExtractor(Protocol):
    def extract(self, path) -> Sequence[ExtractedPage]: ...


class IngestionService:
    def __init__(
        self,
        catalog: DocumentCatalog,
        documents: DocumentRepository,
        extractor: PageExtractor,
        extractions: ExtractionRepository,
        chunks: ChunkRepository,
    ) -> None:
        self.catalog = catalog
        self.documents = documents
        self.extractor = extractor
        self.extractions = extractions
        self.chunks = chunks

    def ingest(self, source: BinaryIO, original_filename: str) -> DocumentRecord:
        document = self.catalog.add_pdf(source, original_filename)
        if document.status == "indexed_lexical":
            return document

        self.documents.set_status(document.id, "processing")
        try:
            pages = self.extractor.extract(self.catalog.pdf_store.path_for(document.id))
            self.extractions.replace_pages(document.id, pages)
            self.chunks.rebuild(document.id)
        except Exception as error:
            self.documents.set_status(document.id, "failed", str(error))
            raise

        completed = self.documents.get(document.id)
        if completed is None:
            raise RuntimeError("Ingested document could not be loaded")
        return completed

from __future__ import annotations

from collections.abc import Sequence
from typing import BinaryIO, Protocol

from docintel.chunking import ChunkRepository
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractedPage, ExtractionRepository
from docintel.metadata import EntityExtractor, MetadataRepository


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
        metadata_repository: MetadataRepository | None = None,
        metadata_extractor: EntityExtractor | None = None,
    ) -> None:
        if (metadata_repository is None) != (metadata_extractor is None):
            raise ValueError("Metadata repository and extractor must be configured together")
        self.catalog = catalog
        self.documents = documents
        self.extractor = extractor
        self.extractions = extractions
        self.chunks = chunks
        self.semantic_index = semantic_index
        self.metadata_repository = metadata_repository
        self.metadata_extractor = metadata_extractor

    def ingest(self, source: BinaryIO, original_filename: str) -> DocumentRecord:
        document = self.catalog.add_pdf(source, original_filename)
        chunker_version = self.chunks.chunker.version
        chunker_matches = document.chunker_version == chunker_version
        chunks = self.chunks.list_document(document.id) if chunker_matches else []
        lexical_ready = (
            document.status in {"indexed_lexical", "index_failed", "ready"} and chunker_matches and bool(chunks)
        )
        semantic_ready = lexical_ready if self.semantic_index is None else (
            document.status == "ready"
            and document.embedding_model == self.semantic_index.model_name
            and lexical_ready
        )
        metadata_ready = self.metadata_extractor is None or (
            document.metadata_status == "ready"
            and document.metadata_model == self.metadata_extractor.version
        )
        if semantic_ready and metadata_ready:
            return document

        try:
            if not lexical_ready:
                self.documents.set_status(document.id, "processing")
                if not self.extractions.has_pages(document.id):
                    pages = self.extractor.extract(self.catalog.pdf_store.path_for(document.id))
                    self.extractions.replace_pages(document.id, pages)
                chunks = self.chunks.rebuild(document.id)
                lexical_ready = True
            if self.semantic_index is not None and not semantic_ready:
                self.semantic_index.replace_document(document.id, chunks)
                self.documents.mark_ready(document.id, self.semantic_index.model_name, chunker_version)
        except Exception as error:
            failure_status = "index_failed" if lexical_ready and self.semantic_index is not None else "failed"
            self.documents.set_status(document.id, failure_status, str(error))
            raise

        if self.metadata_repository is not None and self.metadata_extractor is not None and not metadata_ready:
            self.metadata_repository.mark_processing(document.id)
            try:
                mentions = self.metadata_extractor.extract(self.metadata_repository.list_pages(document.id))
                self.metadata_repository.replace_document(document.id, self.metadata_extractor.version, mentions)
            except Exception as error:
                self.metadata_repository.mark_failed(document.id, self.metadata_extractor.version, str(error))

        completed = self.documents.get(document.id)
        if completed is None:
            raise RuntimeError("Ingested document could not be loaded")
        return completed

from io import BytesIO

import pytest
from docintel.chunking import ChunkRepository, ProvenanceChunker
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository, PdfExtractionError
from docintel.ingestion import IngestionService
from docintel.metadata import EntityMention, MetadataRepository
from docintel.storage import PdfStore


class FakeExtractor:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        pages: list[ExtractedPage] | None = None,
    ) -> None:
        self.calls = 0
        self.error = error
        self.pages = pages

    def extract(self, path) -> list[ExtractedPage]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.pages is not None:
            return self.pages
        return [
            ExtractedPage(
                1,
                612,
                792,
                "Persistent evidence",
                (ExtractedBlock(1, "Persistent evidence", (10, 10, 120, 30)),),
            )
        ]


class FakeSemanticIndex:
    model_name = "fake-embedding-model"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def replace_document(self, document_id: str, chunks) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


class FakeMetadataExtractor:
    def __init__(self, version: str = "fake-ner-v1", *, error: Exception | None = None) -> None:
        self.version = version
        self.calls = 0
        self.error = error

    def extract(self, pages):
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert pages[0].text == "Persistent evidence"
        return [EntityMention(1, "ORGANIZATION", "Persistent", 0, 10, 0.9)]


def _service(
    tmp_path,
    extractor: FakeExtractor,
    semantic_index: FakeSemanticIndex | None = None,
    chunker: ProvenanceChunker | None = None,
    metadata_extractor: FakeMetadataExtractor | None = None,
) -> IngestionService:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    documents = DocumentRepository(database_path)
    return IngestionService(
        DocumentCatalog(PdfStore(tmp_path), documents),
        documents,
        extractor,
        ExtractionRepository(database_path),
        ChunkRepository(database_path, chunker),
        semantic_index,
        MetadataRepository(database_path) if metadata_extractor is not None else None,
        metadata_extractor,
    )


def test_ingestion_builds_index_once_for_unchanged_pdf(tmp_path) -> None:
    extractor = FakeExtractor()
    service = _service(tmp_path, extractor)
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"

    first = service.ingest(BytesIO(pdf_bytes), "first.pdf")
    second = service.ingest(BytesIO(pdf_bytes), "second.pdf")

    assert first.id == second.id
    assert second.status == "indexed_lexical"
    assert extractor.calls == 1
    assert service.chunks.search("evidence")[0].page_start == 1


def test_ingestion_records_extraction_failure(tmp_path) -> None:
    extractor = FakeExtractor(error=RuntimeError("extraction failed"))
    service = _service(tmp_path, extractor)

    with pytest.raises(RuntimeError, match="extraction failed"):
        service.ingest(BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "broken.pdf")

    document_id = next((tmp_path / "pdfs").rglob("*.pdf")).stem
    document = service.documents.get(document_id)
    assert document is not None
    assert document.status == "failed"
    assert document.error_message == "extraction failed"


def test_ingestion_rejects_pdf_without_searchable_text(tmp_path) -> None:
    extractor = FakeExtractor(pages=[ExtractedPage(1, 612, 792, "", ())])
    service = _service(tmp_path, extractor)

    with pytest.raises(PdfExtractionError, match="no searchable text"):
        service.ingest(BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "empty.pdf")

    document_id = next((tmp_path / "pdfs").rglob("*.pdf")).stem
    document = service.documents.get(document_id)
    assert document is not None
    assert document.status == "failed"
    assert document.error_message == "PDF contains no searchable text"
    assert service.chunks.list_document(document_id) == []


def test_ingestion_records_semantic_index_and_reuses_ready_document(tmp_path) -> None:
    extractor = FakeExtractor()
    semantic_index = FakeSemanticIndex()
    service = _service(tmp_path, extractor, semantic_index)
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"

    first = service.ingest(BytesIO(pdf_bytes), "first.pdf")
    second = service.ingest(BytesIO(pdf_bytes), "second.pdf")

    assert first.status == second.status == "ready"
    assert first.embedding_model == "fake-embedding-model"
    assert extractor.calls == 1
    assert semantic_index.calls == 1


def test_ingestion_rebuilds_changed_chunker_without_reextracting(tmp_path) -> None:
    extractor = FakeExtractor()
    semantic_index = FakeSemanticIndex()
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"
    initial = _service(tmp_path, extractor, semantic_index)
    initial.ingest(BytesIO(pdf_bytes), "first.pdf")

    changed = _service(
        tmp_path,
        extractor,
        semantic_index,
        ProvenanceChunker(max_words=80, overlap=15),
    )
    rebuilt = changed.ingest(BytesIO(pdf_bytes), "second.pdf")

    assert extractor.calls == 1
    assert semantic_index.calls == 2
    assert rebuilt.chunker_version == "blocks-v1-80-15"


def test_ingestion_preserves_lexical_data_when_semantic_index_fails(tmp_path) -> None:
    extractor = FakeExtractor()
    semantic_index = FakeSemanticIndex(error=RuntimeError("index failed"))
    service = _service(tmp_path, extractor, semantic_index)

    with pytest.raises(RuntimeError, match="index failed"):
        service.ingest(BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "example.pdf")

    document_id = next((tmp_path / "pdfs").rglob("*.pdf")).stem
    document = service.documents.get(document_id)
    assert document is not None
    assert document.status == "index_failed"
    assert service.chunks.search("evidence")


def test_ingestion_enriches_once_and_reuses_current_metadata(tmp_path) -> None:
    extractor = FakeExtractor()
    semantic_index = FakeSemanticIndex()
    metadata_extractor = FakeMetadataExtractor()
    service = _service(tmp_path, extractor, semantic_index, metadata_extractor=metadata_extractor)
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"

    first = service.ingest(BytesIO(pdf_bytes), "first.pdf")
    second = service.ingest(BytesIO(pdf_bytes), "second.pdf")

    assert first.metadata_status == second.metadata_status == "ready"
    assert first.metadata_model == "fake-ner-v1"
    assert extractor.calls == 1
    assert semantic_index.calls == 1
    assert metadata_extractor.calls == 1


def test_changed_metadata_version_does_not_reextract_or_reembed(tmp_path) -> None:
    extractor = FakeExtractor()
    semantic_index = FakeSemanticIndex()
    initial_metadata = FakeMetadataExtractor("fake-ner-v1")
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"
    initial = _service(tmp_path, extractor, semantic_index, metadata_extractor=initial_metadata)
    initial.ingest(BytesIO(pdf_bytes), "first.pdf")

    changed_metadata = FakeMetadataExtractor("fake-ner-v2")
    changed = _service(tmp_path, extractor, semantic_index, metadata_extractor=changed_metadata)
    result = changed.ingest(BytesIO(pdf_bytes), "second.pdf")

    assert result.metadata_model == "fake-ner-v2"
    assert extractor.calls == 1
    assert semantic_index.calls == 1
    assert initial_metadata.calls == 1
    assert changed_metadata.calls == 1


def test_metadata_failure_preserves_ready_search_state(tmp_path) -> None:
    metadata_extractor = FakeMetadataExtractor(error=RuntimeError("NER unavailable"))
    service = _service(tmp_path, FakeExtractor(), FakeSemanticIndex(), metadata_extractor=metadata_extractor)

    result = service.ingest(BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "example.pdf")

    assert result.status == "ready"
    assert result.metadata_status == "failed"
    assert result.metadata_error == "NER unavailable"
    assert service.chunks.search("evidence")

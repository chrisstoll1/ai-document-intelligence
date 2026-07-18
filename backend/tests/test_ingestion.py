from io import BytesIO

import pytest
from docintel.chunking import ChunkRepository
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.ingestion import IngestionService
from docintel.storage import PdfStore


class FakeExtractor:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    def extract(self, path) -> list[ExtractedPage]:
        self.calls += 1
        if self.error is not None:
            raise self.error
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


def _service(
    tmp_path, extractor: FakeExtractor, semantic_index: FakeSemanticIndex | None = None
) -> IngestionService:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    documents = DocumentRepository(database_path)
    return IngestionService(
        DocumentCatalog(PdfStore(tmp_path), documents),
        documents,
        extractor,
        ExtractionRepository(database_path),
        ChunkRepository(database_path),
        semantic_index,
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

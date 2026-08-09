from io import BytesIO

from docintel.api import AppServices, create_app
from docintel.chunking import ChunkRepository
from docintel.config import Settings
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.generation import GroundedAnswer, GroundedClaim, GroundingContext
from docintel.indexing import ChromaSemanticIndex
from docintel.ingestion import IngestionService
from docintel.lifecycle import DocumentLifecycleService
from docintel.metadata import EntityMention
from docintel.search import HybridSearchService, PersistentSearchResult
from docintel.storage import PdfStore
from fastapi.testclient import TestClient


class FakeIngestion:
    def ingest(self, source, filename: str) -> DocumentRecord:
        assert source.read() == b"%PDF-1.7\nexample\n%%EOF"
        return DocumentRecord(
            "a" * 64,
            "pdfs/aa/file.pdf",
            "application/pdf",
            26,
            "ready",
            None,
            "fake",
            "blocks-v1-120-20",
        )


class FakeDocuments:
    def list_all(self):
        return []

    def get(self, document_id: str):
        if document_id != "a" * 64:
            return None
        return FakeIngestion().ingest(BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "example.pdf")


class FakeMetadata:
    def list_document(self, document_id: str):
        assert document_id == "a" * 64
        return [EntityMention(1, "ORGANIZATION", "Example Ltd", 0, 11, 0.9)]


class FakeSearch:
    def search(self, query: str, *, limit: int = 5) -> list[PersistentSearchResult]:
        return [
            PersistentSearchResult(
                "b" * 64,
                "a" * 64,
                "example.pdf",
                "Matching evidence",
                1,
                1,
                0.01,
                1,
                2,
            )
        ][:limit]


class FakeGeneration:
    def answer(self, query: str, *, limit: int = 5) -> GroundedAnswer:
        assert query == "What is supported?"
        context = GroundingContext("C1", FakeSearch().search(query, limit=limit)[0])
        return GroundedAnswer(
            status="answered",
            answer="Matching evidence supports the answer.",
            claims=(GroundedClaim("Matching evidence supports the answer.", ("C1",)),),
            contexts=(context,),
        )


class FakeLifecycle:
    def delete(self, document_id: str) -> bool:
        return document_id == "a" * 64

    def reset(self) -> int:
        return 2


class FakePageExtractor:
    def extract(self, path) -> list[ExtractedPage]:
        assert path.exists()
        return [
            ExtractedPage(
                1,
                612,
                792,
                "Personal data privacy controls",
                (ExtractedBlock(1, "Personal data privacy controls", (10, 10, 180, 30)),),
            )
        ]


class FakeEncoder:
    def encode_documents(self, documents: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        return [
            [1.0, 0.0] if any(term in sentence.lower() for term in ("privacy", "confidential")) else [0.0, 1.0]
            for sentence in documents
        ]

    def encode_query(self, query: str, *, normalize_embeddings: bool) -> list[float]:
        return self.encode_documents([query], normalize_embeddings=normalize_embeddings)[0]


def _services(settings: Settings) -> AppServices:
    return AppServices(
        FakeIngestion(),
        FakeDocuments(),
        object(),
        FakeSearch(),
        metadata=FakeMetadata(),
        generation=FakeGeneration(),
        lifecycle=FakeLifecycle(),
    )


def _integration_services(settings: Settings) -> AppServices:
    initialize_database(settings.database_path)
    documents = DocumentRepository(settings.database_path)
    pdf_store = PdfStore(settings.data_dir)
    chunks = ChunkRepository(settings.database_path)
    semantic_index = ChromaSemanticIndex(
        settings.data_dir / "chroma", encoder=FakeEncoder(), model_name="fake-model"
    )
    search = HybridSearchService(settings.database_path, chunks, semantic_index)
    return AppServices(
        IngestionService(
            DocumentCatalog(pdf_store, documents),
            documents,
            FakePageExtractor(),
            ExtractionRepository(settings.database_path),
            chunks,
            semantic_index,
        ),
        documents,
        pdf_store,
        search,
        semantic_index,
        lifecycle=DocumentLifecycleService(documents, pdf_store, semantic_index),
    )


def test_health_endpoint_reports_ok(tmp_path) -> None:
    with TestClient(create_app(Settings(tmp_path), service_builder=_services)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_application_lifespan_initializes_local_stores(tmp_path) -> None:
    with TestClient(create_app(Settings(tmp_path))) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert (tmp_path / "docintel.sqlite3").exists()
    assert (tmp_path / "chroma").is_dir()


def test_upload_and_search_endpoints_return_typed_results(tmp_path) -> None:
    with TestClient(create_app(Settings(tmp_path), service_builder=_services)) as client:
        upload = client.post(
            "/api/documents",
            files={"upload": ("example.pdf", BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "application/pdf")},
        )
        search = client.post("/api/search", json={"query": "evidence", "limit": 3})
        answer = client.post("/api/answer", json={"query": "What is supported?"})
        metadata = client.get(f"/api/documents/{'a' * 64}/metadata")

    assert upload.status_code == 200
    assert upload.json()["status"] == "ready"
    assert upload.json()["metadata_status"] == "pending"
    assert search.status_code == 200
    assert search.json()[0]["document_name"] == "example.pdf"
    assert search.json()[0]["keyword_rank"] == 1
    assert answer.status_code == 200
    assert answer.json()["status"] == "answered"
    assert answer.json()["claims"][0]["citation_ids"] == ["C1"]
    assert answer.json()["contexts"][0]["document_name"] == "example.pdf"
    assert metadata.status_code == 200
    assert metadata.json()["entities"][0]["text"] == "Example Ltd"


def test_answer_endpoint_reports_unavailable_without_generator(tmp_path) -> None:
    services = _services(Settings(tmp_path))

    def without_generation(settings: Settings) -> AppServices:
        return AppServices(
            services.ingestion,
            services.documents,
            services.pdf_store,
            services.search,
            metadata=services.metadata,
        )

    with TestClient(create_app(Settings(tmp_path), service_builder=without_generation)) as client:
        response = client.post("/api/answer", json={"query": "What is supported?"})

    assert response.status_code == 503


def test_document_delete_and_reset_endpoints_report_results(tmp_path) -> None:
    with TestClient(create_app(Settings(tmp_path), service_builder=_services)) as client:
        deleted = client.delete(f"/api/documents/{'a' * 64}")
        missing = client.delete(f"/api/documents/{'b' * 64}")
        reset = client.delete("/api/documents")

    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert reset.json() == {"deleted_count": 2}


def test_api_vertical_slice_persists_and_searches_uploaded_pdf(tmp_path) -> None:
    pdf_bytes = b"%PDF-1.7\nexample\n%%EOF"
    with TestClient(create_app(Settings(tmp_path), service_builder=_integration_services)) as client:
        upload = client.post(
            "/api/documents",
            files={"upload": ("privacy.pdf", BytesIO(pdf_bytes), "application/pdf")},
        )
        document_id = upload.json()["id"]
        listing = client.get("/api/documents")
        detail = client.get(f"/api/documents/{document_id}")
        source = client.get(f"/api/documents/{document_id}/pdf")
        search = client.post("/api/search", json={"query": "confidential information"})
        deleted = client.delete(f"/api/documents/{document_id}")
        missing_detail = client.get(f"/api/documents/{document_id}")
        search_after_delete = client.post("/api/search", json={"query": "confidential information"})
        reset = client.delete("/api/documents")

    assert upload.json()["status"] == "ready"
    assert listing.json()[0]["filename"] == "privacy.pdf"
    assert detail.json()["embedding_model"] == "fake-model"
    assert source.content == pdf_bytes
    assert search.json()[0]["document_id"] == document_id
    assert search.json()[0]["semantic_rank"] == 1
    assert deleted.status_code == 204
    assert missing_detail.status_code == 404
    assert search_after_delete.json() == []
    assert reset.json() == {"deleted_count": 0}

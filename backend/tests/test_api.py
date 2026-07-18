from io import BytesIO

from docintel.api import AppServices, create_app
from docintel.chunking import ChunkRepository
from docintel.config import Settings
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRecord, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.indexing import ChromaSemanticIndex
from docintel.ingestion import IngestionService
from docintel.search import HybridSearchService, PersistentSearchResult
from docintel.storage import PdfStore
from fastapi.testclient import TestClient


class FakeIngestion:
    def ingest(self, source, filename: str) -> DocumentRecord:
        assert source.read() == b"%PDF-1.7\nexample\n%%EOF"
        return DocumentRecord("a" * 64, "pdfs/aa/file.pdf", "application/pdf", 26, "ready", None, "fake")


class FakeDocuments:
    def list_all(self):
        return []


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
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        return [
            [1.0, 0.0] if any(term in sentence.lower() for term in ("privacy", "confidential")) else [0.0, 1.0]
            for sentence in sentences
        ]


def _services(settings: Settings) -> AppServices:
    return AppServices(FakeIngestion(), FakeDocuments(), object(), FakeSearch())


def _integration_services(settings: Settings) -> AppServices:
    initialize_database(settings.database_path)
    documents = DocumentRepository(settings.database_path)
    pdf_store = PdfStore(settings.data_dir)
    chunks = ChunkRepository(settings.database_path)
    semantic_index = ChromaSemanticIndex(
        settings.data_dir / "chroma", encoder=FakeEncoder(), model_name="fake-model"
    )
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
        HybridSearchService(settings.database_path, chunks, semantic_index),
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

    assert upload.status_code == 200
    assert upload.json()["status"] == "ready"
    assert search.status_code == 200
    assert search.json()[0]["document_name"] == "example.pdf"
    assert search.json()[0]["keyword_rank"] == 1


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

    assert upload.json()["status"] == "ready"
    assert listing.json()[0]["filename"] == "privacy.pdf"
    assert detail.json()["embedding_model"] == "fake-model"
    assert source.content == pdf_bytes
    assert search.json()[0]["document_id"] == document_id
    assert search.json()[0]["semantic_rank"] == 1

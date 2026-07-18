from io import BytesIO

from docintel.api import AppServices, create_app
from docintel.config import Settings
from docintel.documents import DocumentRecord
from docintel.search import PersistentSearchResult
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


def _services(settings: Settings) -> AppServices:
    return AppServices(FakeIngestion(), FakeDocuments(), object(), FakeSearch())


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

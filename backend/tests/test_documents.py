from io import BytesIO

from docintel.db import database_connection, initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.storage import PdfStore


def test_document_catalog_persists_pdf_and_upload_names(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    catalog = DocumentCatalog(PdfStore(tmp_path), DocumentRepository(database_path))
    pdf_bytes = b"%PDF-1.7\npersisted document\n%%EOF"

    first = catalog.add_pdf(BytesIO(pdf_bytes), "first.pdf")
    second = catalog.add_pdf(BytesIO(pdf_bytes), "folder/second.pdf")

    with database_connection(database_path) as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        names = [
            row["original_filename"]
            for row in connection.execute(
                "SELECT original_filename FROM document_names ORDER BY id"
            ).fetchall()
        ]

    assert first == second
    assert first.status == "stored"
    assert document_count == 1
    assert names == ["first.pdf", "second.pdf"]


def test_document_repository_returns_none_for_unknown_document(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)

    assert DocumentRepository(database_path).get("missing") is None


def test_document_repository_deletes_records_and_reset_is_idempotent(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    repository = DocumentRepository(database_path)
    catalog = DocumentCatalog(PdfStore(tmp_path), repository)
    first = catalog.add_pdf(BytesIO(b"%PDF-1.7\nfirst\n%%EOF"), "first.pdf")
    catalog.add_pdf(BytesIO(b"%PDF-1.7\nsecond\n%%EOF"), "second.pdf")

    assert repository.delete(first.id) is True
    assert repository.delete(first.id) is False
    assert len(repository.list_all()) == 1
    assert repository.delete_all() == 1
    assert repository.delete_all() == 0

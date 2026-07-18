from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from docintel.db import database_connection
from docintel.storage import PdfStore, StoredPdf


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    storage_key: str
    media_type: str
    size_bytes: int
    status: str
    error_message: str | None
    embedding_model: str | None


class DocumentRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def record_stored_pdf(self, stored: StoredPdf, original_filename: str) -> DocumentRecord:
        filename = Path(original_filename.strip()).name
        if not filename:
            raise ValueError("original_filename must not be empty")

        with database_connection(self.database_path) as connection, connection:
            connection.execute(
                """
                INSERT INTO documents (id, storage_key, media_type, size_bytes)
                VALUES (?, ?, 'application/pdf', ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (stored.document_id, stored.storage_key, stored.size_bytes),
            )
            connection.execute(
                "INSERT INTO document_names (document_id, original_filename) VALUES (?, ?)",
                (stored.document_id, filename),
            )
        record = self.get(stored.document_id)
        if record is None:
            raise RuntimeError("Stored document was not persisted")
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, storage_key, media_type, size_bytes, status, error_message, embedding_model
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        return DocumentRecord(**dict(row)) if row is not None else None

    def set_status(self, document_id: str, status: str, error_message: str | None = None) -> None:
        with database_connection(self.database_path) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, error_message, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown document: {document_id}")

    def mark_ready(self, document_id: str, embedding_model: str) -> None:
        with database_connection(self.database_path) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = 'ready', error_message = NULL, embedding_model = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (embedding_model, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown document: {document_id}")


class DocumentCatalog:
    def __init__(self, pdf_store: PdfStore, repository: DocumentRepository) -> None:
        self.pdf_store = pdf_store
        self.repository = repository

    def add_pdf(self, source: BinaryIO, original_filename: str) -> DocumentRecord:
        stored = self.pdf_store.put(source)
        return self.repository.record_stored_pdf(stored, original_filename)

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
    chunker_version: str | None
    metadata_status: str = "pending"
    metadata_model: str | None = None
    metadata_error: str | None = None


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
                SELECT id, storage_key, media_type, size_bytes, status, error_message,
                       embedding_model, chunker_version, metadata_status, metadata_model, metadata_error
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

    def mark_ready(self, document_id: str, embedding_model: str, chunker_version: str) -> None:
        with database_connection(self.database_path) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET status = 'ready', error_message = NULL, embedding_model = ?, chunker_version = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (embedding_model, chunker_version, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown document: {document_id}")

    def list_all(self) -> list[tuple[DocumentRecord, str]]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT documents.id, documents.storage_key, documents.media_type,
                       documents.size_bytes, documents.status, documents.error_message,
                        documents.embedding_model, documents.chunker_version,
                        documents.metadata_status, documents.metadata_model, documents.metadata_error,
                       COALESCE(
                           (SELECT original_filename FROM document_names
                            WHERE document_id = documents.id ORDER BY id DESC LIMIT 1),
                           documents.id
                       ) AS original_filename
                FROM documents
                ORDER BY documents.created_at DESC, documents.id
                """
            ).fetchall()
        return [
            (
                DocumentRecord(
                    id=row["id"],
                    storage_key=row["storage_key"],
                    media_type=row["media_type"],
                    size_bytes=row["size_bytes"],
                    status=row["status"],
                    error_message=row["error_message"],
                    embedding_model=row["embedding_model"],
                    chunker_version=row["chunker_version"],
                    metadata_status=row["metadata_status"],
                    metadata_model=row["metadata_model"],
                    metadata_error=row["metadata_error"],
                ),
                row["original_filename"],
            )
            for row in rows
        ]

    def latest_name(self, document_id: str) -> str | None:
        with database_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT original_filename
                FROM document_names
                WHERE document_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return row["original_filename"] if row is not None else None

    def delete(self, document_id: str) -> bool:
        with database_connection(self.database_path) as connection, connection:
            connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE document_id = ?)",
                (document_id,),
            )
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def delete_all(self) -> int:
        with database_connection(self.database_path) as connection, connection:
            connection.execute("DELETE FROM chunks_fts")
            cursor = connection.execute("DELETE FROM documents")
        return cursor.rowcount


class DocumentCatalog:
    def __init__(self, pdf_store: PdfStore, repository: DocumentRepository) -> None:
        self.pdf_store = pdf_store
        self.repository = repository

    def add_pdf(self, source: BinaryIO, original_filename: str) -> DocumentRecord:
        stored = self.pdf_store.put(source)
        return self.repository.record_stored_pdf(stored, original_filename)

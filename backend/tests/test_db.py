import sqlite3

import pytest
from docintel.db import SCHEMA_V1_SQL, SCHEMA_VERSION, connect_database, initialize_database


def test_initialize_database_creates_schema_and_preserves_existing_rows(tmp_path) -> None:
    database_path = tmp_path / "nested" / "docintel.sqlite3"

    initialize_database(database_path)
    with connect_database(database_path) as connection:
        table_names = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        connection.execute(
            "INSERT INTO documents (id, storage_key, media_type, size_bytes) VALUES (?, ?, ?, ?)",
            ("a" * 64, "pdfs/aa/document.pdf", "application/pdf", 100),
        )

    initialize_database(database_path)
    with connect_database(database_path) as connection:
        document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert {"documents", "document_names", "pages", "blocks"} <= table_names
    assert document_count == 1
    assert schema_version == SCHEMA_VERSION
    assert journal_mode == "delete"


def test_document_names_require_an_existing_document(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)

    with connect_database(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO document_names (document_id, original_filename) VALUES (?, ?)",
                ("missing", "missing.pdf"),
            )


def test_initialize_database_upgrades_version_one_schema(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    with connect_database(database_path) as connection:
        connection.executescript(f"{SCHEMA_V1_SQL}\nPRAGMA user_version = 1;")

    initialize_database(database_path)

    with connect_database(database_path) as connection:
        schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        page_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'pages'"
        ).fetchone()
    assert schema_version == SCHEMA_VERSION
    assert page_table["name"] == "pages"

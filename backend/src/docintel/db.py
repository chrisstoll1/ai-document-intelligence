from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 4

SCHEMA_V1_SQL = """
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    status TEXT NOT NULL DEFAULT 'stored',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_names (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_V2_SQL = """
CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    width REAL NOT NULL CHECK (width > 0),
    height REAL NOT NULL CHECK (height > 0),
    text TEXT NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('direct', 'ocr')),
    UNIQUE (document_id, page_number)
);

CREATE TABLE blocks (
    id INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    block_order INTEGER NOT NULL CHECK (block_order > 0),
    text TEXT NOT NULL,
    x0 REAL NOT NULL,
    top REAL NOT NULL,
    x1 REAL NOT NULL,
    bottom REAL NOT NULL,
    method TEXT NOT NULL CHECK (method IN ('direct', 'ocr')),
    confidence REAL,
    UNIQUE (page_id, block_order)
);
"""

SCHEMA_V3_SQL = """
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    text TEXT NOT NULL,
    page_start INTEGER NOT NULL CHECK (page_start > 0),
    page_end INTEGER NOT NULL CHECK (page_end >= page_start),
    chunker_version TEXT NOT NULL,
    UNIQUE (document_id, ordinal, chunker_version)
);

CREATE TABLE chunk_spans (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    block_id INTEGER NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    span_order INTEGER NOT NULL CHECK (span_order > 0),
    block_start INTEGER NOT NULL CHECK (block_start >= 0),
    block_end INTEGER NOT NULL CHECK (block_end >= block_start),
    chunk_start INTEGER NOT NULL CHECK (chunk_start >= 0),
    chunk_end INTEGER NOT NULL CHECK (chunk_end >= chunk_start),
    PRIMARY KEY (chunk_id, span_order)
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);
"""

SCHEMA_V4_SQL = """
ALTER TABLE documents ADD COLUMN embedding_model TEXT;
"""

MIGRATIONS = {
    1: SCHEMA_V1_SQL,
    2: SCHEMA_V2_SQL,
    3: SCHEMA_V3_SQL,
    4: SCHEMA_V4_SQL,
}


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


@contextmanager
def database_connection(path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect_database(path)
    try:
        yield connection
    finally:
        connection.close()


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = connect_database(path)
    try:
        current_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than supported version {SCHEMA_VERSION}"
            )
        if current_version == SCHEMA_VERSION:
            return

        connection.execute("PRAGMA journal_mode = DELETE")
        for target_version in range(current_version + 1, SCHEMA_VERSION + 1):
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                f"{MIGRATIONS[target_version]}\n"
                f"PRAGMA user_version = {target_version};\n"
                "COMMIT;"
            )
    finally:
        connection.close()

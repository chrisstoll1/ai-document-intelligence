from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from docintel.db import database_connection
from docintel.ner_evaluation import ENTITY_LABELS


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class EntityMention:
    page_number: int
    label: str
    text: str
    char_start: int
    char_end: int
    confidence: float | None = None
    normalized_text: str | None = None

    def __post_init__(self) -> None:
        if self.page_number <= 0:
            raise ValueError("page_number must be positive")
        if self.label not in ENTITY_LABELS:
            raise ValueError(f"Unsupported entity label: {self.label}")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("Entity character offsets are invalid")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("Entity confidence must be between zero and one")
        if self.normalized_text is None:
            object.__setattr__(self, "normalized_text", normalize_entity_text(self.text))


@dataclass(frozen=True)
class MetadataPageMatch:
    document_id: str
    page_number: int
    label: str
    normalized_text: str


def normalize_entity_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class EntityExtractor(Protocol):
    version: str

    def extract(self, pages: Sequence[PageText]) -> Sequence[EntityMention]: ...


class SpacyEntityExtractor:
    LABEL_MAP = {
        "PERSON": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
    }

    def __init__(
        self,
        model_name: str = "en_core_web_trf",
        model_version: str = "3.8.0",
        *,
        pipeline=None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.version = f"spacy:{model_name}@{model_version}:labels-v1:normalization-v1"
        self._pipeline = pipeline

    def extract(self, pages: Sequence[PageText]) -> list[EntityMention]:
        pipeline = self._get_pipeline()
        mentions = []
        for page in pages:
            document = pipeline(page.text)
            mentions.extend(
                EntityMention(
                    page_number=page.page_number,
                    label=self.LABEL_MAP[entity.label_],
                    text=entity.text,
                    char_start=entity.start_char,
                    char_end=entity.end_char,
                )
                for entity in document.ents
                if entity.label_ in self.LABEL_MAP
            )
        return mentions

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                import spacy
            except ImportError as error:
                raise RuntimeError("The selected spaCy NER runtime is not installed") from error
            self._pipeline = spacy.load(self.model_name)
        return self._pipeline


class MetadataRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def list_pages(self, document_id: str) -> list[PageText]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                "SELECT page_number, text FROM pages WHERE document_id = ? ORDER BY page_number",
                (document_id,),
            ).fetchall()
        return [PageText(row["page_number"], row["text"]) for row in rows]

    def list_document(self, document_id: str) -> list[EntityMention]:
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT page_number, label, text, char_start, char_end, confidence, normalized_text
                FROM entity_mentions
                WHERE document_id = ?
                ORDER BY page_number, char_start, char_end, label
                """,
                (document_id,),
            ).fetchall()
        return [EntityMention(**dict(row)) for row in rows]

    def find_exact_pages(
        self,
        entity_keys: Sequence[tuple[str, str]],
        *,
        model_version: str,
    ) -> list[MetadataPageMatch]:
        unique_keys = sorted(set(entity_keys))
        if not unique_keys:
            return []
        conditions = " OR ".join(
            "(entity_mentions.label = ? AND entity_mentions.normalized_text = ?)" for _ in unique_keys
        )
        parameters = [value for key in unique_keys for value in key]
        with database_connection(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT entity_mentions.document_id, entity_mentions.page_number,
                                entity_mentions.label, entity_mentions.normalized_text
                FROM entity_mentions
                JOIN documents ON documents.id = entity_mentions.document_id
                WHERE documents.metadata_status = 'ready'
                  AND documents.metadata_model = ?
                  AND ({conditions})
                ORDER BY entity_mentions.document_id, entity_mentions.page_number,
                         entity_mentions.label, entity_mentions.normalized_text
                """,
                [model_version, *parameters],
            ).fetchall()
        return [MetadataPageMatch(**dict(row)) for row in rows]

    def mark_processing(self, document_id: str) -> None:
        self._set_status(document_id, "processing")

    def mark_failed(self, document_id: str, model_version: str, error: str) -> None:
        self._set_status(document_id, "failed", model_version=model_version, error=error)

    def replace_document(
        self,
        document_id: str,
        model_version: str,
        mentions: Sequence[EntityMention],
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version must not be empty")
        with database_connection(self.database_path) as connection, connection:
            page_rows = connection.execute(
                "SELECT page_number, text FROM pages WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            if not page_rows:
                raise ValueError(f"Document has no extracted pages: {document_id}")
            page_text = {row["page_number"]: row["text"] for row in page_rows}
            for mention in mentions:
                source_text = page_text.get(mention.page_number)
                if source_text is None:
                    raise ValueError(f"Unknown page {mention.page_number} for document {document_id}")
                if source_text[mention.char_start : mention.char_end] != mention.text:
                    raise ValueError("Entity mention text does not match the persisted page offsets")

            connection.execute("DELETE FROM entity_mentions WHERE document_id = ?", (document_id,))
            connection.executemany(
                """
                INSERT INTO entity_mentions (
                    document_id, page_number, label, text, normalized_text,
                    char_start, char_end, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        mention.page_number,
                        mention.label,
                        mention.text,
                        mention.normalized_text,
                        mention.char_start,
                        mention.char_end,
                        mention.confidence,
                    )
                    for mention in mentions
                ],
            )
            cursor = connection.execute(
                """
                UPDATE documents
                SET metadata_status = 'ready', metadata_model = ?, metadata_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (model_version, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown document: {document_id}")

    def _set_status(
        self,
        document_id: str,
        status: str,
        *,
        model_version: str | None = None,
        error: str | None = None,
    ) -> None:
        with database_connection(self.database_path) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET metadata_status = ?, metadata_model = COALESCE(?, metadata_model),
                    metadata_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (status, model_version, error, document_id),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Unknown document: {document_id}")

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CHUNK_MAX_WORDS = 120
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_NER_MODEL = "en_core_web_trf"
DEFAULT_NER_MODEL_VERSION = "3.8.0"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = DEFAULT_EMBEDDING_REVISION
    embedding_query_prompt: str | None = None
    chunk_max_words: int = DEFAULT_CHUNK_MAX_WORDS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ner_model: str = DEFAULT_NER_MODEL
    ner_model_version: str = DEFAULT_NER_MODEL_VERSION

    @property
    def database_path(self) -> Path:
        return self.data_dir / "docintel.sqlite3"

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("DOCINTEL_DATA_DIR", "data")),
            embedding_model=os.environ.get("DOCINTEL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_revision=os.environ.get("DOCINTEL_EMBEDDING_REVISION", DEFAULT_EMBEDDING_REVISION),
            embedding_query_prompt=os.environ.get("DOCINTEL_EMBEDDING_QUERY_PROMPT"),
            chunk_max_words=int(os.environ.get("DOCINTEL_CHUNK_MAX_WORDS", DEFAULT_CHUNK_MAX_WORDS)),
            chunk_overlap=int(os.environ.get("DOCINTEL_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
            ner_model=os.environ.get("DOCINTEL_NER_MODEL", DEFAULT_NER_MODEL),
            ner_model_version=os.environ.get("DOCINTEL_NER_MODEL_VERSION", DEFAULT_NER_MODEL_VERSION),
        )

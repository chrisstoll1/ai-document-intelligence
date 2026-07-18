from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @property
    def database_path(self) -> Path:
        return self.data_dir / "docintel.sqlite3"

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            data_dir=Path(os.environ.get("DOCINTEL_DATA_DIR", "data")),
            embedding_model=os.environ.get("DOCINTEL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )

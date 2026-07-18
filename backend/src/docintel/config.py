from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path

    @property
    def database_path(self) -> Path:
        return self.data_dir / "docintel.sqlite3"

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(data_dir=Path(os.environ.get("DOCINTEL_DATA_DIR", "data")))

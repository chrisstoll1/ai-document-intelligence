from __future__ import annotations

from pathlib import Path

from .models import Document


TEXT_EXTENSIONS = {".txt", ".md"}


class DocumentLoader:
    def __init__(self, extensions: set[str] | None = None) -> None:
        self.extensions = extensions or TEXT_EXTENSIONS

    def load_folder(self, folder: Path) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(folder.rglob("*")):
            if not self._can_read(path):
                continue
            documents.append(self._read_document(path))
        return documents

    def _can_read(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.extensions

    def _read_document(self, path: Path) -> Document:
        return Document(
            id=path.stem.lower().replace(" ", "-"),
            title=path.stem.replace("_", " ").title(),
            text=path.read_text(encoding="utf-8"),
            source_path=str(path),
        )


def load_documents(folder: Path) -> list[Document]:
    return DocumentLoader().load_folder(folder)

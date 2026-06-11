from __future__ import annotations

from pathlib import Path

from .models import Document


TEXT_EXTENSIONS = {".txt", ".md"}


def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem.lower().replace(" ", "-"),
                title=path.stem.replace("_", " ").title(),
                text=text,
                source_path=str(path),
            )
        )
    return documents

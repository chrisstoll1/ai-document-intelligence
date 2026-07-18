from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

PDF_HEADER = b"%PDF-"
READ_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class InvalidPdfError(ValueError):
    pass


class PdfTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class StoredPdf:
    document_id: str
    storage_key: str
    path: Path
    size_bytes: int
    already_exists: bool


class PdfStore:
    def __init__(self, data_dir: Path, *, max_bytes: int = 50 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.data_dir = data_dir
        self.pdf_dir = data_dir / "pdfs"
        self.max_bytes = max_bytes

    def put(self, source: BinaryIO) -> StoredPdf:
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self._write_temporary(source)
        try:
            document_id, size_bytes = self._inspect(temporary_path)
            destination = self.path_for(document_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            already_exists = destination.exists()
            if already_exists:
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
            return StoredPdf(
                document_id=document_id,
                storage_key=destination.relative_to(self.data_dir).as_posix(),
                path=destination,
                size_bytes=size_bytes,
                already_exists=already_exists,
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def path_for(self, document_id: str) -> Path:
        if SHA256_RE.fullmatch(document_id) is None:
            raise ValueError("document_id must be a lowercase SHA-256 digest")
        return self.pdf_dir / document_id[:2] / f"{document_id}.pdf"

    def _write_temporary(self, source: BinaryIO) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(dir=self.pdf_dir, suffix=".tmp", delete=False)
        temporary_path = Path(temporary_file.name)
        try:
            with temporary_file:
                size_bytes = 0
                while chunk := source.read(READ_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > self.max_bytes:
                        raise PdfTooLargeError(f"PDF exceeds the {self.max_bytes}-byte limit")
                    temporary_file.write(chunk)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    @staticmethod
    def _inspect(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as stored_file:
            header = stored_file.read(len(PDF_HEADER))
            if header != PDF_HEADER:
                raise InvalidPdfError("File does not start with a PDF header")
            digest.update(header)
            size_bytes += len(header)
            while chunk := stored_file.read(READ_SIZE):
                digest.update(chunk)
                size_bytes += len(chunk)
        return digest.hexdigest(), size_bytes

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

LINE_TOLERANCE = 3.0


class PdfExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractedBlock:
    order: int
    text: str
    bbox: tuple[float, float, float, float]
    method: str = "direct"


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    width: float
    height: float
    text: str
    blocks: tuple[ExtractedBlock, ...]
    method: str = "direct"


class PdfExtractor:
    def extract(self, path: Path) -> list[ExtractedPage]:
        try:
            with pdfplumber.open(path, unicode_norm="NFKC") as document:
                return [self._extract_page(page) for page in document.pages]
        except Exception as error:
            raise PdfExtractionError(f"Could not extract PDF: {path.name}") from error

    @staticmethod
    def _extract_page(page) -> ExtractedPage:
        words = sorted(page.extract_words(), key=lambda word: (float(word["top"]), float(word["x0"])))
        lines: list[list[dict]] = []
        for word in words:
            if not lines or abs(float(word["top"]) - float(lines[-1][0]["top"])) > LINE_TOLERANCE:
                lines.append([word])
            else:
                lines[-1].append(word)

        blocks = tuple(
            ExtractedBlock(
                order=order,
                text=" ".join(str(word["text"]) for word in line),
                bbox=(
                    min(float(word["x0"]) for word in line),
                    min(float(word["top"]) for word in line),
                    max(float(word["x1"]) for word in line),
                    max(float(word["bottom"]) for word in line),
                ),
            )
            for order, line in enumerate(lines, start=1)
        )
        return ExtractedPage(
            page_number=page.page_number,
            width=float(page.width),
            height=float(page.height),
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
        )

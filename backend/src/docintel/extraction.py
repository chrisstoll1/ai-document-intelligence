from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pdfplumber
import pytesseract
from PIL import Image

LINE_TOLERANCE = 3.0
MINIMUM_DIRECT_CHARACTERS = 20
OCR_DPI = 300


class PdfExtractionError(RuntimeError):
    pass


class OcrUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_key: tuple[int, int, int]


class OcrEngine(Protocol):
    def recognize(self, image: Image.Image) -> Sequence[OcrWord]: ...


class TesseractOcrEngine:
    def __init__(self, *, language: str = "eng") -> None:
        self.language = language

    def recognize(self, image: Image.Image) -> list[OcrWord]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                config="--psm 3",
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as error:
            raise OcrUnavailableError("Tesseract is required to process image-only PDF pages") from error

        words = []
        for index, value in enumerate(data["text"]):
            text = value.strip()
            confidence = float(data["conf"][index])
            if not text or confidence < 0:
                continue
            words.append(
                OcrWord(
                    text=text,
                    left=int(data["left"][index]),
                    top=int(data["top"][index]),
                    width=int(data["width"][index]),
                    height=int(data["height"][index]),
                    confidence=confidence,
                    line_key=(
                        int(data["block_num"][index]),
                        int(data["par_num"][index]),
                        int(data["line_num"][index]),
                    ),
                )
            )
        return words


@dataclass(frozen=True)
class ExtractedBlock:
    order: int
    text: str
    bbox: tuple[float, float, float, float]
    method: str = "direct"
    confidence: float | None = None


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    width: float
    height: float
    text: str
    blocks: tuple[ExtractedBlock, ...]
    method: str = "direct"


class PdfExtractor:
    def __init__(
        self,
        *,
        ocr_engine: OcrEngine | None = None,
        minimum_direct_characters: int = MINIMUM_DIRECT_CHARACTERS,
        ocr_dpi: int = OCR_DPI,
    ) -> None:
        if minimum_direct_characters < 0:
            raise ValueError("minimum_direct_characters must not be negative")
        if ocr_dpi <= 0:
            raise ValueError("ocr_dpi must be positive")
        self.ocr_engine = ocr_engine or TesseractOcrEngine()
        self.minimum_direct_characters = minimum_direct_characters
        self.ocr_dpi = ocr_dpi

    def extract(self, path: Path) -> list[ExtractedPage]:
        try:
            with pdfplumber.open(path, unicode_norm="NFKC") as document:
                return [self._extract_page(page) for page in document.pages]
        except OcrUnavailableError:
            raise
        except Exception as error:
            raise PdfExtractionError(f"Could not extract PDF: {path.name}") from error

    def _extract_page(self, page) -> ExtractedPage:
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
        direct_page = ExtractedPage(
            page_number=page.page_number,
            width=float(page.width),
            height=float(page.height),
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
        )
        if sum(character.isalnum() for character in direct_page.text) >= self.minimum_direct_characters:
            return direct_page
        return self._extract_page_with_ocr(page)

    def _extract_page_with_ocr(self, page) -> ExtractedPage:
        image = page.to_image(resolution=self.ocr_dpi, antialias=True).original
        words_by_line: dict[tuple[int, int, int], list[OcrWord]] = defaultdict(list)
        for word in self.ocr_engine.recognize(image):
            words_by_line[word.line_key].append(word)

        scale_x = float(page.width) / image.width
        scale_y = float(page.height) / image.height
        blocks = tuple(
            ExtractedBlock(
                order=order,
                text=" ".join(word.text for word in words),
                bbox=(
                    min(word.left for word in words) * scale_x,
                    min(word.top for word in words) * scale_y,
                    max(word.left + word.width for word in words) * scale_x,
                    max(word.top + word.height for word in words) * scale_y,
                ),
                method="ocr",
                confidence=sum(word.confidence for word in words) / len(words),
            )
            for order, words in enumerate(words_by_line.values(), start=1)
        )
        return ExtractedPage(
            page_number=page.page_number,
            width=float(page.width),
            height=float(page.height),
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            method="ocr",
        )

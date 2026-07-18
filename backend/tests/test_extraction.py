from io import BytesIO

from docintel.db import database_connection, initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.extraction import (
    ExtractedBlock,
    ExtractedPage,
    ExtractionRepository,
    OcrWord,
    PdfExtractor,
)
from docintel.storage import PdfStore


class FakeOcrEngine:
    def recognize(self, image) -> list[OcrWord]:
        assert image.width > 0
        return [
            OcrWord("Scanned", 100, 200, 120, 30, 91.0, (1, 1, 1)),
            OcrWord("evidence", 230, 200, 140, 30, 89.0, (1, 1, 1)),
        ]


def _write_text_pdf(path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    )
    path.write_bytes(pdf)


def test_pdf_extractor_preserves_page_text_and_geometry(tmp_path) -> None:
    pdf_path = tmp_path / "text.pdf"
    _write_text_pdf(pdf_path, "Direct extraction works")

    pages = PdfExtractor().extract(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].width == 612
    assert pages[0].height == 792
    assert pages[0].text == "Direct extraction works"
    assert pages[0].blocks[0].method == "direct"
    assert pages[0].blocks[0].bbox[0] == 72


def test_pdf_extractor_routes_empty_page_through_ocr(tmp_path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_text_pdf(pdf_path, "")

    pages = PdfExtractor(ocr_engine=FakeOcrEngine()).extract(pdf_path)

    assert pages[0].method == "ocr"
    assert pages[0].text == "Scanned evidence"
    assert pages[0].blocks[0].method == "ocr"
    assert pages[0].blocks[0].confidence == 90.0
    assert pages[0].blocks[0].bbox[0] > 0


def test_extraction_repository_replaces_page_and_block_records(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    document = DocumentCatalog(PdfStore(tmp_path), DocumentRepository(database_path)).add_pdf(
        source=BytesIO(b"%PDF-1.7\nexample\n%%EOF"),
        original_filename="example.pdf",
    )
    repository = ExtractionRepository(database_path)
    page = ExtractedPage(
        page_number=1,
        width=612,
        height=792,
        text="First version",
        blocks=(ExtractedBlock(1, "First version", (10, 20, 100, 40)),),
    )

    repository.replace_pages(document.id, [page])
    repository.replace_pages(document.id, [page])

    with database_connection(database_path) as connection:
        page_count = connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        block = connection.execute("SELECT text, x0, method FROM blocks").fetchone()
        status = connection.execute("SELECT status FROM documents").fetchone()[0]
    assert page_count == 1
    assert dict(block) == {"text": "First version", "x0": 10.0, "method": "direct"}
    assert status == "extracted"

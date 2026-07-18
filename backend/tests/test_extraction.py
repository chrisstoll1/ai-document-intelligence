from docintel.extraction import PdfExtractor


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

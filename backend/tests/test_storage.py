import hashlib
from io import BytesIO

import pytest
from docintel.storage import InvalidPdfError, PdfStore, PdfTooLargeError


def test_pdf_store_uses_content_hash_and_reuses_identical_file(tmp_path) -> None:
    pdf_bytes = b"%PDF-1.7\nexample document\n%%EOF"
    expected_id = hashlib.sha256(pdf_bytes).hexdigest()
    store = PdfStore(tmp_path)

    first = store.put(BytesIO(pdf_bytes))
    second = store.put(BytesIO(pdf_bytes))

    assert first.document_id == expected_id
    assert first.storage_key == f"pdfs/{expected_id[:2]}/{expected_id}.pdf"
    assert first.path.read_bytes() == pdf_bytes
    assert first.size_bytes == len(pdf_bytes)
    assert first.already_exists is False
    assert second.already_exists is True
    assert list((tmp_path / "pdfs").rglob("*.pdf")) == [first.path]


@pytest.mark.parametrize(
    ("pdf_bytes", "max_bytes", "expected_error"),
    [
        (b"not a pdf", 100, InvalidPdfError),
        (b"%PDF-1.7\ntoo large", 10, PdfTooLargeError),
    ],
)
def test_pdf_store_rejects_invalid_input_without_leaving_files(
    tmp_path, pdf_bytes: bytes, max_bytes: int, expected_error: type[ValueError]
) -> None:
    store = PdfStore(tmp_path, max_bytes=max_bytes)

    with pytest.raises(expected_error):
        store.put(BytesIO(pdf_bytes))

    assert list((tmp_path / "pdfs").rglob("*.*")) == []


def test_pdf_store_delete_is_idempotent_and_prunes_hash_directory(tmp_path) -> None:
    store = PdfStore(tmp_path)
    stored = store.put(BytesIO(b"%PDF-1.7\ndelete me\n%%EOF"))

    assert store.delete(stored.document_id) is True
    assert store.delete(stored.document_id) is False
    assert not stored.path.exists()
    assert not stored.path.parent.exists()
    with pytest.raises(ValueError, match="SHA-256"):
        store.delete("invalid")

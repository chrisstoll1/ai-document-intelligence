from io import BytesIO

from docintel.chunking import ChunkRepository, ProvenanceChunker, SourceBlock
from docintel.db import database_connection, initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.storage import PdfStore


def test_chunker_splits_long_blocks_with_exact_overlapping_spans() -> None:
    block = SourceBlock(id=7, page_number=2, text="one two three four five six")
    chunker = ProvenanceChunker(max_words=4, overlap=1)

    chunks = chunker.chunk("a" * 64, [block])

    assert [chunk.text for chunk in chunks] == ["one two three four", "four five six"]
    assert chunks[0].spans[0].block_start == 0
    assert block.text[chunks[1].spans[0].block_start : chunks[1].spans[0].block_end] == "four five six"
    assert chunks[0].page_start == chunks[0].page_end == 2


def test_chunk_repository_rebuilds_fts_and_provenance(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    document = DocumentCatalog(PdfStore(tmp_path), DocumentRepository(database_path)).add_pdf(
        BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "example.pdf"
    )
    ExtractionRepository(database_path).replace_pages(
        document.id,
        [
            ExtractedPage(
                1,
                612,
                792,
                "Privacy controls\nWeather forecast",
                (
                    ExtractedBlock(1, "Privacy controls", (10, 10, 100, 20)),
                    ExtractedBlock(2, "Weather forecast", (10, 30, 100, 40)),
                ),
            )
        ],
    )
    repository = ChunkRepository(database_path, ProvenanceChunker(max_words=2, overlap=0))

    first = repository.rebuild(document.id)
    second = repository.rebuild(document.id)
    hits = repository.search("privacy")

    with database_connection(database_path) as connection:
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        span_count = connection.execute("SELECT COUNT(*) FROM chunk_spans").fetchone()[0]
    assert first == second
    assert chunk_count == 2
    assert span_count == 2
    assert hits[0].chunk_id == first[0].id
    assert hits[0].page_start == 1

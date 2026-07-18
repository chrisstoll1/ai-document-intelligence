from io import BytesIO

from docintel.chunking import ChunkRepository, ProvenanceChunker
from docintel.db import initialize_database
from docintel.documents import DocumentCatalog, DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.indexing import SemanticHit
from docintel.search import HybridSearchService
from docintel.storage import PdfStore


class FakeSemanticIndex:
    def __init__(self, hits: list[SemanticHit]) -> None:
        self.hits = hits

    def query(self, query: str, *, limit: int = 20) -> list[SemanticHit]:
        return self.hits[:limit]


def test_persistent_hybrid_search_fuses_and_hydrates_candidates(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    document = DocumentCatalog(PdfStore(tmp_path), DocumentRepository(database_path)).add_pdf(
        BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "evidence.pdf"
    )
    ExtractionRepository(database_path).replace_pages(
        document.id,
        [
            ExtractedPage(
                1,
                612,
                792,
                "The zebra is protected\nUnrelated weather",
                (
                    ExtractedBlock(1, "The zebra is protected", (10, 10, 150, 20)),
                    ExtractedBlock(2, "Unrelated weather", (10, 30, 150, 40)),
                ),
            )
        ],
    )
    chunks = ChunkRepository(database_path, ProvenanceChunker(max_words=4, overlap=0))
    indexed = chunks.rebuild(document.id)
    semantic = FakeSemanticIndex(
        [SemanticHit(indexed[1].id, 0.1), SemanticHit(indexed[0].id, 0.2)]
    )

    results = HybridSearchService(database_path, chunks, semantic).search("zebra")

    assert results[0].chunk_id == indexed[0].id
    assert results[0].document_name == "evidence.pdf"
    assert results[0].page_start == 1
    assert results[0].keyword_rank == 1
    assert results[0].semantic_rank == 2

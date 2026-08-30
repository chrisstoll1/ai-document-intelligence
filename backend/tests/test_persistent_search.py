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
        self.queries: list[str] = []

    def query(self, query: str, *, limit: int = 20) -> list[SemanticHit]:
        self.queries.append(query)
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

    semantic_only = HybridSearchService(
        database_path,
        chunks,
        semantic,
        keyword_weight=0,
        semantic_weight=1,
    ).search("zebra")
    assert semantic_only[0].chunk_id == indexed[1].id


def test_persistent_hybrid_search_orders_tied_results_deterministically(tmp_path) -> None:
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
                "Alpha evidence\nBeta evidence",
                (
                    ExtractedBlock(1, "Alpha evidence", (10, 10, 150, 20)),
                    ExtractedBlock(2, "Beta evidence", (10, 30, 150, 40)),
                ),
            )
        ],
    )
    chunks = ChunkRepository(database_path, ProvenanceChunker(max_words=2, overlap=0))
    indexed = chunks.rebuild(document.id)
    semantic = FakeSemanticIndex([SemanticHit(chunk.id, 0.1) for chunk in indexed])

    results = HybridSearchService(
        database_path,
        chunks,
        semantic,
        keyword_weight=0,
        semantic_weight=1,
    ).search("missing keyword")

    assert [result.chunk_id for result in results] == [chunk.id for chunk in indexed]


def test_persistent_hybrid_search_reserves_results_for_each_source(tmp_path) -> None:
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
                "zebra alpha\nzebra beta\ngamma delta\nzebra epsilon",
                (
                    ExtractedBlock(1, "zebra alpha", (10, 10, 150, 20)),
                    ExtractedBlock(2, "zebra beta", (10, 30, 150, 40)),
                    ExtractedBlock(3, "gamma delta", (10, 50, 150, 60)),
                    ExtractedBlock(4, "zebra epsilon", (10, 70, 150, 80)),
                ),
            )
        ],
    )
    chunks = ChunkRepository(database_path, ProvenanceChunker(max_words=2, overlap=0))
    indexed = chunks.rebuild(document.id)
    semantic = FakeSemanticIndex(
        [
            SemanticHit(indexed[2].id, 0.1),
            SemanticHit(indexed[1].id, 0.2),
            SemanticHit(indexed[0].id, 0.3),
            SemanticHit(indexed[3].id, 0.4),
        ]
    )

    results = HybridSearchService(database_path, chunks, semantic).search("zebra", limit=2)

    assert {result.chunk_id for result in results} == {indexed[0].id, indexed[2].id}


def test_persistent_hybrid_search_normalizes_joined_alphanumeric_terms(tmp_path) -> None:
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    document = DocumentCatalog(PdfStore(tmp_path), DocumentRepository(database_path)).add_pdf(
        BytesIO(b"%PDF-1.7\nexample\n%%EOF"), "handbook.pdf"
    )
    ExtractionRepository(database_path).replace_pages(
        document.id,
        [
            ExtractedPage(
                1,
                612,
                792,
                "Square 9 phone policy\nUnrelated evidence",
                (
                    ExtractedBlock(1, "Square 9 phone policy", (10, 10, 150, 20)),
                    ExtractedBlock(2, "Unrelated evidence", (10, 30, 250, 40)),
                ),
            )
        ],
    )
    chunks = ChunkRepository(database_path, ProvenanceChunker(max_words=7, overlap=0))
    indexed = chunks.rebuild(document.id)
    semantic = FakeSemanticIndex([SemanticHit(indexed[0].id, 0.1)])

    results = HybridSearchService(database_path, chunks, semantic).search("what is square9 policy")

    assert semantic.queries == ["what is square 9 policy"]
    assert results[0].chunk_id == indexed[0].id

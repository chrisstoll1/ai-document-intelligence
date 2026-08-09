import pytest
from docintel.db import initialize_database
from docintel.documents import DocumentRepository
from docintel.extraction import ExtractedBlock, ExtractedPage, ExtractionRepository
from docintel.metadata import EntityMention, MetadataRepository, PageText, SpacyEntityExtractor, normalize_entity_text
from docintel.storage import StoredPdf


def _document(tmp_path):
    database_path = tmp_path / "docintel.sqlite3"
    initialize_database(database_path)
    documents = DocumentRepository(database_path)
    document = documents.record_stored_pdf(
        StoredPdf(
            document_id="a" * 64,
            storage_key="pdfs/aa/document.pdf",
            path=tmp_path / "pdfs" / "aa" / "document.pdf",
            size_bytes=100,
            already_exists=False,
        ),
        "document.pdf",
    )
    page = ExtractedPage(
        1,
        612,
        792,
        "Alice joined Example Ltd in London.",
        (ExtractedBlock(1, "Alice joined Example Ltd in London.", (10, 10, 200, 30)),),
    )
    ExtractionRepository(database_path).replace_pages(document.id, [page])
    return documents, MetadataRepository(database_path), document.id


def test_entity_normalization_is_casefolded_and_compatibility_normalized() -> None:
    assert normalize_entity_text("  EXAMPLE\u00a0Ltd  ") == "example ltd"


def test_metadata_repository_replaces_mentions_and_validates_source_offsets(tmp_path) -> None:
    documents, repository, document_id = _document(tmp_path)
    first = [
        EntityMention(1, "PERSON", "Alice", 0, 5, 0.98),
        EntityMention(1, "ORGANIZATION", "Example Ltd", 13, 24, 0.91),
    ]

    repository.mark_processing(document_id)
    repository.replace_document(document_id, "fake-ner@revision:labels-v1", first)
    repository.replace_document(document_id, "fake-ner@revision:labels-v1", first[:1])

    mentions = repository.list_document(document_id)
    document = documents.get(document_id)
    assert mentions == [EntityMention(1, "PERSON", "Alice", 0, 5, 0.98, "alice")]
    assert document is not None
    assert document.metadata_status == "ready"
    assert document.metadata_model == "fake-ner@revision:labels-v1"
    assert document.metadata_error is None

    with pytest.raises(ValueError, match="does not match"):
        repository.replace_document(
            document_id,
            "fake-ner@revision:labels-v1",
            [EntityMention(1, "LOCATION", "Paris", 28, 34)],
        )


def test_metadata_failure_does_not_change_core_document_status(tmp_path) -> None:
    documents, repository, document_id = _document(tmp_path)
    documents.set_status(document_id, "ready")

    repository.mark_failed(document_id, "fake-ner@revision:labels-v1", "model unavailable")

    document = documents.get(document_id)
    assert document is not None
    assert document.status == "ready"
    assert document.metadata_status == "failed"
    assert document.metadata_error == "model unavailable"


def test_spacy_extractor_maps_evaluated_labels_and_preserves_offsets() -> None:
    class Entity:
        def __init__(self, label, text, start, end) -> None:
            self.label_ = label
            self.text = text
            self.start_char = start
            self.end_char = end

    class Pipeline:
        def __call__(self, text):
            return type(
                "Document",
                (),
                {
                    "ents": [
                        Entity("PERSON", "Alice", 0, 5),
                        Entity("ORG", "Example Ltd", 13, 24),
                        Entity("DATE", "2020", 28, 32),
                    ]
                },
            )()

    extractor = SpacyEntityExtractor(pipeline=Pipeline())

    mentions = extractor.extract([PageText(1, "Alice joined Example Ltd in 2020.")])

    assert mentions == [
        EntityMention(1, "PERSON", "Alice", 0, 5),
        EntityMention(1, "ORGANIZATION", "Example Ltd", 13, 24),
    ]
    assert extractor.version == "spacy:en_core_web_trf@3.8.0:labels-v1:normalization-v1"


def test_exact_metadata_page_matches_require_label_normalization_and_current_model(tmp_path) -> None:
    _, repository, document_id = _document(tmp_path)
    repository.replace_document(
        document_id,
        "selected-model",
        [EntityMention(1, "ORGANIZATION", "Example Ltd", 13, 24)],
    )

    matches = repository.find_exact_pages(
        [("ORGANIZATION", "example ltd"), ("LOCATION", "example ltd")],
        model_version="selected-model",
    )

    assert [(match.document_id, match.page_number, match.label) for match in matches] == [
        (document_id, 1, "ORGANIZATION")
    ]
    assert repository.find_exact_pages([("ORGANIZATION", "example")], model_version="selected-model") == []
    assert repository.find_exact_pages([("ORGANIZATION", "example ltd")], model_version="stale-model") == []

from docintel.chunking import ProvenanceChunk
from docintel.indexing import ChromaSemanticIndex


class FakeEncoder:
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        return [
            [1.0, 0.0] if any(term in sentence.lower() for term in ("privacy", "confidential")) else [0.0, 1.0]
            for sentence in sentences
        ]


def _chunk(chunk_id: str, text: str, ordinal: int) -> ProvenanceChunk:
    return ProvenanceChunk(chunk_id, "d" * 64, ordinal, text, 1, 1, "test-v1", ())


def test_chroma_index_persists_and_replaces_document_chunks(tmp_path) -> None:
    path = tmp_path / "chroma"
    index = ChromaSemanticIndex(path, encoder=FakeEncoder(), model_name="fake-model")
    privacy = _chunk("a" * 64, "Privacy controls", 1)
    weather = _chunk("b" * 64, "Weather forecast", 2)

    index.replace_document("d" * 64, [privacy, weather])
    first_hits = index.query("confidential information")

    reopened = ChromaSemanticIndex(path, encoder=FakeEncoder(), model_name="fake-model")
    reopened.replace_document("d" * 64, [privacy])
    second_hits = reopened.query("weather")

    assert first_hits[0].chunk_id == privacy.id
    assert [hit.chunk_id for hit in second_hits] == [privacy.id]

    index.close()
    reopened.close()

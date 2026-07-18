import hashlib
import json
from pathlib import Path

from docintel.config import Settings
from docintel.extraction import MINIMUM_DIRECT_CHARACTERS, OCR_DPI
from docintel.search import KEYWORD_WEIGHT, RRF_K, SEMANTIC_WEIGHT


def test_settings_reads_data_directory_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOCINTEL_DATA_DIR", "custom-data")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_REVISION", "model-revision")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_QUERY_PROMPT", "query prompt: ")
    monkeypatch.setenv("DOCINTEL_CHUNK_MAX_WORDS", "80")
    monkeypatch.setenv("DOCINTEL_CHUNK_OVERLAP", "15")

    settings = Settings.from_environment()

    assert settings.data_dir == Path("custom-data")
    assert settings.database_path == Path("custom-data/docintel.sqlite3")
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_revision == "model-revision"
    assert settings.embedding_query_prompt == "query prompt: "
    assert settings.chunk_max_words == 80
    assert settings.chunk_overlap == 15


def test_frozen_retrieval_configuration_matches_application_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((root / "evaluation" / "config" / "retrieval_v1.json").read_text(encoding="utf-8"))
    settings = Settings(Path("data"))

    assert frozen["embedding"]["model"] == settings.embedding_model
    assert frozen["embedding"]["revision"] == settings.embedding_revision
    assert frozen["chunking"]["max_words"] == settings.chunk_max_words
    assert frozen["chunking"]["overlap_words"] == settings.chunk_overlap
    assert frozen["fusion"]["rrf_k"] == RRF_K
    assert frozen["fusion"]["keyword_weight"] == KEYWORD_WEIGHT
    assert frozen["fusion"]["semantic_weight"] == SEMANTIC_WEIGHT


def test_frozen_ocr_configuration_matches_application_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((root / "evaluation" / "config" / "ocr_v1.json").read_text(encoding="utf-8"))
    result = json.loads((root / frozen["selection_evidence"]["result"]).read_text(encoding="utf-8"))
    benchmark_bytes = (root / frozen["selection_evidence"]["benchmark"]).read_bytes()

    assert frozen["selected_engine"]["name"] == "Tesseract"
    assert frozen["selected_engine"]["language"] == "eng"
    assert frozen["selected_engine"]["page_segmentation_mode"] == 3
    assert frozen["routing"]["minimum_direct_alphanumeric_characters"] == MINIMUM_DIRECT_CHARACTERS
    assert frozen["routing"]["page_render_dpi"] == OCR_DPI
    assert frozen["selection_evidence"]["benchmark_manifest_sha256"] == hashlib.sha256(benchmark_bytes).hexdigest()
    assert frozen["selection_evidence"]["cer"] == result["engines"]["tesseract"]["metrics"]["overall"]["cer"]
    assert frozen["selection_evidence"]["wer"] == result["engines"]["tesseract"]["metrics"]["overall"]["wer"]

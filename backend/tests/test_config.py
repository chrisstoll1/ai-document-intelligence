from pathlib import Path

from docintel.config import Settings


def test_settings_reads_data_directory_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOCINTEL_DATA_DIR", "custom-data")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_QUERY_PROMPT", "query prompt: ")
    monkeypatch.setenv("DOCINTEL_CHUNK_MAX_WORDS", "80")
    monkeypatch.setenv("DOCINTEL_CHUNK_OVERLAP", "15")

    settings = Settings.from_environment()

    assert settings.data_dir == Path("custom-data")
    assert settings.database_path == Path("custom-data/docintel.sqlite3")
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_query_prompt == "query prompt: "
    assert settings.chunk_max_words == 80
    assert settings.chunk_overlap == 15

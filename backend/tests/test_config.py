from pathlib import Path

from docintel.config import Settings


def test_settings_reads_data_directory_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOCINTEL_DATA_DIR", "custom-data")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("DOCINTEL_EMBEDDING_QUERY_PROMPT", "query prompt: ")

    settings = Settings.from_environment()

    assert settings.data_dir == Path("custom-data")
    assert settings.database_path == Path("custom-data/docintel.sqlite3")
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_query_prompt == "query prompt: "

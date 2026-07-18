from pathlib import Path

from docintel.config import Settings


def test_settings_reads_data_directory_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("DOCINTEL_DATA_DIR", "custom-data")

    settings = Settings.from_environment()

    assert settings.data_dir == Path("custom-data")
    assert settings.database_path == Path("custom-data/docintel.sqlite3")

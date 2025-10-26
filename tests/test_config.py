from pathlib import Path

from app.config import OCRConfig, load_config


def test_load_default(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('{"languages": ["id", "en"], "min_confidence": 0.9}', encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, OCRConfig)
    assert config.min_confidence == 0.9
    assert config.languages == ["id", "en"]

import pytest

from app.config import OCRConfig
from app.postprocess import PostProcessor


def test_postprocess_normalization() -> None:
    config = OCRConfig()
    processor = PostProcessor(config)
    result = processor.run({"text": "tanggal 1/2/2023 harga rp 1.234,56", "blocks": []})
    assert "2023" in result.text
    assert "IDR" in result.text
    assert "emails" not in result.metadata


def test_language_detection_indonesian(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.postprocess.detect_langs", None)
    processor = PostProcessor(OCRConfig())
    result = processor.run({"text": "Halo apa kabar kamu sedang di mana", "blocks": []})
    assert result.language == "id"


def test_language_detection_english(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.postprocess.detect_langs", None)
    processor = PostProcessor(OCRConfig())
    result = processor.run({"text": "Please pay this invoice within seven days", "blocks": []})
    assert result.language == "en"

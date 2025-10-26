from app.config import OCRConfig
from app.postprocess import PostProcessor


def test_postprocess_normalization() -> None:
    config = OCRConfig()
    processor = PostProcessor(config)
    result = processor.run({"text": "tanggal 1/2/2023 harga rp 1.234,56", "blocks": []})
    assert "2023" in result.text
    assert "IDR" in result.text
    assert "emails" not in result.metadata

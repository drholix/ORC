from pathlib import Path

from app.cache import OCRCache


def test_cache_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite3"
    cache = OCRCache(db)
    key = cache.compute_key(b"data")
    cache.set(key, {"text": "hello"})
    result = cache.get(key)
    assert result is not None
    assert result["text"] == "hello"

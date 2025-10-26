from app.config import OCRConfig
from app.service import OCRService


def test_service_process_image(tmp_path, sample_image):
    config = OCRConfig(cache_path=str(tmp_path / "cache.sqlite3"))
    service = OCRService(config)
    response1 = service.process_image(sample_image)
    response2 = service.process_image(sample_image)
    assert response1.text
    assert response2.text
    assert response2.meta["pipeline"] == response1.meta["pipeline"]
    assert response2.meta["image_size"] == response1.meta["image_size"]


def test_service_process_path_directory(tmp_path, sample_image):
    config = OCRConfig(cache_path=str(tmp_path / "cache.sqlite3"))
    service = OCRService(config)
    responses = service.process_path(sample_image.parent)
    assert responses

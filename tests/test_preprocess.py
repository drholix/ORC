from app.config import OCRConfig
from app.preprocess import Preprocessor


def test_preprocess_runs() -> None:
    config = OCRConfig(max_image_size=128)
    preprocessor = Preprocessor(config)
    image = [[[255, 255, 255] for _ in range(64)] for _ in range(64)]
    result = preprocessor.run(image)
    if hasattr(result.image, "shape"):
        assert result.image.shape[0] == 64
    else:
        assert len(result.image) == 64
    if "noop" in result.steps:
        assert result.steps == ["noop"]
    else:
        assert "grayscale" in result.steps
        assert "morphology" in result.steps

"""Tests for the inference helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import OCRConfig
from app.inference import DummyOCREngine, PaddleOCREngine, create_engine


@pytest.fixture(autouse=True)
def clear_paddle_module():
    """Ensure paddleocr stubs do not leak between tests."""

    original = sys.modules.pop("paddleocr", None)
    try:
        yield
    finally:
        if original is not None:
            sys.modules["paddleocr"] = original


@pytest.fixture(autouse=True)
def _patch_image_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalise the ensure_bgr_image helper for lightweight tests."""

    monkeypatch.setattr("app.inference.ensure_bgr_image", lambda image: image)


def test_paddle_engine_drops_unsupported_show_log(monkeypatch):
    """Verify that PaddleOCREngine removes unknown kwargs like ``show_log``."""

    class FakePaddleOCR:
        def __init__(
            self,
            *,
            use_angle_cls: bool | None = None,
            lang: str,
            use_gpu: bool,
            ocr_version: str | None = None,
            det_model_dir: str | None = None,
            rec_model_dir: str | None = None,
            cls_model_dir: str | None = None,
            structure_version: str | None = None,
            **kwargs,
        ) -> None:
            self.init_kwargs = {
                "use_angle_cls": use_angle_cls,
                "lang": lang,
                "use_gpu": use_gpu,
                "ocr_version": ocr_version,
                "det_model_dir": det_model_dir,
                "rec_model_dir": rec_model_dir,
                "cls_model_dir": cls_model_dir,
                "structure_version": structure_version,
                **kwargs,
            }

        def ocr(self, image, cls=True):  # pragma: no cover - not exercised
            return []

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)

    assert "show_log" not in engine._ocr_kwargs


def test_paddle_engine_handles_device_param(monkeypatch):
    """Ensure GPU toggles map to ``device`` when ``use_gpu`` is unsupported."""

    captured_kwargs: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(
            self,
            *,
            use_angle_cls: bool | None = None,
            lang: str,
            device: str,
            **kwargs,
        ) -> None:
            captured_kwargs.update(
                {
                    "use_angle_cls": use_angle_cls,
                    "lang": lang,
                    "device": device,
                }
            )
            captured_kwargs.update(kwargs)

        def ocr(self, image, cls=True):  # pragma: no cover - not exercised
            return []

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(enable_gpu=True)
    PaddleOCREngine(config)

    assert captured_kwargs["device"] == "gpu"
    assert "use_gpu" not in captured_kwargs
    assert captured_kwargs.get("use_angle_cls") in {None, False}


def test_paddle_engine_drops_unknown_kwargs_and_retries(monkeypatch):
    """ValueErrors that mention unknown args should trigger a retry without them."""

    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(
            self,
            *,
            use_angle_cls: bool | None = None,
            lang: str,
            use_gpu: bool | None = None,
            **kwargs,
        ) -> None:
            calls.append({
                "use_angle_cls": use_angle_cls,
                "lang": lang,
                "use_gpu": use_gpu,
                **kwargs,
            })
            if len(calls) == 1:
                raise ValueError("Unknown argument: use_gpu")

        def ocr(self, image, cls=True):  # pragma: no cover - not exercised
            return []

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)

    assert len(calls) == 2  # first attempt fails, second succeeds
    assert "use_gpu" not in engine._ocr_kwargs


def test_create_engine_falls_back_to_dummy(monkeypatch):
    """If Paddle initialization fails at runtime we fall back to the dummy engine."""

    def boom(self, config):  # pragma: no cover - invoked via create_engine
        raise RuntimeError("paddle missing")

    monkeypatch.setenv("OCR_FAKE_ENGINE", "0")
    monkeypatch.setattr("app.inference.PaddleOCREngine.__init__", boom, raising=False)

    engine = create_engine(OCRConfig())

    assert isinstance(engine, DummyOCREngine)


def test_paddle_engine_runtime_drops_cls_argument(monkeypatch):
    """TypeErrors mentioning ``cls`` should trigger a retry without that kwarg."""

    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1 and "cls" in kwargs:
                raise TypeError("PaddleOCR.predict() got an unexpected keyword argument 'cls'")
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("hello", 0.99),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "hello"
    assert len(calls) == 2
    first_call, second_call = calls
    assert first_call.get("cls") is True
    expected_without_cls = {key: value for key, value in first_call.items() if key != "cls"}
    assert second_call == expected_without_cls


def test_paddle_engine_runtime_drops_unknown_argument(monkeypatch):
    """Unknown runtime kwargs should be removed before retrying."""

    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise TypeError(
                    "PaddleOCR.predict() got an unexpected keyword argument 'det_db_thresh'"
                )
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("world", 0.97),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    engine._runtime_call_kwargs["det_db_thresh"] = 0.25

    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "world"
    assert len(calls) == 2
    assert "det_db_thresh" not in calls[1]


def test_paddle_engine_inversion_strategy_recovers_text(monkeypatch):
    """Dark themed captures should succeed via the inversion preprocessing strategy."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            pixel = int(image[0][0][0])
            if pixel == 0:
                return [[]]
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("inverted", 0.93),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    engine = PaddleOCREngine(OCRConfig())
    result = engine.infer(image, ["en"])

    assert result.text == "inverted"
    assert result.blocks[0]["confidence"] == pytest.approx(0.93)


def test_paddle_engine_scale_strategy_expands_small_inputs(monkeypatch):
    """Very small selections should succeed after the scale-up retry."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            height, width = image.shape[:2]
            if max(height, width) <= 8:
                return [[]]
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("scaled", 0.92),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    engine = PaddleOCREngine(OCRConfig())
    result = engine.infer(image, ["en"])

    assert result.text == "scaled"
    assert result.blocks[0]["confidence"] == pytest.approx(0.92)


def test_paddle_engine_preprocessing_without_cv2(monkeypatch):
    """The preprocessing retry should function even when OpenCV is unavailable."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            pixel = int(image[0][0][0])
            if pixel == 10:
                return [[]]
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("fallback", 0.88),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setattr("app.inference.cv2", None, raising=False)

    image = np.full((2, 2, 3), 10, dtype=np.uint8)
    engine = PaddleOCREngine(OCRConfig())
    result = engine.infer(image, ["en"])

    assert result.text == "fallback"
    assert result.blocks[0]["confidence"] == pytest.approx(0.88)


def test_paddle_engine_runtime_handles_missing_cls_parameter(monkeypatch):
    """When ``ocr`` lacks the ``cls`` parameter it should not be passed."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("text", 0.95),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)

    assert "cls" not in engine._runtime_call_kwargs
    result = engine.infer([[0, 0], [0, 0]], ["en"])
    assert result.text == "text"


def test_paddle_engine_disables_angle_cls_when_textline_enabled(monkeypatch):
    captured: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            captured.update(kwargs)

        def ocr(self, image, **kwargs):
            return [[[([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]), ("text", 0.95)]]]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    engine = PaddleOCREngine(OCRConfig(use_textline_orientation=True))

    assert captured.get("use_textline_orientation") is True
    assert captured.get("use_angle_cls") in {None, False}
    assert engine.textline_orientation_enabled is True
    assert engine.angle_cls_enabled is False


def test_resolve_language_auto_defaults_to_en() -> None:
    assert PaddleOCREngine._resolve_language(["auto"]) == "en"
    assert PaddleOCREngine._resolve_language(["id", "en"]) == "en"


def test_paddle_engine_falls_back_to_cpu_when_gpu_fails(monkeypatch):
    """GPU initialization errors should trigger a CPU retry."""

    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(self, *, use_gpu: bool | None = None, **kwargs) -> None:
            snapshot = dict(kwargs)
            snapshot["use_gpu"] = use_gpu
            calls.append(snapshot)
            if use_gpu:
                raise RuntimeError("CUDA init failed")

        def ocr(self, image, cls=True):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("cpu", 0.9),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(enable_gpu=True)
    engine = PaddleOCREngine(config)

    assert len(calls) >= 2
    assert calls[0].get("use_gpu") is True
    assert calls[-1].get("use_gpu") in (False, None)
    assert engine.device == "cpu"

    result = engine.infer([[0, 0], [0, 0]], ["en"])
    assert result.device == "cpu"
    assert result.text == "cpu"


def test_paddle_engine_handles_triplet_entries(monkeypatch):
    """Tuple payloads with extra metadata should still be parsed."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("triplet", 0.88),
                        {"extra": True},
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "triplet"
    assert result.blocks[0]["confidence"] == pytest.approx(0.88)


def test_paddle_engine_handles_dict_entries(monkeypatch):
    """Dict payloads from PaddleOCR-VL should be normalised."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            return [
                [
                    {
                        "bbox": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        "text": "dict",
                        "score": 0.92,
                    }
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "dict"
    assert result.blocks[0]["confidence"] == pytest.approx(0.92)


def test_paddle_engine_handles_nested_dict_entries(monkeypatch):
    """Nested ``res`` payloads with ``line`` metadata should be parsed."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            return [
                {
                    "res": [
                        {
                            "line": {
                                "points": [
                                    [0.0, 0.0],
                                    [1.0, 0.0],
                                    [1.0, 1.0],
                                    [0.0, 1.0],
                                ],
                                "content": "nested",
                                "confidence": 0.87,
                            }
                        }
                    ]
                }
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "nested"
    assert result.blocks[0]["confidence"] == pytest.approx(0.87)


def test_paddle_engine_handles_data_payload(monkeypatch):
    """Top-level dicts with ``data`` lists should be treated as a page."""

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = kwargs

        def ocr(self, image, **kwargs):
            return {
                "data": [
                    {
                        "box": [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                        ],
                        "text": "data",
                        "score": 0.91,
                    }
                ]
            }

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig()
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "data"
    assert result.blocks[0]["confidence"] == pytest.approx(0.91)


def test_paddle_engine_falls_back_when_language_missing(monkeypatch):
    """Language errors should trigger a retry with a fallback locale."""

    calls: list[str] = []

    class FakePaddleOCR:
        def __init__(self, *, lang: str, **kwargs) -> None:  # pragma: no cover - init only
            calls.append(lang)
            if lang == "multilingual":
                raise ValueError(
                    "No models are available for the language 'multilingual' and OCR version 'PP-OCRv5'."
                )

        def ocr(self, image, **kwargs):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("fallback", 0.91),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(languages=["ru", "en"], ocr_version="PP-OCRv5")
    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["ru", "en"])

    assert calls[0] == "multilingual"
    assert calls[-1] == "en"
    assert engine.lang == calls[-1]
    assert result.language == engine.lang
    assert result.text == "fallback"


def test_paddle_engine_applies_detection_overrides(monkeypatch):
    """Detection thresholds should propagate to the PaddleOCR constructor."""

    captured_kwargs: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            captured_kwargs.update(kwargs)

        def ocr(self, image, **kwargs):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("override", 0.9),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(
        text_det_limit_side_len=720,
        text_det_limit_type="min",
        text_detection_model_name="custom_det",
        text_recognition_model_name="custom_rec",
        text_det_unclip_ratio=2.1,
        text_det_box_thresh=0.45,
        text_det_thresh=0.15,
        text_rec_score_thresh=0.4,
        use_doc_preprocessor=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    engine = PaddleOCREngine(config)
    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert result.text == "override"
    assert captured_kwargs["text_det_limit_side_len"] == 720
    assert captured_kwargs["text_det_limit_type"] == "min"
    assert captured_kwargs["text_det_unclip_ratio"] == 2.1
    assert captured_kwargs["text_det_box_thresh"] == 0.45
    assert captured_kwargs["text_det_thresh"] == 0.15
    assert captured_kwargs["text_rec_score_thresh"] == 0.4
    assert captured_kwargs["text_detection_model_name"] == "custom_det"
    assert captured_kwargs["text_recognition_model_name"] == "custom_rec"
    assert captured_kwargs["use_doc_preprocessor"] is False
    assert captured_kwargs["use_doc_orientation_classify"] is False
    assert captured_kwargs["use_doc_unwarping"] is False
    assert captured_kwargs["use_textline_orientation"] is False


def test_paddle_engine_falls_back_to_mobile_recognition(monkeypatch):
    """Unsupported recognition models should fall back to a mobile variant."""

    attempts: list[str | None] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            attempts.append(kwargs.get("text_recognition_model_name"))
            if kwargs.get("text_recognition_model_name") == "en_PP-OCRv5_server_rec":
                raise ValueError(
                    "The model (en_PP-OCRv5_server_rec) is not supported!"
                )

        def ocr(self, image, **kwargs):
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("fallback", 0.9),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(text_recognition_model_name="en_PP-OCRv5_server_rec")
    engine = PaddleOCREngine(config)

    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert attempts[0] == "en_PP-OCRv5_server_rec"
    assert any(name == "en_PP-OCRv5_mobile_rec" for name in attempts[1:])
    assert result.text == "fallback"


def test_paddle_engine_relaxed_retry(monkeypatch):
    """Empty results should trigger the relaxed fallback engine."""

    instances: list[object] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # pragma: no cover - init only
            self.kwargs = dict(kwargs)
            instances.append(self)

        def ocr(self, image, **kwargs):
            if self is instances[0]:
                return [[]]
            return [
                [
                    (
                        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        ("relaxed", 0.87),
                    )
                ]
            ]

    sys.modules["paddleocr"] = SimpleNamespace(PaddleOCR=FakePaddleOCR)

    config = OCRConfig(min_confidence=0.2)
    engine = PaddleOCREngine(config)

    result = engine.infer([[0, 0], [0, 0]], ["en"])

    assert len(instances) >= 2
    assert result.text == "relaxed"
    assert any(
        isinstance(getattr(instance, "kwargs", {}), dict)
        and instance.kwargs.get("text_det_box_thresh") in {0.3, 0.4}
        for instance in instances
    )

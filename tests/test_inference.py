"""Tests for the inference helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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


def test_paddle_engine_drops_unsupported_show_log(monkeypatch):
    """Verify that PaddleOCREngine removes unknown kwargs like ``show_log``."""

    class FakePaddleOCR:
        def __init__(
            self,
            *,
            use_angle_cls: bool,
            lang: str,
            use_gpu: bool,
            ocr_version: str | None = None,
            det_model_dir: str | None = None,
            rec_model_dir: str | None = None,
            cls_model_dir: str | None = None,
            structure_version: str | None = None,
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
        def __init__(self, *, use_angle_cls: bool, lang: str, device: str, **kwargs) -> None:
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


def test_paddle_engine_drops_unknown_kwargs_and_retries(monkeypatch):
    """ValueErrors that mention unknown args should trigger a retry without them."""

    calls: list[dict[str, object]] = []

    class FakePaddleOCR:
        def __init__(
            self,
            *,
            use_angle_cls: bool,
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

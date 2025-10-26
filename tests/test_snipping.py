from __future__ import annotations

import types

import pytest

from app import snipping


class _DummyGrab:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    @property
    def ndim(self) -> int:
        return len(self.shape)


class _RecordingMSS:
    def __init__(self, channels: int | None = 4) -> None:
        self.last_monitor = None
        self._channels = channels

    def __enter__(self) -> "_RecordingMSS":
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - nothing to clean up
        return None

    def grab(self, monitor: dict[str, int]) -> _DummyGrab:
        self.last_monitor = monitor
        height = monitor["height"]
        width = monitor["width"]
        if self._channels is None:
            return _DummyGrab((height, width))
        return _DummyGrab((height, width, self._channels))


@pytest.fixture(autouse=True)
def _restore_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure optional dependencies are restored after every test."""

    original_mss = snipping.mss
    original_np = snipping.np
    original_cv2 = snipping.cv2
    try:
        yield
    finally:
        monkeypatch.setattr(snipping, "mss", original_mss, raising=False)
        monkeypatch.setattr(snipping, "np", original_np, raising=False)
        monkeypatch.setattr(snipping, "cv2", original_cv2, raising=False)


def test_capture_region_rounds_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingMSS()
    monkeypatch.setattr(snipping, "mss", types.SimpleNamespace(mss=lambda: recorder))
    captured = {}
    monkeypatch.setattr(snipping, "ensure_bgr_image", lambda image: captured.setdefault("image", image))
    monkeypatch.setattr(
        snipping,
        "np",
        types.SimpleNamespace(
            array=lambda *args, **kwargs: _DummyGrab((2, 2, 4)),
            uint8="uint8",
        ),
    )
    monkeypatch.setattr(
        snipping,
        "cv2",
        types.SimpleNamespace(COLOR_BGRA2BGR=1, cvtColor=lambda img, _code: img),
    )

    selection = snipping.Selection(left=1.2, top=3.7, width=50.4, height=10.9)  # type: ignore[arg-type]
    result = snipping._capture_region(selection)

    assert recorder.last_monitor == {"left": 1, "top": 4, "width": 50, "height": 11}
    assert result is captured["image"]


def test_capture_region_upgrades_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingMSS()
    monkeypatch.setattr(snipping, "mss", types.SimpleNamespace(mss=lambda: recorder))

    monkeypatch.setattr(
        snipping,
        "np",
        types.SimpleNamespace(
            array=lambda *args, **kwargs: _DummyGrab((10, 10)),
            uint8="uint8",
        ),
    )

    converted = {}

    def _convert(image: _DummyGrab, code: int) -> str:  # type: ignore[override]
        converted["image"] = image
        converted["code"] = code
        return "converted"

    monkeypatch.setattr(snipping, "ensure_bgr_image", lambda image: _convert(image, 2))

    selection = snipping.Selection(left=0, top=0, width=10, height=10)
    result = snipping._capture_region(selection)

    assert result == "converted"
    assert isinstance(converted["image"], _DummyGrab)


def test_run_snipping_ocr_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    selection = snipping.Selection(left=0, top=0, width=100, height=50)

    monkeypatch.setattr(snipping, "tk", object(), raising=False)
    monkeypatch.setattr(snipping, "mss", object(), raising=False)
    monkeypatch.setattr(snipping, "np", object(), raising=False)
    monkeypatch.setattr(snipping, "cv2", object(), raising=False)
    monkeypatch.setattr(snipping, "ensure_bgr_image", lambda image: image)
    monkeypatch.setattr(snipping, "_gpu_available", lambda: False)

    class _Selector:
        def select(self) -> snipping.Selection:
            return selection

    monkeypatch.setattr(snipping, "RegionSelector", _Selector)
    monkeypatch.setattr(snipping, "_capture_region", lambda _sel: "image")
    monkeypatch.setattr(snipping, "_encode_image", lambda _image: b"payload")

    captured: dict[str, object] = {}

    class _Response:
        def __init__(self) -> None:
            self.text = "Detected"
            self.meta = {"duration_ms": 42.0}

    class _Service:
        def __init__(self, config: snipping.OCRConfig) -> None:
            captured["config"] = config

        def process_image(self, payload: bytes) -> _Response:
            captured["payload"] = payload
            return _Response()

    monkeypatch.setattr(snipping, "OCRService", _Service)
    monkeypatch.setattr(snipping, "_copy_to_clipboard", lambda text: captured.setdefault("copied", text))
    monkeypatch.setattr(
        snipping,
        "_show_popup",
        lambda text, duration: captured.setdefault("popup", (text, duration)),
    )

    result = snipping.run_snipping_ocr(snipping.OCRConfig())

    assert result == "Detected"
    assert captured["payload"] == b"payload"
    assert captured["copied"] == "Detected"
    assert captured["popup"] == ("Detected", 42.0)

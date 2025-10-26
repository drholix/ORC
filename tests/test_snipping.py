from __future__ import annotations

import types

import pytest

from app import snipping


class _DummyGrab:
    shape = (2, 2, 4)


class _RecordingMSS:
    def __init__(self) -> None:
        self.last_monitor = None

    def __enter__(self) -> "_RecordingMSS":
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - nothing to clean up
        return None

    def grab(self, monitor: dict[str, int]) -> _DummyGrab:
        self.last_monitor = monitor
        return _DummyGrab()


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
    monkeypatch.setattr(snipping, "np", types.SimpleNamespace(array=lambda grab: _DummyGrab()))
    monkeypatch.setattr(
        snipping,
        "cv2",
        types.SimpleNamespace(COLOR_BGRA2BGR=1, cvtColor=lambda img, _code: img),
    )

    selection = snipping.Selection(left=1.2, top=3.7, width=50.4, height=10.9)  # type: ignore[arg-type]
    snipping._capture_region(selection)

    assert recorder.last_monitor == {"left": 1, "top": 4, "width": 50, "height": 11}

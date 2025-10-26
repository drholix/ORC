"""Interactive OCR snipping utility built for lightweight laptop setups."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - GUI dependency
    import tkinter as tk
except Exception:  # pragma: no cover - tkinter may be unavailable
    tk = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import mss  # type: ignore
except ImportError:  # pragma: no cover
    mss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyperclip  # type: ignore
except ImportError:  # pragma: no cover
    pyperclip = None  # type: ignore

from .config import OCRConfig, load_config
from .image_utils import ensure_bgr_image
from .inference import DummyOCREngine
from .service import OCRService


@dataclass
class Selection:
    left: int
    top: int
    width: int
    height: int

    @property
    def is_valid(self) -> bool:
        return self.width > 1 and self.height > 1


class RegionSelector:
    """Draw a translucent full-screen overlay to pick a rectangle."""

    def __init__(self) -> None:
        if tk is None:  # pragma: no cover - best effort guard
            raise RuntimeError("tkinter is required for the snipping tool")
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.25)
        self.root.configure(background="#000000")
        self.root.title("OCR Snipping Overlay (press Esc to cancel)")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = 0
        self.start_y = 0
        self.rect = None
        self.selection: Optional[Selection] = None

        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self.root.bind("<Escape>", self._on_escape)

    def _on_button_press(self, event: "tk.Event[tk.Misc]") -> None:  # type: ignore[name-defined]
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="#00ff99",
            width=2,
        )

    def _on_move_press(self, event: "tk.Event[tk.Misc]") -> None:  # type: ignore[name-defined]
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def _on_button_release(self, event: "tk.Event[tk.Misc]") -> None:  # type: ignore[name-defined]
        end_x = int(round(self.canvas.canvasx(event.x)))
        end_y = int(round(self.canvas.canvasy(event.y)))
        left = int(round(min(self.start_x, end_x)))
        top = int(round(min(self.start_y, end_y)))
        width = int(round(abs(end_x - self.start_x)))
        height = int(round(abs(end_y - self.start_y)))
        self.selection = Selection(left=left, top=top, width=width, height=height)
        self.root.quit()

    def _on_escape(self, _event: "tk.Event[tk.Misc]") -> None:  # type: ignore[name-defined]
        self.selection = None
        self.root.quit()

    def select(self) -> Optional[Selection]:
        self.root.mainloop()
        self.root.destroy()
        return self.selection


def _capture_region(selection: Selection) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
    if not selection.is_valid:
        return None
    if mss is None or np is None:
        raise RuntimeError("mss and numpy are required to capture the screen")
    with mss.mss() as sct:  # type: ignore[attr-defined]
        monitor = {
            "left": int(round(selection.left)),
            "top": int(round(selection.top)),
            "width": int(round(selection.width)),
            "height": int(round(selection.height)),
        }
        grab = sct.grab(monitor)
        image = np.array(grab, dtype=np.uint8)
    return ensure_bgr_image(image)


def _encode_image(image: "np.ndarray") -> bytes:  # type: ignore[name-defined]
    if cv2 is None:
        raise RuntimeError("opencv-python is required to encode screenshots")
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode screenshot for OCR")
    return buffer.tobytes()


def _copy_to_clipboard(text: str) -> None:
    if not text:
        return
    if pyperclip is not None:  # pragma: no branch - straightforward preference
        try:
            pyperclip.copy(text)
            return
        except pyperclip.PyperclipException:  # pragma: no cover - environment specific
            pass
    if tk is None:  # pragma: no cover - fallback path
        return
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


def _show_popup(text: str, duration_ms: Optional[float]) -> None:
    if tk is None:
        print("OCR result:\n" + text)
        if duration_ms is not None:
            print(f"Processing time: {duration_ms:.1f} ms")
        return
    root = tk.Tk()
    root.title("Snipping OCR Result")
    root.attributes("-topmost", True)
    info = "Text copied to clipboard"
    if duration_ms is not None:
        info += f" — {duration_ms:.0f} ms"
    label = tk.Label(root, text=info, font=("Segoe UI", 10))
    label.pack(padx=12, pady=(12, 4))
    text_widget = tk.Text(root, wrap="word", height=10, width=60)
    text_widget.pack(padx=12, pady=4)
    if text:
        text_widget.insert("1.0", text)
    else:
        text_widget.insert("1.0", "No text detected.")
    text_widget.configure(state="disabled")
    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack(pady=(0, 12))
    root.mainloop()


def _gpu_available() -> bool:
    """Best-effort probe to check whether Paddle can see a GPU."""

    try:  # pragma: no cover - depends on heavy optional dependency
        import paddle  # type: ignore

        device = getattr(paddle, "device", None)
        cuda = getattr(device, "cuda", None)
        if callable(getattr(cuda, "device_count", None)):
            return bool(cuda.device_count())  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - probing failure falls back to CPU
        return False
    return False


def run_snipping_ocr(config: Optional[OCRConfig] = None) -> Optional[str]:
    """Launch the interactive snipping overlay and run OCR on the captured area."""

    if tk is None:
        raise RuntimeError("tkinter is required for the snipping workflow")
    if mss is None:
        raise RuntimeError("Install the 'mss' package to capture screen regions")
    if np is None or cv2 is None:
        raise RuntimeError("numpy and opencv-python are required for snipping OCR")

    if config is not None:
        resolved_config = dataclasses.replace(config)
    else:
        resolved_config = load_config()
    if resolved_config.enable_gpu and not _gpu_available():
        print("GPU not detected. Falling back to CPU for snipping OCR.")
        resolved_config.enable_gpu = False
    service = OCRService(resolved_config)
    engine = getattr(service, "engine", None)
    if isinstance(engine, DummyOCREngine):
        print(
            "Warning: PaddleOCR dependencies are missing. The snipping tool "
            "will return placeholder text until paddlepaddle and paddleocr "
            "are installed."
        )

    selector = RegionSelector()
    selection = selector.select()
    if selection is None or not selection.is_valid:
        print("Selection cancelled.")
        return None

    image = _capture_region(selection)
    if image is None:
        print("Selection was too small to capture.")
        return None

    payload = _encode_image(image)
    response = service.process_image(payload)
    text = response.text.strip()
    _copy_to_clipboard(text)
    duration_ms = response.meta.get("duration_ms") if response.meta else None
    if isinstance(duration_ms, (int, float)):
        duration = float(duration_ms)
    else:
        duration = None
    _show_popup(text, duration)
    return text or None


def main() -> None:
    """Allow running via ``python -m app.snipping``."""

    try:
        run_snipping_ocr()
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"Snipping OCR failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - module execution helper
    main()

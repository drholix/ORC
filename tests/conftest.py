import base64
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OCR_FAKE_ENGINE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """Ensure the synthetic sample image exists for tests without shipping binaries."""
    output_dir = ROOT / "sample_data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_text.png"
    if not output_path.exists():
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            png_bytes = base64.b64decode(_SMALL_SAMPLE_PNG)
            output_path.write_bytes(png_bytes)
        else:
            canvas = np.full((320, 960, 3), 255, dtype=np.uint8)
            cv2.putText(
                canvas,
                "Halo PaddleOCR!",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Fast Laptop OCR 2025",
                (40, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (40, 40, 40),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "ID & EN text demo",
                (40, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (80, 80, 80),
                2,
                cv2.LINE_AA,
            )
            if not cv2.imwrite(str(output_path), canvas):
                raise RuntimeError("Failed to create synthetic sample image")
    return output_path


_SMALL_SAMPLE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAASwAAAA8CAYAAAC6sZ0JAAAACXBIWXMAAAsTAAALEwEAmpwYAAAB"
    "9UlEQVR4nO3WsQ3CQBCF4VuAAlIgAlIgAhIgAhLAbEniXHKTYICz6NzeZFNzjWeaO4/Ffn2cN9sN"
    "AAAAAAAAAADgbyfr6Vvv7em7vGN/39/nu/NVZ/35m2991k++9Q3l7c6We7bb3P8b7fVnY73utl/v"
    "zbu9X9b7vmvTfb9h90n7vkvb7/t8Xd/tv2n3ifveS9vv+3xd3+2/afdu+56Zfd2z7X+u+V7Zfd2z7"
    "X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V"
    "7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2"
    "z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u"
    "+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zf"
    "d2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X"
    "+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Zfd2z7X+u+V7Z"
    "fd2z7X+u+V7Y8b7+6H3XG/3vXb/13a/9q2+71sfe/AAAAAAAAAAAAPwOfj6AMKODFUsAAAAASUVOR"
    "K5CYII="
)

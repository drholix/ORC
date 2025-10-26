"""Utility to generate a synthetic sample image without shipping binary assets."""
from __future__ import annotations

import base64
from pathlib import Path


def build_sample_image(width: int = 960, height: int = 320):
    """Create a white canvas with contrasting Indonesian/English text."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) and NumPy are required to build the sample image."
        ) from exc

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
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
    return canvas


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "sample_data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "sample_text.png"

    try:
        image = build_sample_image()
    except RuntimeError:
        output_path.write_bytes(base64.b64decode(_SMALL_SAMPLE_PNG))
        print(
            "OpenCV not available; wrote fallback sample image instead. "
            "Install opencv-python-headless for the high-contrast demo."
        )
    else:
        import cv2  # type: ignore

        if cv2.imwrite(str(output_path), image):
            print(f"Sample image written to {output_path}")
        else:
            raise RuntimeError("Failed to write sample image")


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


if __name__ == "__main__":
    main()

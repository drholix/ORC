"""Utility helpers for working with images across the OCR pipeline."""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency at runtime
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency at runtime
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


def ensure_uint8(image: Any) -> Any:
    """Coerce the image to ``uint8`` where possible.

    ``cv2`` expects 8-bit images for most operations. If numpy is unavailable we
    simply return the image untouched and let the caller decide how to proceed.
    """

    if np is None:
        return image
    if isinstance(image, np.ndarray):  # type: ignore[arg-type]
        if image.dtype != np.uint8:
            return image.astype(np.uint8, copy=False)
        return image
    try:
        array = np.asarray(image)
    except Exception:  # pragma: no cover - non-array inputs
        return image
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    return array


def ensure_bgr_image(image: Any) -> Any:
    """Guarantee that the provided image is a 3-channel BGR array.

    PaddleOCR (via PaddleX) expects an ``H×W×3`` ``uint8`` array. This helper
    normalises grayscale inputs as well as BGRA/RGBA frames produced by MSS on
    Windows. When OpenCV or NumPy are unavailable we fall back to returning the
    original object so the caller can decide how to handle it.
    """

    if cv2 is None or np is None:
        return image
    array = ensure_uint8(image)
    if not hasattr(array, "shape"):
        return array
    if array.ndim == 2:  # type: ignore[attr-defined]
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:  # pragma: no cover - unexpected dimensionality
        raise ValueError(f"Unsupported image shape: {getattr(array, 'shape', None)}")
    channels = array.shape[2]
    if channels == 3:
        return array
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    # Some edge cases (e.g. NV12) require a different handling. Rather than
    # guessing we raise so the caller can fall back to a safer path.
    raise ValueError(f"Unsupported channel count: {channels}")


__all__ = ["ensure_bgr_image", "ensure_uint8"]

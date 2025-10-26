"""Image preprocessing utilities using OpenCV with graceful fallbacks."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from .config import OCRConfig
from .image_utils import ensure_bgr_image

LOGGER = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class PreprocessResult:
    image: Any
    steps: List[str]
    metadata: Dict[str, float]
    original: Any | None = None
    processed: Any | None = None


class Preprocessor:
    """Apply a deterministic preprocessing pipeline for OCR."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = LOGGER.bind(module="preprocess")

    def run(self, image: Any) -> PreprocessResult:
        """Run the full preprocessing pipeline."""

        steps: List[str] = []
        metadata: Dict[str, float] = {}
        if cv2 is None or np is None:
            steps.append("noop")
            return PreprocessResult(
                image=image,
                steps=steps,
                metadata=metadata,
                original=image,
                processed=image,
            )

        working = self._ensure_max_size(self._as_array(image))
        steps.append("ensure_max_size")

        color_reference = ensure_bgr_image(working)
        steps.append("ensure_bgr")

        profile = (self.config.preprocess_profile or "document").lower()
        if profile in {"none", "raw"}:
            return self._finalize(color_reference, color_reference, steps, metadata)
        if profile in {"screen", "ui", "light"}:
            return self._run_screen_profile(color_reference, steps, metadata)
        return self._run_document_profile(color_reference, steps, metadata)

    def _run_screen_profile(
        self,
        color_reference: Any,
        steps: List[str],
        metadata: Dict[str, float],
    ) -> PreprocessResult:
        processed = color_reference
        if cv2 is not None:
            start = time.perf_counter()
            processed = self._screen_enhance(color_reference)
            metadata["screen_enhance_ms"] = (time.perf_counter() - start) * 1000
            steps.append("screen_enhance")
        return self._finalize(color_reference, processed, steps, metadata)

    def _run_document_profile(
        self,
        color_reference: Any,
        steps: List[str],
        metadata: Dict[str, float],
    ) -> PreprocessResult:
        start = time.perf_counter()
        gray = self._to_gray(color_reference)
        metadata["grayscale_ms"] = (time.perf_counter() - start) * 1000
        steps.append("grayscale")

        start = time.perf_counter()
        angle = self._estimate_skew(gray)
        rotated = self._rotate(gray, angle)
        metadata["deskew_ms"] = (time.perf_counter() - start) * 1000
        metadata["deskew_angle"] = angle
        steps.append("deskew")

        start = time.perf_counter()
        rotated = self._auto_rotate(rotated)
        metadata["autorotate_ms"] = (time.perf_counter() - start) * 1000
        steps.append("auto_rotate")

        start = time.perf_counter()
        denoised = cv2.fastNlMeansDenoising(rotated, h=7)  # type: ignore[arg-type]
        metadata["denoise_ms"] = (time.perf_counter() - start) * 1000
        steps.append("denoise")

        start = time.perf_counter()
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        metadata["unsharp_ms"] = (time.perf_counter() - start) * 1000
        steps.append("unsharp_mask")

        start = time.perf_counter()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        metadata["clahe_ms"] = (time.perf_counter() - start) * 1000
        steps.append("clahe")

        start = time.perf_counter()
        threshold = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15,
        )
        metadata["binarize_ms"] = (time.perf_counter() - start) * 1000
        steps.append("adaptive_threshold")

        start = time.perf_counter()
        morphed = self._morphology(threshold)
        metadata["morph_ms"] = (time.perf_counter() - start) * 1000
        steps.append("morphology")

        processed = ensure_bgr_image(morphed)
        return self._finalize(color_reference, processed, steps, metadata)

    def _finalize(
        self,
        color_reference: Any,
        processed: Any,
        steps: List[str],
        metadata: Dict[str, float],
    ) -> PreprocessResult:
        if np is not None and hasattr(color_reference, "shape"):
            metadata.setdefault(
                "original_shape", list(color_reference.shape)  # type: ignore[index]
            )
            metadata.setdefault(
                "original_mean", float(color_reference.mean())  # type: ignore[arg-type]
            )
        if np is not None and hasattr(processed, "shape"):
            metadata.setdefault(
                "processed_shape", list(processed.shape)  # type: ignore[index]
            )
            metadata.setdefault(
                "processed_mean", float(processed.mean())  # type: ignore[arg-type]
            )
        return PreprocessResult(
            image=color_reference,
            steps=steps,
            metadata=metadata,
            original=color_reference,
            processed=processed,
        )

    def _screen_enhance(self, image: Any) -> Any:
        if cv2 is None:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _ensure_max_size(self, image: Any) -> Any:
        if not hasattr(image, "shape"):
            return image
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.config.max_image_size:
            return image
        scale = self.config.max_image_size / max_dim
        self.logger.info("downscale", scale=scale)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def _as_array(self, image: Any) -> Any:
        if np is None:
            return image
        if isinstance(image, np.ndarray):
            return image
        if hasattr(image, "__array__"):
            return np.asarray(image, dtype=np.uint8)
        if isinstance(image, list):
            try:
                return np.asarray(image, dtype=np.uint8)
            except TypeError:
                return np.asarray(image)
        return image

    def _to_gray(self, image: Any) -> Any:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _estimate_skew(self, gray: Any) -> float:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        if lines is None:
            return 0.0
        angles: List[float] = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            if 80 < angle < 100 or 260 < angle < 280:
                continue
            angles.append(angle - 90)
        if not angles:
            return 0.0
        return float(np.median(angles))

    def _auto_rotate(self, gray: Any) -> Any:
        candidates = {
            0: gray,
            90: cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(gray, cv2.ROTATE_180),
            270: cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        scores = {angle: self._projection_score(img) for angle, img in candidates.items()}
        best_angle = max(scores, key=scores.get)
        if best_angle != 0:
            self.logger.info("autorotate", angle=best_angle)
        return candidates[best_angle]

    def _projection_score(self, gray: Any) -> float:
        projection = np.sum(gray, axis=1)
        return float(np.var(projection))

    def _rotate(self, image: Any, angle: float) -> Any:
        if abs(angle) < 0.1:
            return image
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _morphology(self, image: Any) -> Any:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel_grid = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        gridless = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_grid, iterations=1)
        return gridless


def load_image_from_bytes(data: bytes):
    if cv2 is None or np is None:
        return None
    np_data = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_COLOR)


def load_image(path: str):
    if cv2 is None:
        return None
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image

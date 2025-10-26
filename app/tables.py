"""Table and form utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


@dataclass
class TableExtractionResult:
    cells: List[np.ndarray]
    bounding_boxes: List[Tuple[int, int, int, int]]


def detect_table_cells(image: np.ndarray) -> TableExtractionResult:
    """Detect table cells using morphological line detection."""

    if cv2 is None or np is None:
        return TableExtractionResult(cells=[], bounding_boxes=[])

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0.0)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells: List[np.ndarray] = []
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 500:
            continue
        cells.append(image[y : y + h, x : x + w])
        boxes.append((x, y, x + w, y + h))
    return TableExtractionResult(cells=cells, bounding_boxes=boxes)

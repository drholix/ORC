"""PDF utilities for the OCR pipeline."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def extract_pdf(path: Path, max_pages: Optional[int] = None) -> Tuple[List[bytes], List[str]]:
    """Convert a PDF into images and collect embedded text per page."""

    if PdfReader is None or convert_from_path is None:
        raise RuntimeError("PDF support requires pdf2image and pypdf dependencies")

    reader = PdfReader(str(path))
    embedded_text: List[str] = []
    pages = reader.pages[: max_pages or len(reader.pages)]
    for page in pages:
        try:
            embedded_text.append(page.extract_text() or "")
        except Exception:  # pragma: no cover - fallback path
            embedded_text.append("")

    images = convert_from_path(str(path), fmt="png")
    if max_pages is not None:
        images = images[:max_pages]

    image_bytes: List[bytes] = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes.append(buffer.getvalue())
    return image_bytes, embedded_text

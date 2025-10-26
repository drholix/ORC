"""High level orchestration of the OCR pipeline."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
import structlog

from .cache import OCRCache
from .config import OCRConfig, load_config
from .inference import create_engine
from .pdf import extract_pdf
from .postprocess import PostProcessor
from .preprocess import Preprocessor, load_image, load_image_from_bytes
from .tables import detect_table_cells

LOGGER = structlog.get_logger(__name__)

SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


@dataclass
class OCRResponse:
    text: str
    language: str
    blocks: List[Dict[str, object]]
    meta: Dict[str, object]


class RateLimiter:
    def __init__(self, capacity: int):
        self._semaphore = threading.Semaphore(capacity)

    def __enter__(self):
        self._semaphore.acquire()

    def __exit__(self, exc_type, exc, tb):
        self._semaphore.release()


class OCRService:
    """Main entry point for performing OCR on different sources."""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or load_config()
        self.logger = LOGGER.bind(module="service")
        self.cache = OCRCache(Path(self.config.cache_path))
        self.preprocessor = Preprocessor(self.config)
        self.postprocessor = PostProcessor(self.config)
        self.engine = create_engine(self.config)
        self.ratelimiter = RateLimiter(self.config.rate_limit)

    def _download(self, url: str) -> bytes:
        if requests is None:
            raise RuntimeError("requests library is required for URL inputs")
        with self.ratelimiter:
            response = requests.get(url, timeout=self.config.download_timeout, stream=True)
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(self.config.download_chunk_size):
                if chunk:
                    size += len(chunk)
                    if size > self.config.max_file_size_mb * 1024 * 1024:
                        raise ValueError("File too large")
                    chunks.append(chunk)
            return b"".join(chunks)

    def _load_source(self, source: Union[str, bytes, Path]) -> Tuple[object, bytes, Dict[str, object]]:
        meta: Dict[str, object] = {}
        if isinstance(source, (str, Path)) and str(source).lower().startswith("http"):
            data = self._download(str(source))
            image = load_image_from_bytes(data)
            meta["source"] = str(source)
        elif isinstance(source, Path) or isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError("File too large")
            data = path.read_bytes()
            image = load_image(str(path))
            meta["path"] = str(path)
        elif isinstance(source, bytes):
            data = source
            image = load_image_from_bytes(source)
            meta["source"] = "bytes"
        else:
            raise TypeError("Unsupported source type")
        if image is None:
            image = self._placeholder_image()
        meta["image_size"] = self._image_size(image)
        return image, data, meta

    def process_image(self, source: Union[str, bytes, Path]) -> OCRResponse:
        image, data, meta = self._load_source(source)
        key = self.cache.compute_key(data)

        cached = self.cache.get(key)
        if cached:
            return OCRResponse(
                text=cached["text"],
                language=cached.get("language", "mix"),
                blocks=cached.get("blocks", []),
                meta=cached.get("meta", {}),
            )

        start = time.perf_counter()
        preprocess_result = self.preprocessor.run(image)
        engine_output = self.engine.infer(preprocess_result.image, self.config.languages)
        post_result = self.postprocessor.run({
            "text": engine_output.text,
            "blocks": engine_output.blocks,
        })
        duration_ms = (time.perf_counter() - start) * 1000

        table_meta: Dict[str, object] = {}
        if self.config.table_mode:
            tables = detect_table_cells(preprocess_result.image)
            table_meta["tables"] = [
                {"bbox": bbox, "cells": len(tables.cells)} for bbox in tables.bounding_boxes
            ]

        meta.update(
            {
                "duration_ms": duration_ms,
                "device": "gpu" if self.config.enable_gpu else "cpu",
                "engine": {
                    "name": self.config.engine,
                    "ocr_version": self.config.ocr_version,
                    "language": engine_output.language,
                },
                "pipeline": preprocess_result.steps,
                "pipeline_metadata": preprocess_result.metadata,
                "engine_duration_ms": engine_output.duration_ms,
                "postprocess_ms": post_result.metadata.get("postprocess_ms"),
                **table_meta,
            }
        )

        if self.config.handwriting_mode:
            warnings = list(meta.get("warnings", []))
            warnings.append(self.config.handwriting_warning)
            meta["warnings"] = warnings

        result = {
            "text": post_result.text,
            "language": post_result.language,
            "blocks": post_result.blocks,
            "meta": meta,
        }
        self.cache.set(key, result)
        return OCRResponse(**result)

    def process_batch(self, sources: Sequence[Union[str, Path]]) -> List[OCRResponse]:
        responses: List[OCRResponse] = []
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            future_map = {
                executor.submit(self.process_image, source): source for source in sources
            }
            for future in as_completed(future_map, timeout=self.config.batch_timeout):
                responses.append(future.result())
        return responses

    def process_pdf(self, path: Union[str, Path]) -> List[OCRResponse]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        image_bytes, embedded_texts = extract_pdf(pdf_path, self.config.max_pdf_pages)
        responses: List[OCRResponse] = []
        for page_index, bytes_ in enumerate(image_bytes):
            response = self.process_image(bytes_)
            embedded_text = embedded_texts[page_index] if page_index < len(embedded_texts) else ""
            meta = dict(response.meta)
            meta["page_index"] = page_index
            meta["source_pdf"] = str(pdf_path)
            text = response.text
            if embedded_text and self.config.pdf_text:
                text = f"{embedded_text}\n--- OCR ---\n{text}"
            responses.append(
                OCRResponse(
                    text=text,
                    language=response.language,
                    blocks=response.blocks,
                    meta=meta,
                )
            )
        return responses

    def process_path(self, path: Union[str, Path]) -> List[OCRResponse]:
        path_obj = Path(path)
        if path_obj.is_dir():
            images = [
                p
                for p in sorted(path_obj.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_TYPES
            ]
            return self.process_batch(images)
        if path_obj.suffix.lower() == ".pdf":
            return self.process_pdf(path_obj)
        return [self.process_image(path_obj)]

    def warm_cache(self, paths: Sequence[Union[str, Path]]) -> None:
        for path in paths:
            try:
                self.process_image(path)
            except Exception as exc:  # pragma: no cover - warmup best effort
                self.logger.warning("warm_cache_failed", path=str(path), error=str(exc))

    def _placeholder_image(self) -> List[List[List[int]]]:
        return [[[255, 255, 255] for _ in range(10)] for _ in range(10)]

    def _image_size(self, image: object) -> List[int]:
        if np is not None and hasattr(image, "shape"):
            return [int(image.shape[1]), int(image.shape[0])]  # type: ignore[index]
        if isinstance(image, list) and image and isinstance(image[0], list):
            width = len(image[0])
            height = len(image)
            return [width, height]
        return [0, 0]

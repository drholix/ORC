"""Inference engines for OCR."""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Protocol, Sequence

import structlog

from .config import OCRConfig

LOGGER = structlog.get_logger(__name__)

try:  # pragma: no cover - optional dependency for runtime execution
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is required only when running real OCR
    np = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from numpy.typing import NDArray
else:  # pragma: no cover - fallback for environments without numpy
    NDArray = Any


class OCRResult(Protocol):
    """Protocol describing a single OCR output record."""

    bbox: List[List[float]]
    text: str
    score: float


@dataclass
class EngineOutput:
    text: str
    blocks: List[Dict[str, Any]]
    duration_ms: float
    raw: Any
    language: str


class BaseOCREngine(Protocol):
    """Interface for interchangeable OCR engines."""

    def infer(self, image: NDArray, languages: Sequence[str]) -> EngineOutput:
        ...


class DummyOCREngine:
    """A lightweight OCR engine used for testing and CPU-only environments."""

    def __init__(self) -> None:
        self.logger = LOGGER.bind(engine="dummy")

    def infer(self, image: NDArray, languages: Sequence[str]) -> EngineOutput:  # noqa: D401
        start = time.perf_counter()
        height, width = self._dims(image)
        text = "dummy-ocr-output"
        block = {
            "bbox": [0, 0, width, height],
            "text": text,
            "confidence": 0.5,
            "block_id": 0,
            "line_id": 0,
        }
        self.logger.debug("dummy_infer", languages=list(languages))
        return EngineOutput(
            text=text,
            blocks=[block],
            duration_ms=(time.perf_counter() - start) * 1000,
            raw=None,
            language=languages[0] if languages else "en",
        )

    def _dims(self, image: Any) -> tuple[int, int]:
        if hasattr(image, "shape"):
            return int(image.shape[0]), int(image.shape[1])  # type: ignore[index]
        if isinstance(image, list) and image and isinstance(image[0], list):
            return len(image), len(image[0])
        return 0, 0


class PaddleOCREngine:
    """Wrapper around :class:`paddleocr.PaddleOCR`."""

    def __init__(self, config: OCRConfig):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:  # pragma: no cover - heavy dependency
            raise RuntimeError(
                "PaddleOCR is not installed. Please install paddleocr>=2.x."
            ) from exc

        use_gpu = config.enable_gpu and bool(os.environ.get("PADDLEOCR_USE_GPU", "1") == "1")
        lang = self._resolve_language(config.languages)
        ocr_kwargs: Dict[str, Any] = {
            "use_angle_cls": True,
            "lang": lang,
            "use_gpu": use_gpu,
            "show_log": False,
        }
        if config.ocr_version:
            ocr_kwargs["ocr_version"] = config.ocr_version
        if config.det_model_dir:
            ocr_kwargs["det_model_dir"] = config.det_model_dir
        if config.rec_model_dir:
            ocr_kwargs["rec_model_dir"] = config.rec_model_dir
        if config.cls_model_dir:
            ocr_kwargs["cls_model_dir"] = config.cls_model_dir
        if config.structure_version:
            ocr_kwargs["structure_version"] = config.structure_version

        self._engine_factory = PaddleOCR
        if "show_log" not in inspect.signature(PaddleOCR.__init__).parameters:
            ocr_kwargs.pop("show_log", None)
        self._ocr_kwargs = ocr_kwargs
        self.ocr = self._engine_factory(**self._ocr_kwargs)  # type: ignore[call-arg]
        self.config = config
        self.logger = LOGGER.bind(
            engine="paddle",
            gpu=use_gpu,
            lang=lang,
            ocr_version=config.ocr_version,
        )
        self.lang = lang

    def infer(self, image: NDArray, languages: Sequence[str]) -> EngineOutput:
        requested_lang = self._resolve_language(languages)
        if requested_lang and requested_lang != self.lang:
            self.logger.info(
                "paddle_lang_switch",
                previous=self.lang,
                requested=requested_lang,
            )
            self.lang = requested_lang
            self._ocr_kwargs["lang"] = requested_lang
            self.ocr = self._engine_factory(**self._ocr_kwargs)  # type: ignore[call-arg]
        if np is not None and not isinstance(image, np.ndarray):  # type: ignore[arg-type]
            image = np.asarray(image)  # type: ignore[assignment]
        start = time.perf_counter()
        results = self.ocr.ocr(image, cls=True)
        duration_ms = (time.perf_counter() - start) * 1000
        blocks: List[Dict[str, Any]] = []
        texts: List[str] = []
        block_id = 0
        line_id = 0
        for page in results:
            for bbox, info in page:
                text, score = info
                if score < self.config.min_confidence:
                    continue
                blocks.append(
                    {
                        "bbox": self._flatten_bbox(bbox),
                        "text": text,
                        "confidence": float(score),
                        "block_id": block_id,
                        "line_id": line_id,
                    }
                )
                texts.append(text)
                line_id += 1
            block_id += 1
        return EngineOutput(
            text="\n".join(texts),
            blocks=blocks,
            duration_ms=duration_ms,
            raw=results,
            language=self.lang,
        )

    @staticmethod
    def _resolve_language(languages: Sequence[str]) -> str:
        if not languages:
            return "en"
        normalized = [lang.lower().strip() for lang in languages if lang]
        mapping = {
            "en": "en",
            "english": "en",
            "id": "latin",
            "indonesian": "latin",
            "latin": "latin",
            "multi": "multilingual",
            "multilingual": "multilingual",
            "ar": "arabic",
            "arabic": "arabic",
            "ru": "cyrillic",
            "russian": "cyrillic",
        }
        for candidate in normalized:
            if candidate in mapping and candidate not in {"id", "indonesian"}:
                return mapping[candidate]
        if len(normalized) == 1 and normalized[0] in mapping:
            return mapping[normalized[0]]
        if len(normalized) > 1:
            return "latin"
        return mapping.get(normalized[0], normalized[0])

    @staticmethod
    def _flatten_bbox(bbox: Iterable[Iterable[float]]) -> List[float]:
        flat: List[float] = []
        for point in bbox:
            if isinstance(point, Iterable):
                flat.extend(float(coord) for coord in point)
            else:
                flat.append(float(point))
        return flat


def create_engine(config: OCRConfig, force_dummy: bool = False) -> BaseOCREngine:
    """Factory that returns a configured OCR engine."""

    if force_dummy or os.environ.get("OCR_FAKE_ENGINE") == "1":
        return DummyOCREngine()
    engine_choice = (config.engine or "paddle").lower()
    if engine_choice in {"dummy", "fake"}:
        return DummyOCREngine()
    if engine_choice in {"paddle", "paddleocr", "ppocr"}:
        return PaddleOCREngine(config)
    raise ValueError(f"Unsupported OCR engine '{config.engine}'")

"""Inference engines for OCR."""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Protocol, Sequence

import structlog

from .config import OCRConfig
from .image_utils import ensure_bgr_image

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
    device: str


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
            device="cpu",
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
        device_choice = "gpu" if use_gpu else "cpu"

        textline_enabled = bool(config.use_textline_orientation)
        angle_cls_setting = config.use_angle_cls
        if angle_cls_setting is None:
            angle_cls_setting = False
        if textline_enabled and angle_cls_setting:
            LOGGER.warning(
                "angle_cls_textline_conflict",
                message="Disabling angle classifier because textline orientation is enabled.",
            )
            angle_cls_setting = False

        ocr_kwargs: Dict[str, Any] = {
            "lang": lang,
            "use_gpu": use_gpu,
            "device": device_choice,
            "show_log": False,
        }
        if angle_cls_setting is True:
            ocr_kwargs["use_angle_cls"] = True
        elif config.use_angle_cls is False:
            ocr_kwargs["use_angle_cls"] = False
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
        if config.text_detection_model_name:
            ocr_kwargs["text_detection_model_name"] = config.text_detection_model_name
        if config.text_detection_model_dir:
            ocr_kwargs["text_detection_model_dir"] = config.text_detection_model_dir
        if config.text_recognition_model_name:
            ocr_kwargs["text_recognition_model_name"] = config.text_recognition_model_name
        if config.text_recognition_model_dir:
            ocr_kwargs["text_recognition_model_dir"] = config.text_recognition_model_dir

        detection_overrides = {
            "text_det_limit_side_len": config.text_det_limit_side_len,
            "det_limit_side_len": getattr(config, "det_limit_side_len", None),
            "text_det_limit_type": config.text_det_limit_type,
            "det_limit_type": getattr(config, "det_limit_type", None),
            "text_det_unclip_ratio": config.text_det_unclip_ratio,
            "det_db_unclip_ratio": getattr(config, "det_db_unclip_ratio", None),
            "text_det_box_thresh": config.text_det_box_thresh,
            "det_db_box_thresh": getattr(config, "det_db_box_thresh", None),
            "text_det_thresh": config.text_det_thresh,
            "det_db_thresh": getattr(config, "det_db_thresh", None),
            "text_rec_score_thresh": config.text_rec_score_thresh,
            "rec_score_thresh": getattr(config, "rec_score_thresh", None),
        }
        for key, value in detection_overrides.items():
            if value is not None:
                ocr_kwargs[key] = value

        textline_pref = config.use_textline_orientation
        if textline_pref is None:
            textline_pref = False

        doc_overrides = {
            "use_doc_preprocessor": config.use_doc_preprocessor,
            "use_doc_orientation_classify": config.use_doc_orientation_classify,
            "use_doc_unwarping": config.use_doc_unwarping,
            "use_textline_orientation": textline_pref,
        }
        for key, value in doc_overrides.items():
            if value is None:
                continue
            ocr_kwargs[key] = value

        self._engine_factory = PaddleOCR
        signature = inspect.signature(PaddleOCR.__init__)
        supported_params = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        }

        supported_params.discard("self")

        if "device" in signature.parameters:
            ocr_kwargs["device"] = "gpu" if use_gpu else "cpu"

        if "show_log" not in signature.parameters:
            ocr_kwargs.pop("show_log", None)

        if "use_gpu" not in signature.parameters:
            ocr_kwargs.pop("use_gpu", None)

        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            for key in list(ocr_kwargs.keys()):
                if key not in supported_params:
                    LOGGER.debug("drop_unsupported_param", param=key)
                    ocr_kwargs.pop(key, None)
        self.device = "gpu" if use_gpu else "cpu"
        self._ocr_kwargs = ocr_kwargs
        self._sync_feature_flags()
        self._runtime_call_kwargs: Dict[str, Any] = {}
        self._fallback_engine = None
        self._fallback_runtime_kwargs: Dict[str, Any] = {}
        self._gpu_retry_done = False
        self._language_failures: set[str] = set()
        self.ocr = self._create_engine_instance()
        self._refresh_runtime_call_kwargs()
        self.config = config
        current_lang = str(self._ocr_kwargs.get("lang", lang))
        self.logger = LOGGER.bind(
            engine="paddle",
            gpu=use_gpu,
            lang=current_lang,
            ocr_version=config.ocr_version,
            angle_cls=self.angle_cls_enabled,
            textline_orientation=self.textline_orientation_enabled,
        )
        self.lang = current_lang

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
            self.ocr = self._create_engine_instance()
            self._refresh_runtime_call_kwargs()
        if np is not None and not isinstance(image, np.ndarray):  # type: ignore[arg-type]
            image = np.asarray(image)  # type: ignore[assignment]
        image = ensure_bgr_image(image)
        start = time.perf_counter()
        results = self._call_ocr(image)
        duration_ms = (time.perf_counter() - start) * 1000
        blocks, texts = self._build_blocks(results)

        if not texts:
            fallback = self._retry_with_relaxed_settings(image)
            if fallback is not None:
                results, blocks, texts, fallback_duration = fallback
                duration_ms += fallback_duration

        if not texts:
            self.logger.warning(
                "paddle_empty_result",
                duration_ms=duration_ms,
                languages=list(languages),
                runtime_kwargs=list(self._runtime_call_kwargs.keys()),
            )
        return EngineOutput(
            text="\n".join(texts),
            blocks=blocks,
            duration_ms=duration_ms,
            raw=results,
            language=self.lang,
            device=self.device,
        )

    @staticmethod
    def _resolve_language(languages: Sequence[str]) -> str:
        if not languages:
            return "en"
        normalized = [lang.lower().strip() for lang in languages if lang]
        if not normalized:
            return "en"
        auto_tokens = {"auto", "detect", "auto-detect", "auto_detect"}
        latin_aliases = {
            "en",
            "english",
            "id",
            "indonesian",
            "latin",
            "latin-script",
            "mix",
        }
        filtered = [token for token in normalized if token not in auto_tokens]
        if not filtered:
            return "en"
        unique = set(filtered)
        if len(unique) > 1:
            if unique.issubset(latin_aliases):
                return "en"
            return "multilingual"
        candidate = filtered[0]
        mapping = {
            "en": "en",
            "english": "en",
            "latin": "en",
            "latin-script": "en",
            "mix": "en",
            "id": "en",
            "indonesian": "en",
            "multi": "multilingual",
            "multilingual": "multilingual",
            "ar": "arabic",
            "arabic": "arabic",
            "ru": "cyrillic",
            "russian": "cyrillic",
        }
        return mapping.get(candidate, candidate)

    @staticmethod
    def _flatten_bbox(bbox: Iterable[Iterable[float]]) -> List[float]:
        flat: List[float] = []
        for point in bbox:
            if isinstance(point, Iterable):
                flat.extend(float(coord) for coord in point)
            else:
                flat.append(float(point))
        return flat

    def _build_blocks(self, results: Any) -> tuple[List[Dict[str, Any]], List[str]]:
        blocks: List[Dict[str, Any]] = []
        texts: List[str] = []
        block_id = 0
        for page in results:
            line_id = 0
            for entry in page:
                parsed = self._parse_page_entry(entry)
                if parsed is None:
                    self.logger.debug(
                        "skip_ocr_entry",
                        entry_type=type(entry).__name__,
                    )
                    continue
                bbox, text, score = parsed
                if score < self.config.min_confidence or not text:
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
        return blocks, texts

    def _retry_with_relaxed_settings(
        self, image: NDArray
    ) -> tuple[Any, List[Dict[str, Any]], List[str], float] | None:
        """Attempt secondary OCR passes when the first call yields no text."""

        strategies = [self._retry_with_rgb, self._retry_with_relaxed_engine]
        for strategy in strategies:
            outcome = strategy(image)
            if outcome is None:
                continue
            results, blocks, texts, duration_ms, label = outcome
            if texts:
                self.logger.info(
                    "paddle_retry_success",
                    strategy=label,
                    blocks=len(blocks),
                )
                return results, blocks, texts, duration_ms
        return None

    def _retry_with_rgb(
        self, image: NDArray
    ) -> tuple[Any, List[Dict[str, Any]], List[str], float, str] | None:
        if np is None:
            return None
        if not hasattr(image, "shape"):
            return None
        if getattr(image, "ndim", 0) != 3:
            return None
        start = time.perf_counter()
        rgb_image = image[..., ::-1]
        try:
            results = self.ocr.ocr(rgb_image, **self._runtime_call_kwargs)
        except Exception as exc:  # pragma: no cover - runtime guard
            self.logger.debug("paddle_retry_rgb_failed", error=str(exc))
            return None
        duration_ms = (time.perf_counter() - start) * 1000
        blocks, texts = self._build_blocks(results)
        if not texts:
            return None
        return results, blocks, texts, duration_ms, "rgb"

    def _retry_with_relaxed_engine(
        self, image: NDArray
    ) -> tuple[Any, List[Dict[str, Any]], List[str], float, str] | None:
        engine = self._get_relaxed_engine()
        if engine is None:
            return None
        start = time.perf_counter()
        try:
            results = engine.ocr(image, **self._fallback_runtime_kwargs)
        except Exception as exc:  # pragma: no cover - runtime guard
            self.logger.debug("paddle_relaxed_infer_failed", error=str(exc))
            return None
        duration_ms = (time.perf_counter() - start) * 1000
        blocks, texts = self._build_blocks(results)
        if not texts:
            return None
        return results, blocks, texts, duration_ms, "relaxed"

    def _get_relaxed_engine(self):  # type: ignore[no-untyped-def]
        if self._fallback_engine is not None:
            return self._fallback_engine
        relaxed_kwargs = dict(self._ocr_kwargs)
        relaxed_kwargs.update(
            {
                "lang": "en" if self.lang in {"latin", "multilingual"} else self.lang,
                "text_det_box_thresh": min(
                    float(relaxed_kwargs.get("text_det_box_thresh", 0.5) or 0.5), 0.4
                ),
                "text_det_thresh": min(
                    float(relaxed_kwargs.get("text_det_thresh", 0.2) or 0.2), 0.15
                ),
                "text_det_unclip_ratio": max(
                    float(relaxed_kwargs.get("text_det_unclip_ratio", 1.5) or 1.5), 2.0
                ),
                "text_rec_score_thresh": min(
                    float(relaxed_kwargs.get("text_rec_score_thresh", 0.5) or 0.5), 0.3
                ),
                "use_doc_preprocessor": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            }
        )
        relaxed_kwargs.pop("use_angle_cls", None)
        try:
            engine = self._engine_factory(**relaxed_kwargs)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - runtime guard
            self.logger.debug("paddle_relaxed_init_failed", error=str(exc))
            self._fallback_engine = None
            self._fallback_runtime_kwargs = {}
            return None
        self._fallback_engine = engine
        self._fallback_runtime_kwargs = self._compute_runtime_kwargs(engine)
        # ``cls`` can reintroduce the unexpected keyword errors on 3.2, so drop it.
        self._fallback_runtime_kwargs.pop("cls", None)
        return engine

    @staticmethod
    def _compute_runtime_kwargs(engine: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        ocr_method = getattr(engine, "ocr", None)
        if ocr_method is None:
            return kwargs
        try:
            signature = inspect.signature(ocr_method)
        except (TypeError, ValueError):  # pragma: no cover - signature unavailable
            return kwargs
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "cls" in signature.parameters or accepts_kwargs:
            kwargs["cls"] = True
        return kwargs

    def _create_engine_instance(self):  # type: ignore[no-untyped-def]
        attempts = 0
        while True:
            try:
                return self._engine_factory(**self._ocr_kwargs)  # type: ignore[call-arg]
            except ModuleNotFoundError as exc:  # pragma: no cover - heavy optional dependency
                if exc.name == "paddle":
                    raise RuntimeError(
                        "PaddleOCR requires the `paddlepaddle` package. Install the CPU build with"
                        " `pip install paddlepaddle` (or the GPU build with `pip install"
                        " paddlepaddle-gpu`)."
                    ) from exc
                raise
            except ValueError as exc:
                unknown = self._extract_unknown_argument(str(exc))
                if unknown and unknown in self._ocr_kwargs and attempts < 5:
                    LOGGER.debug("drop_unknown_param", param=unknown)
                    self._ocr_kwargs.pop(unknown, None)
                    self._sync_feature_flags()
                    attempts += 1
                    continue
                if (
                    self._handle_mutually_exclusive_error(str(exc))
                    and attempts < 5
                ):
                    attempts += 1
                    continue
                if (
                    self._handle_language_unavailable_error(str(exc))
                    and attempts < 5
                ):
                    attempts += 1
                    continue
                if self._maybe_retry_without_gpu(exc):
                    attempts += 1
                    continue
                raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc
            except TypeError as exc:
                unknown = self._extract_unknown_argument(str(exc))
                if unknown and unknown in self._ocr_kwargs and attempts < 5:
                    LOGGER.debug("drop_unknown_param", param=unknown)
                    self._ocr_kwargs.pop(unknown, None)
                    self._sync_feature_flags()
                    attempts += 1
                    continue
                if (
                    self._handle_mutually_exclusive_error(str(exc))
                    and attempts < 5
                ):
                    attempts += 1
                    continue
                if (
                    self._handle_language_unavailable_error(str(exc))
                    and attempts < 5
                ):
                    attempts += 1
                    continue
                if self._maybe_retry_without_gpu(exc):
                    attempts += 1
                    continue
                raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc
            except Exception as exc:  # pragma: no cover - runtime safeguard
                if self._maybe_retry_without_gpu(exc):
                    attempts += 1
                    continue
                raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc

    @staticmethod
    def _extract_unknown_argument(message: str) -> str | None:
        markers = ["Unknown argument", "unexpected keyword argument"]
        tail = None
        for marker in markers:
            if marker in message:
                tail = message.split(marker, 1)[1]
                break
        if tail is None:
            return None
        # Handle messages like "Unknown argument: use_gpu" or TypeError variants.
        if ":" in tail:
            candidate = tail.split(":", 1)[1]
        else:
            candidate = tail
        candidate = candidate.strip()
        for delimiter in (" ", "\n", ".", ","):
            if delimiter in candidate:
                candidate = candidate.split(delimiter, 1)[0]
        return candidate.strip("'\" ") or None

    def _handle_mutually_exclusive_error(self, message: str) -> bool:
        lowered = message.lower()
        if "mutually exclusive" not in lowered:
            return False
        changed = False
        if "use_textline_orientation" in self._ocr_kwargs:
            LOGGER.warning("paddle_disable_textline_orientation", error=message)
            self._ocr_kwargs.pop("use_textline_orientation", None)
            changed = True
        if "use_angle_cls" in self._ocr_kwargs:
            LOGGER.warning("paddle_disable_angle_cls", error=message)
            self._ocr_kwargs.pop("use_angle_cls", None)
            changed = True
        if changed:
            self._sync_feature_flags()
        return changed

    def _sync_feature_flags(self) -> None:
        self.angle_cls_enabled = bool(self._ocr_kwargs.get("use_angle_cls", False))
        self.textline_orientation_enabled = bool(
            self._ocr_kwargs.get("use_textline_orientation", False)
        )

    def _refresh_runtime_call_kwargs(self) -> None:
        """Determine runtime kwargs (like ``cls``) supported by the engine."""

        self._runtime_call_kwargs = {}
        ocr_method = getattr(self.ocr, "ocr", None)
        if ocr_method is None:
            return
        try:
            signature = inspect.signature(ocr_method)
        except (TypeError, ValueError):  # pragma: no cover - signature unavailable
            return
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "cls" in signature.parameters or accepts_kwargs:
            self._runtime_call_kwargs["cls"] = True

    def _parse_page_entry(self, entry: Any) -> tuple[Any, str, float] | None:
        """Normalise PaddleOCR outputs across versions.

        PaddleOCR 3.x may return tuples, lists, or dict payloads per detection.
        This helper extracts a ``(bbox, text, score)`` triple whenever possible.
        """

        bbox: Any
        text: str = ""
        score: float = 1.0

        if isinstance(entry, dict):
            bbox = (
                entry.get("bbox")
                or entry.get("box")
                or entry.get("points")
                or entry.get("poly")
            )
            text = str(entry.get("text") or entry.get("value") or "")
            raw_score = entry.get("score")
            if raw_score is None:
                raw_score = entry.get("confidence")
            if isinstance(raw_score, (int, float)):
                score = float(raw_score)
            if bbox is None:
                return None
            return bbox, text, score

        if isinstance(entry, (list, tuple)) and entry:
            bbox = entry[0]
            remainder = entry[1:]
            if not remainder:
                return None

            info = remainder[0]

            def _extract_text_score(value: Any) -> tuple[str, float]:
                candidate_text = ""
                candidate_score = 1.0
                if isinstance(value, (list, tuple)) and value:
                    candidate_text = str(value[0])
                    if len(value) > 1 and isinstance(value[1], (int, float)):
                        candidate_score = float(value[1])
                elif isinstance(value, dict):
                    candidate_text = str(
                        value.get("text") or value.get("value") or ""
                    )
                    raw = value.get("score")
                    if raw is None:
                        raw = value.get("confidence")
                    if isinstance(raw, (int, float)):
                        candidate_score = float(raw)
                elif isinstance(value, str):
                    candidate_text = value
                return candidate_text, candidate_score

            text, score = _extract_text_score(info)

            if not text and len(remainder) > 1:
                fallback_text, fallback_score = _extract_text_score(remainder[1])
                if fallback_text:
                    text = fallback_text
                    score = fallback_score

            return bbox, text, score

        return None

    def _call_ocr(self, image: NDArray):  # type: ignore[no-untyped-def]
        attempts = 0
        while True:
            try:
                return self.ocr.ocr(image, **self._runtime_call_kwargs)
            except TypeError as exc:
                unknown = self._extract_unknown_argument(str(exc))
                if (
                    unknown == "cls"
                    and "cls" in self._runtime_call_kwargs
                    and attempts < 3
                ):
                    LOGGER.debug("drop_runtime_param", param="cls")
            self._runtime_call_kwargs.pop("cls", None)
            attempts += 1
            continue
        raise

    def _handle_language_unavailable_error(self, message: str) -> bool:
        lowered = message.lower()
        if "language" not in lowered:
            return False
        if "no models" not in lowered and "not support" not in lowered:
            return False
        current = str(self._ocr_kwargs.get("lang") or "").strip()
        if not current:
            return False
        fallback = self._next_language_candidate(current.lower())
        if fallback is None:
            return False
        LOGGER.warning(
            "paddle_language_fallback",
            previous=current,
            fallback=fallback,
            error=message,
        )
        self._language_failures.add(current.lower())
        self._ocr_kwargs["lang"] = fallback
        self.lang = fallback
        return True

    def _next_language_candidate(self, current: str) -> str | None:
        fallback_chain = {
            "latin": ["en"],
            "multilingual": ["en"],
        }
        tried = {lang.lower() for lang in self._language_failures}
        tried.add(current)
        for candidate in fallback_chain.get(current, []):
            if candidate.lower() not in tried:
                return candidate
        return None

    def _maybe_retry_without_gpu(self, exc: Exception) -> bool:
        """Fallback to CPU when GPU initialization fails."""

        if not self._ocr_kwargs.get("use_gpu"):
            return False
        if self._gpu_retry_done:
            return False
        LOGGER.warning("paddle_gpu_retry_cpu", error=str(exc))
        self._ocr_kwargs["use_gpu"] = False
        if "device" in self._ocr_kwargs:
            self._ocr_kwargs["device"] = "cpu"
        self.device = "cpu"
        self._gpu_retry_done = True
        return True


def create_engine(config: OCRConfig, force_dummy: bool = False) -> BaseOCREngine:
    """Factory that returns a configured OCR engine."""

    if force_dummy or os.environ.get("OCR_FAKE_ENGINE") == "1":
        return DummyOCREngine()
    engine_choice = (config.engine or "paddle").lower()
    if engine_choice in {"dummy", "fake"}:
        return DummyOCREngine()
    if engine_choice in {"paddle", "paddleocr", "ppocr"}:
        try:
            return PaddleOCREngine(config)
        except RuntimeError as exc:
            LOGGER.warning("paddle_engine_unavailable", error=str(exc))
            return DummyOCREngine()
    raise ValueError(f"Unsupported OCR engine '{config.engine}'")

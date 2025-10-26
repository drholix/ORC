"""Post-processing utilities for OCR outputs."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import structlog

from .config import OCRConfig

LOGGER = structlog.get_logger(__name__)

DATE_REGEX = re.compile(r"(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})")
NUMBER_REGEX = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*)(?:[.,](\d+))?(?!\d)")
CURRENCY_REGEX = re.compile(r"(?:rp|idr)\s*(\d+[\d.,]*)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_REGEX = re.compile(r"https?://\S+")


@dataclass
class PostprocessResult:
    text: str
    blocks: List[Dict[str, object]]
    language: str
    metadata: Dict[str, float]


class SpellCorrector:
    """Optional spell correction using pyspellchecker if available."""

    def __init__(self, languages: Iterable[str]):
        try:
            from spellchecker import SpellChecker  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self.spell = None
        else:
            self.spell = SpellChecker(language=None, case_sensitive=False)
            self.spell.word_frequency.load_words([])
            for lang in languages:
                try:
                    self.spell.word_frequency.load_text_file(f"lexicons/{lang}.txt")
                except Exception:  # pragma: no cover - optional
                    continue

    def correct(self, text: str) -> str:
        if not self.spell:
            return text
        corrected_words = []
        for token in text.split():
            corrected_words.append(self.spell.correction(token) or token)
        return " ".join(corrected_words)


class LanguageDetector:
    """Minimal language detector based on character heuristics."""

    def detect(self, text: str, default: str = "mix") -> str:
        if not text:
            return default
        text_lower = text.lower()
        id_score = sum(text_lower.count(ch) for ch in "khsyai")
        en_score = sum(text_lower.count(ch) for ch in "thear")
        if abs(id_score - en_score) < 3:
            return "mix"
        return "id" if id_score > en_score else "en"


class PostProcessor:
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = LOGGER.bind(module="postprocess")
        self.detector = LanguageDetector()
        self.corrector = SpellCorrector(config.languages) if config.use_spell_correction else None

    def run(self, engine_output: Dict[str, object]) -> PostprocessResult:
        start = time.perf_counter()
        text = str(engine_output.get("text", ""))
        blocks = list(engine_output.get("blocks", []))

        if self.corrector:
            text = self.corrector.correct(text)

        language = self.detector.detect(text)
        normalized, extra_meta = self._normalize(text)
        text = normalized
        duration_ms = (time.perf_counter() - start) * 1000
        metadata = {"postprocess_ms": duration_ms, **extra_meta}

        return PostprocessResult(text=text, blocks=blocks, language=language, metadata=metadata)

    def _normalize(self, text: str) -> tuple[str, Dict[str, object]]:
        meta: Dict[str, object] = {}
        text = DATE_REGEX.sub(
            lambda m: f"{int(m.group(1)):02d}-{int(m.group(2)):02d}-{int(m.group(3)):04d}",
            text,
        )
        text = NUMBER_REGEX.sub(lambda m: m.group(0).replace(".", "").replace(",", "."), text)
        text = CURRENCY_REGEX.sub(
            lambda m: f"IDR {m.group(1).replace('.', '').replace(',', '.')}",
            text,
        )
        emails = EMAIL_REGEX.findall(text)
        urls = URL_REGEX.findall(text)
        if emails:
            meta["emails"] = [email.lower() for email in emails]
            for email in emails:
                text = text.replace(email, email.lower())
        if urls:
            meta["urls"] = [url.lower() for url in urls]
            for url in urls:
                text = text.replace(url, url.lower())
        return text, meta


def merge_ocr_with_pdf_text(ocr_text: str, pdf_text: Optional[str]) -> str:
    if not pdf_text:
        return ocr_text
    if not ocr_text:
        return pdf_text
    return f"{pdf_text}\n--- OCR ---\n{ocr_text}"

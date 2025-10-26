"""Configuration management for the OCR service."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML unavailable
    yaml = None


@dataclass
class OCRConfig:
    """Dataclass representing the configurable options for the OCR pipeline."""

    languages: List[str] = field(default_factory=lambda: ["id", "en"])
    min_confidence: float = 0.6
    enable_gpu: bool = False
    engine: str = "paddle"
    ocr_version: Optional[str] = "PP-OCRv5"
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    cls_model_dir: Optional[str] = None
    structure_version: Optional[str] = None
    max_image_size: int = 2048
    table_mode: bool = False
    handwriting_mode: bool = False
    cache_path: str = "cache.sqlite3"
    pdf_text: bool = True
    threads: int = 4
    batch_timeout: float = 60.0
    request_timeout: float = 30.0
    rate_limit: int = 4
    max_file_size_mb: int = 20
    download_timeout: float = 10.0
    download_chunk_size: int = 1024 * 256
    pipeline_debug: bool = False
    use_spell_correction: bool = True
    enable_metrics: bool = True
    downscale_on_oom: bool = True
    max_pdf_pages: Optional[int] = None
    handwriting_warning: str = (
        "Handwriting recognition is experimental. Results may vary."
    )

    @property
    def languages_str(self) -> str:
        return ",".join(self.languages)

    @property
    def primary_language(self) -> str:
        if not self.languages:
            return "en"
        return self.languages[0]


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Optional[Path] = None) -> OCRConfig:
    """Load configuration from a YAML file if it exists."""

    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fp:
            raw = fp.read()
        if yaml:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        return OCRConfig(**data)
    return OCRConfig()


def save_default_config(path: Optional[Path] = None) -> Path:
    """Write a default configuration file for users to tweak."""

    config = OCRConfig()
    config_path = path or DEFAULT_CONFIG_PATH
    payload = dataclasses.asdict(config)
    with config_path.open("w", encoding="utf-8") as fp:
        if yaml:
            yaml.safe_dump(payload, fp, sort_keys=False)
        else:
            json.dump(payload, fp, indent=2)
    return config_path

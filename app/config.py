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


LEGACY_KEY_ALIASES = {
    "det_limit_side_len": "text_det_limit_side_len",
    "det_limit_type": "text_det_limit_type",
    "det_db_unclip_ratio": "text_det_unclip_ratio",
    "det_db_box_thresh": "text_det_box_thresh",
    "det_db_thresh": "text_det_thresh",
    "text_det_db_thresh": "text_det_thresh",
    "rec_score_thresh": "text_rec_score_thresh",
    "det_model_name": "text_detection_model_name",
    "rec_model_name": "text_recognition_model_name",
}


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
    text_detection_model_name: Optional[str] = "PP-OCRv5_server_det"
    text_detection_model_dir: Optional[str] = None
    text_recognition_model_name: Optional[str] = "en_PP-OCRv5_mobile_rec"
    text_recognition_model_dir: Optional[str] = None
    use_angle_cls: Optional[bool] = None
    text_det_limit_side_len: Optional[int] = 1080
    text_det_limit_type: Optional[str] = "max"
    text_det_unclip_ratio: Optional[float] = 1.5
    text_det_box_thresh: Optional[float] = 0.5
    text_det_thresh: Optional[float] = 0.2
    text_rec_score_thresh: Optional[float] = 0.5
    use_doc_preprocessor: Optional[bool] = False
    use_doc_orientation_classify: Optional[bool] = False
    use_doc_unwarping: Optional[bool] = False
    use_textline_orientation: Optional[bool] = False
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


def _apply_legacy_aliases(data: dict) -> dict:
    for old_key, new_key in LEGACY_KEY_ALIASES.items():
        if old_key in data and new_key not in data:
            data[new_key] = data[old_key]
        if old_key in data:
            data.pop(old_key, None)
    return data


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
        if not isinstance(data, dict):
            raise TypeError("Configuration file must define a mapping")
        data = _apply_legacy_aliases(dict(data))
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

import json
from pathlib import Path

from app.config import OCRConfig, load_config


def test_load_default(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('{"languages": ["id", "en"], "min_confidence": 0.9}', encoding="utf-8")
    config = load_config(path)
    assert isinstance(config, OCRConfig)
    assert config.min_confidence == 0.9
    assert config.languages == ["id", "en"]


def test_legacy_detection_aliases(tmp_path: Path) -> None:
    payload = {
        "det_limit_side_len": 640,
        "det_limit_type": "min",
        "det_db_unclip_ratio": 2.0,
        "det_db_box_thresh": 0.33,
        "det_db_thresh": 0.12,
        "rec_score_thresh": 0.44,
    }
    path = tmp_path / "legacy_config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(path)
    assert config.text_det_limit_side_len == 640
    assert config.text_det_limit_type == "min"
    assert config.text_det_unclip_ratio == 2.0
    assert config.text_det_box_thresh == 0.33
    assert config.text_det_db_thresh == 0.12
    assert config.text_rec_score_thresh == 0.44

"""Example of running cProfile over the OCR service."""
from __future__ import annotations

import cProfile
import pstats
from pathlib import Path

from app.service import OCRService


def profile_image(path: Path, output: Path) -> None:
    service = OCRService()
    profile = cProfile.Profile()
    profile.enable()
    service.process_image(path)
    profile.disable()
    profile.dump_stats(str(output))
    stats = pstats.Stats(profile)
    stats.sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile OCR pipeline")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ocr.prof"))
    args = parser.parse_args()
    profile_image(args.image, args.output)

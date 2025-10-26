"""Command line interface for the OCR pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .config import OCRConfig, load_config
from .service import OCRService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR image-to-text toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run OCR on a file or directory")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, default=Path("ocr_output.json"))
    run_parser.add_argument("--config", type=Path, default=None)
    run_parser.add_argument("--threads", type=int, default=None)
    run_parser.add_argument("--lang", type=str, default=None)
    run_parser.add_argument("--min-conf", type=float, default=None)
    run_parser.add_argument("--engine", type=str, choices=["paddle", "dummy"], default=None)
    run_parser.add_argument("--ocr-version", type=str, default=None)
    run_parser.add_argument("--det-model", type=str, default=None)
    run_parser.add_argument("--rec-model", type=str, default=None)
    run_parser.add_argument("--cls-model", type=str, default=None)
    run_parser.add_argument("--pdf-text", action="store_true", default=True)
    run_parser.add_argument("--no-pdf-text", action="store_false", dest="pdf_text")
    run_parser.add_argument("--handwriting", action="store_true", default=False)
    run_parser.add_argument("--table-mode", action="store_true", default=False)

    batch_parser = subparsers.add_parser("batch", help="Batch process a folder")
    batch_parser.add_argument("--folder", type=Path, required=True)
    batch_parser.add_argument("--output", type=Path, default=Path("batch_output.json"))
    batch_parser.add_argument("--limit", type=int, default=None)

    warm_parser = subparsers.add_parser("cache-warm", help="Warm the OCR cache")
    warm_parser.add_argument("--folder", type=Path, required=True)
    warm_parser.add_argument("--limit", type=int, default=None)

    snip_parser = subparsers.add_parser(
        "snip",
        help="Open a snipping overlay, run OCR, and copy text to the clipboard",
    )
    snip_parser.add_argument("--config", type=Path, default=None)
    snip_parser.add_argument("--lang", type=str, default=None)
    snip_parser.add_argument("--min-conf", type=float, default=None)
    snip_parser.add_argument("--engine", type=str, choices=["paddle", "dummy"], default=None)
    snip_parser.add_argument("--ocr-version", type=str, default=None)
    snip_parser.add_argument("--det-model", type=str, default=None)
    snip_parser.add_argument("--rec-model", type=str, default=None)
    snip_parser.add_argument("--cls-model", type=str, default=None)

    return parser


def handle_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.threads is not None:
        config.threads = args.threads
    if args.lang:
        config.languages = [chunk.strip() for chunk in args.lang.split(",") if chunk.strip()]
    if args.min_conf is not None:
        config.min_confidence = args.min_conf
    if args.engine:
        config.engine = args.engine
    if args.ocr_version:
        config.ocr_version = args.ocr_version
    if args.det_model:
        config.det_model_dir = args.det_model
    if args.rec_model:
        config.rec_model_dir = args.rec_model
    if args.cls_model:
        config.cls_model_dir = args.cls_model
    config.pdf_text = args.pdf_text
    config.handwriting_mode = args.handwriting
    config.table_mode = args.table_mode

    service = OCRService(config)
    responses = service.process_path(args.input)
    payload = [response.__dict__ for response in responses]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved OCR results to {args.output}")


def handle_batch(args: argparse.Namespace) -> None:
    service = OCRService()
    paths = [
        p
        for p in sorted(args.folder.iterdir())
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".webp"}
    ]
    if args.limit is not None:
        paths = paths[: args.limit]
    responses = service.process_batch(paths)
    payload = [response.__dict__ for response in responses]
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Processed {len(responses)} files into {args.output}")


def handle_cache_warm(args: argparse.Namespace) -> None:
    service = OCRService()
    paths = [p for p in sorted(args.folder.iterdir()) if p.is_file()]
    if args.limit is not None:
        paths = paths[: args.limit]
    service.warm_cache(paths)
    print(f"Warmed cache for {len(paths)} files")


def handle_snip(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.lang:
        config.languages = [chunk.strip() for chunk in args.lang.split(",") if chunk.strip()]
    if args.min_conf is not None:
        config.min_confidence = args.min_conf
    if args.engine:
        config.engine = args.engine
    if args.ocr_version:
        config.ocr_version = args.ocr_version
    if args.det_model:
        config.det_model_dir = args.det_model
    if args.rec_model:
        config.rec_model_dir = args.rec_model
    if args.cls_model:
        config.cls_model_dir = args.cls_model
    from .snipping import run_snipping_ocr

    run_snipping_ocr(config)


def _load_config(config_path: Optional[Path]) -> OCRConfig:
    return load_config(config_path) if config_path else load_config()


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        handle_run(args)
    elif args.command == "batch":
        handle_batch(args)
    elif args.command == "cache-warm":
        handle_cache_warm(args)
    elif args.command == "snip":
        handle_snip(args)
    else:  # pragma: no cover - safeguard
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

"""Simple benchmarking harness for the OCR service."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from app.service import OCRService


def benchmark(paths, workers: int) -> None:
    service = OCRService()
    service.config.threads = workers
    durations = []
    start_all = time.perf_counter()
    for response in service.process_batch(paths):
        durations.append(response.meta.get("duration_ms", 0.0))
    total = time.perf_counter() - start_all
    if durations:
        print(f"Workers: {workers}")
        print(f"Count: {len(durations)}")
        print(f"p50: {statistics.median(durations):.2f} ms")
        print(f"p95: {statistics.quantiles(durations, n=20)[18]:.2f} ms")
        print(f"Throughput: {len(durations) / total:.2f} img/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR performance")
    parser.add_argument("path", type=Path, help="Folder with images to benchmark")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    paths = [p for p in args.path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".webp"}]
    if not paths:
        raise SystemExit("No images found for benchmarking")
    benchmark(paths, args.workers)


if __name__ == "__main__":
    main()

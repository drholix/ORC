"""Compute CER/WER for OCR predictions against ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cer(reference: str, hypothesis: str) -> float:
    return _levenshtein(reference, hypothesis) / max(1, len(reference))


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return _levenshtein(ref_words, hyp_words) / max(1, len(ref_words))


def _levenshtein(ref, hyp) -> int:
    if isinstance(ref, str):
        ref = list(ref)
    if isinstance(hyp, str):
        hyp = list(hyp)
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CER/WER for OCR outputs")
    parser.add_argument("predictions", type=Path, help="JSON file from CLI output")
    parser.add_argument("ground_truth", type=Path, help="JSON lines with `text` fields")
    args = parser.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    truths = [json.loads(line) for line in args.ground_truth.read_text(encoding="utf-8").splitlines()]
    if len(preds) != len(truths):
        raise SystemExit("Prediction and ground truth counts do not match")

    total_cer = 0.0
    total_wer = 0.0
    for pred, truth in zip(preds, truths):
        total_cer += cer(truth["text"], pred["text"])
        total_wer += wer(truth["text"], pred["text"])

    print(f"CER: {total_cer / len(preds):.4f}")
    print(f"WER: {total_wer / len(preds):.4f}")


if __name__ == "__main__":
    main()

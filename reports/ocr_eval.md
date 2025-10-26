# OCR Evaluation Report

Use this document to track recognition quality over time. Suggested workflow:

1. Collect paired datasets with ground truth transcriptions for Bahasa Indonesia and English (see `sample_data/`).
2. Run `python scripts/evaluate_accuracy.py` (to be implemented) to compute CER/WER.
3. Record dataset description, model version, preprocessing settings, and results here.

| Dataset | CER | WER | Avg Confidence | Notes |
| ------- | --- | --- | -------------- | ----- |
| Example | TBD | TBD | TBD | Baseline run with PaddleOCR Latin model |

Document improvement ideas, issues, and potential augmentations in this file to maintain transparency for new contributors.

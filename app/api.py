"""FastAPI application exposing the OCR service."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .config import OCRConfig, load_config
from .service import OCRResponse, OCRService

_PROMETHEUS_SPEC = importlib.util.find_spec("prometheus_client")
_PROMETHEUS_AVAILABLE = _PROMETHEUS_SPEC is not None

if _PROMETHEUS_AVAILABLE:
    from prometheus_client import (  # type: ignore=unused-ignore
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )
else:

    class _NoopTimer:
        def __enter__(self) -> "_NoopTimer":  # pragma: no cover - trivial
            return self

        def __exit__(
            self,
            _exc_type: object,
            _exc: object,
            _tb: object,
        ) -> None:  # pragma: no cover - trivial
            return None

    @dataclass
    class _NoopMetric:
        """Simple stand-in that exposes the Counter/Histogram API surface."""

        def labels(self, *args: object, **kwargs: object) -> "_NoopMetric":
            return self

        def inc(self, *args: object, **kwargs: object) -> None:
            return None

        def observe(self, *args: object, **kwargs: object) -> None:
            return None

        def time(self) -> _NoopTimer:
            return _NoopTimer()

    class Counter(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    class Histogram(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    def generate_latest(*_args: object, **_kwargs: object) -> bytes:  # pragma: no cover
        return b""


REQUEST_COUNTER = Counter("ocr_requests_total", "Total OCR requests", ["endpoint"])
LATENCY_HISTOGRAM = Histogram("ocr_request_latency_seconds", "OCR latency", ["endpoint"])


def create_app(config: Optional[OCRConfig] = None) -> FastAPI:
    config = config or load_config()
    service = OCRService(config)
    app = FastAPI(title="OCR Service", version="1.0")

    @app.post("/ocr")
    async def ocr_endpoint(
        file: Optional[UploadFile] = File(default=None),
        url: Optional[str] = None,
        pdf_text: bool = True,
    ) -> Dict[str, Any]:
        REQUEST_COUNTER.labels(endpoint="ocr").inc()
        with LATENCY_HISTOGRAM.labels(endpoint="ocr").time():
            if file is None and not url:
                raise HTTPException(status_code=400, detail="Provide file or url")

            if url:
                service.config.pdf_text = pdf_text
                response = service.process_image(url)
                return _response_to_dict(response)

            data = await file.read()
            service.config.pdf_text = pdf_text
            response = service.process_image(data)
            return _response_to_dict(response)

    @app.get("/healthz")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        if not _PROMETHEUS_AVAILABLE:
            return JSONResponse(
                {"detail": "prometheus_client not installed; metrics disabled."},
                status_code=503,
            )

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _response_to_dict(response: OCRResponse) -> Dict[str, Any]:
    return {
        "text": response.text,
        "language": response.language,
        "blocks": response.blocks,
        "meta": response.meta,
    }

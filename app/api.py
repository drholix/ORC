"""FastAPI application exposing the OCR service."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .config import OCRConfig, load_config
from .service import OCRResponse, OCRService

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
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def _response_to_dict(response: OCRResponse) -> Dict[str, Any]:
    return {
        "text": response.text,
        "language": response.language,
        "blocks": response.blocks,
        "meta": response.meta,
    }

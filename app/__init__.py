"""OCR toolkit package."""
from .config import OCRConfig, load_config
from .service import OCRResponse, OCRService

__all__ = ["OCRConfig", "OCRService", "OCRResponse", "load_config"]

"""Minimal :mod:`structlog` compatible shim backed by :mod:`logging`."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

_BASIC_CONFIGURED = False


def _configure_logging() -> None:
    global _BASIC_CONFIGURED
    if _BASIC_CONFIGURED:
        return
    level_name = os.environ.get("OCR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _BASIC_CONFIGURED = True


class _Logger:
    def __init__(self, name: str | None = None, **context: Any) -> None:
        _configure_logging()
        self._logger = logging.getLogger(name or "ocr")
        self._context: Dict[str, Any] = dict(context)

    def bind(self, **context: Any) -> "_Logger":
        new_context = self._context.copy()
        new_context.update(context)
        return _Logger(self._logger.name, **new_context)

    # ``structlog`` passes an "event" positional argument. We map it to
    # the logging message while appending the bound context for observability.
    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        payload = self._context.copy()
        payload.update(kwargs)
        extras = " ".join(f"{key}={value}" for key, value in payload.items())
        message = event if event else ""
        if extras:
            message = f"{message} {extras}".strip()
        self._logger.log(level, message)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)


def get_logger(name: str | None = None) -> _Logger:
    return _Logger(name=name)

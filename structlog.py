"""Minimal structlog shim for environments without the real dependency."""
from __future__ import annotations

from typing import Any


class _Logger:
    def __init__(self, **context: Any) -> None:
        self._context = context

    def bind(self, **context: Any) -> "_Logger":
        new_context = self._context.copy()
        new_context.update(context)
        return _Logger(**new_context)

    def info(self, msg: str, **kwargs: Any) -> None:
        pass

    def debug(self, msg: str, **kwargs: Any) -> None:
        pass

    def warning(self, msg: str, **kwargs: Any) -> None:
        pass

    def error(self, msg: str, **kwargs: Any) -> None:
        pass


def get_logger(name: str | None = None) -> _Logger:
    return _Logger(name=name)

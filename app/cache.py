"""Simple SQLite based cache for OCR results."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

LOGGER = structlog.get_logger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    result TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


@dataclass
class CacheEntry:
    key: str
    result: Dict[str, Any]


class OCRCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_db()

    def _ensure_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.commit()

    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor

    def compute_key(self, data: bytes) -> str:
        return sha256(data).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cursor = self._execute("SELECT result FROM cache WHERE key=?", (key,))
        row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, key: str, result: Dict[str, Any]) -> None:
        payload = json.dumps(result)
        self._execute("REPLACE INTO cache (key, result) VALUES (?, ?)", (key, payload))

    def delete(self, key: str) -> None:
        """Remove a cache entry if it exists."""

        self._execute("DELETE FROM cache WHERE key=?", (key,))

    def get_or_set(self, data: bytes, factory) -> Dict[str, Any]:
        key = self.compute_key(data)
        cached = self.get(key)
        if cached is not None:
            LOGGER.debug("cache_hit", key=key)
            return cached
        LOGGER.debug("cache_miss", key=key)
        result = factory()
        self.set(key, result)
        return result

"""Minimal Ollama chat client with schema-constrained output and an on-disk response cache.

Constrained decoding: Ollama compiles the JSON schema passed as ``format`` into a
grammar, so the model can only emit tokens that keep the output valid. With an
``enum`` of section ids this guarantees every predicted id exists in the document.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        cache_path: Path | None = None,
        timeout: float = 300.0,
        seed: int = 13,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.seed = seed
        self._http = httpx.Client(timeout=timeout)
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def check(self) -> None:
        """Fail early with a clear message if Ollama is not running or the model is missing."""
        try:
            tags = self._http.get(f"{self.host}/api/tags").json()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama is not reachable at {self.host}. Start it with `ollama serve`.") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaError(f"Model {self.model!r} not found in Ollama. Run `ollama pull {self.model}`.")

    def chat_json(self, system: str, user: str, schema: dict[str, Any], num_predict: int = 128) -> dict[str, Any]:
        """Return the parsed JSON object produced under ``schema`` (deterministic, cached)."""
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "options": {"temperature": 0, "seed": self.seed, "num_predict": num_predict},
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cached = self._get(key)
        if cached is not None:
            return cached
        resp = self._http.post(f"{self.host}/api/chat", json=payload)
        if resp.status_code != 200:
            raise OllamaError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        content = resp.json()["message"]["content"]
        try:
            out = json.loads(content)
        except json.JSONDecodeError as e:  # truncated by num_predict
            raise OllamaError(f"Model returned invalid JSON: {content[:200]!r}") from e
        self._put(key, out)
        return out

    def _get(self, key: str) -> dict | None:
        if not self._db:
            return None
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, value: dict) -> None:
        if not self._db:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()

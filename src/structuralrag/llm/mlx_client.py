"""Run a local Hugging Face model (optionally with a LoRA adapter) through Apple MLX.

Used for the fine-tuned generative retriever. It has the same ``chat_json`` interface as
OllamaClient, so GenerativeRetriever works with either. Difference: MLX has no grammar-
constrained decoding, so the output is parsed as JSON, falling back to the section ids
mentioned in the text; ids are then checked against the report by the caller.
Greedy decoding (temperature 0) and the same SQLite response cache keep runs deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from structuralrag.llm.ollama_client import OllamaClient

_JSON = re.compile(r"\{.*\}", re.DOTALL)
# A dotted section id ("2", "2.1.3") that is not a list marker ("1. ") or part of a longer number.
_SECTION_ID = re.compile(r"(?<![\d.])(\d+(?:\.\d+)*)(?![\d]|\.\d|\.\s|\.$)")


def ids_in_text(text: str) -> list[str]:
    """Section ids mentioned in free text, in order. Fallback when a model ignores the JSON format."""
    return list(dict.fromkeys(_SECTION_ID.findall(text)))


def parse_json(text: str) -> dict[str, Any]:
    """First JSON object in the model output, or {} if there is none."""
    m = _JSON.search(text)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return out if isinstance(out, dict) else {}


class MLXClient(OllamaClient):
    def __init__(
        self, model_path: str, adapter_path: str | None = None, cache_path: Path | None = None, label: str = ""
    ):
        super().__init__(model=label or Path(model_path).name, host="mlx://local", cache_path=cache_path)
        self.model_path, self.adapter_path = model_path, adapter_path
        self._loaded = None

    def check(self) -> None:
        self._load()

    def _load(self):
        if self._loaded is None:
            from mlx_lm import load

            self._loaded = load(self.model_path, adapter_path=self.adapter_path)
        return self._loaded

    def prompt(self, system: str, user: str) -> str:
        _, tok = self._load()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def chat_json(self, system: str, user: str, schema: dict[str, Any], num_predict: int = 128) -> dict[str, Any]:
        key = hashlib.sha256(
            json.dumps([self.model_path, self.adapter_path, system, user, num_predict]).encode()
        ).hexdigest()
        cached = self._get(key)
        if cached is not None:
            return cached
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tok = self._load()
        text = generate(model, tok, self.prompt(system, user), max_tokens=num_predict, sampler=make_sampler(temp=0.0))
        out = parse_json(text)
        if not out and "sections" in schema.get("properties", {}):
            # Without grammar-constrained decoding a zero-shot model often answers in prose; read the
            # ids it mentions (the retriever still drops ids that are not in the report).
            out = {"sections": ids_in_text(text), "format_fallback": True}
        self._put(key, out)
        return out

"""JSONL persistence for documents, queries and results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from structuralrag.schema import Document, Query

T = TypeVar("T", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    with path.open() as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


def load_documents(path: Path) -> list[Document]:
    return read_jsonl(path, Document)


def load_queries(path: Path) -> list[Query]:
    return read_jsonl(path, Query)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))

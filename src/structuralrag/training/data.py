"""Training question sets: ICT only, or ICT mixed with LLM-written paraphrases."""

from __future__ import annotations

import random

from structuralrag.schema import Query
from structuralrag.utils.config import DATA_DIR
from structuralrag.utils.io import load_queries

MODES = ("ict", "mixed")


def training_queries(mode: str, seed: int, total: int | None = None, llm_repeat: int = 3) -> list[Query]:
    """``ict``: train-split ICT questions. ``mixed``: every LLM-written train question repeated
    ``llm_repeat`` times (so ~1k paraphrases are not swamped by ~11k ICT questions) plus ICT
    questions, ``total`` in all when given."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = random.Random(seed)
    ict = load_queries(DATA_DIR / "train_queries.jsonl")
    if mode == "ict":
        return rng.sample(ict, min(total, len(ict))) if total else ict
    llm = load_queries(DATA_DIR / "train_queries_llm.jsonl") * llm_repeat
    n_ict = len(ict) if total is None else max(total - len(llm), 0)
    out = llm + rng.sample(ict, min(n_ict, len(ict)))
    rng.shuffle(out)
    return out

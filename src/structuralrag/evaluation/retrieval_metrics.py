"""Per-query retrieval metrics. All functions take a ranked list of ids and a gold set."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Hit@k: 1 if any gold item is in the top k (single-target queries: equals Recall@k)."""
    return float(any(r in gold for r in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int | None = None) -> float:
    for i, r in enumerate(ranked[:k] if k else ranked, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def per_query_metrics(ranked: Sequence[str], gold: set[str], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    m = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in ks}
    m["mrr@10"] = reciprocal_rank(ranked, gold, 10)
    m["ndcg@10"] = ndcg_at_k(ranked, gold, 10)
    return m

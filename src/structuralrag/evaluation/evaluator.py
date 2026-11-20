"""Run retrievers over a query set and aggregate metrics, CIs and latency."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from structuralrag.evaluation.retrieval_metrics import per_query_metrics
from structuralrag.evaluation.statistical_tests import bootstrap_ci
from structuralrag.evaluation.structural_metrics import structural_metrics
from structuralrag.retrieval.base import Retriever
from structuralrag.schema import Query

TOP_K = 10


@dataclass
class RunResult:
    method: str
    scope: str
    per_query: list[dict] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)

    def metric_array(self, name: str) -> np.ndarray:
        return np.array([q[name] for q in self.per_query], dtype=float)

    def summary(self, metrics: tuple[str, ...] = ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")) -> dict:
        out: dict = {"method": self.method, "scope": self.scope, "n_queries": len(self.per_query)}
        for m in metrics:
            vals = self.metric_array(m)
            lo, hi = bootstrap_ci(vals)
            out[m] = float(vals.mean())
            out[f"{m}_ci"] = [lo, hi]
        for m in ("correct_document", "same_branch"):
            out[m] = float(self.metric_array(m).mean())
        dist = self.metric_array("tree_distance")
        out["mean_tree_distance"] = float(np.nanmean(dist)) if np.any(~np.isnan(dist)) else None
        out.update({f"latency_{k}": v for k, v in self.latency_ms.items()})
        return out


def evaluate(
    retriever: Retriever,
    queries: list[Query],
    scope: str = "document",
    batch_size: int = 64,
    latency_sample: int = 50,
) -> RunResult:
    """scope='document': search only within the query's document (STAIR-style section routing).
    scope='corpus': search all sections of all documents."""
    if scope not in {"document", "corpus"}:
        raise ValueError(f"Unknown scope {scope!r}")
    corpus = retriever.corpus
    res = RunResult(method=retriever.name, scope=scope)
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        scores = retriever.score_batch([q.question for q in batch])
        for q, sc in zip(batch, scores, strict=True):
            ranked_idx = retriever.rank(sc, TOP_K, q.document_id if scope == "document" else None)
            ranked = [corpus.sections[i].uid for i, _ in ranked_idx]
            gold = q.target_uids
            row = {"query_id": q.query_id, "document_id": q.document_id, "top1": ranked[0]}
            row.update(per_query_metrics(ranked, gold))
            row.update(structural_metrics(ranked[0], gold, corpus))
            row["target_level"] = q.metadata.get("target_level")
            res.per_query.append(row)

    # Latency: end-to-end single-query search (encode + score + rank), as a user would experience.
    sample = queries[:latency_sample]
    if sample:
        retriever.search(sample[0].question)  # warm-up
        times = []
        for q in sample:
            t0 = time.perf_counter()
            retriever.search(q.question, top_k=5, document_id=q.document_id if scope == "document" else None)
            times.append((time.perf_counter() - t0) * 1000)
        res.latency_ms = {
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
        }
    return res

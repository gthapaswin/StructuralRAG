"""Tune the structure-aware weights on dev queries."""

from __future__ import annotations

import itertools

import numpy as np

from structuralrag.evaluation.retrieval_metrics import reciprocal_rank
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.structure_aware import StructureAwareRetriever, StructureParams


def tune_structure(suite: RetrieverSuite, queries, grid: dict) -> tuple[StructureParams, list[dict]]:
    """Grid search on dev queries (document scope, MRR@10). Components are computed once."""
    sa = suite.structure_aware()
    comp = sa.components([q.question for q in queries])
    doc_idx = [suite.corpus.doc_indices(q.document_id) for q in queries]
    gold = [suite.corpus.uid_to_idx[next(iter(q.target_uids))] for q in queries]
    trials = []
    for alpha, beta, sw in itertools.product(grid["alpha"], grid["beta"], grid["sparse_weight"]):
        p = StructureParams(alpha=alpha, beta=beta, sparse_weight=sw)
        scores = StructureAwareRetriever.combine(comp, p)
        rr = []
        for s, idx, g in zip(scores, doc_idx, gold, strict=True):
            order = idx[np.argsort(-s[idx], kind="stable")]
            rr.append(reciprocal_rank(list(order), {g}, 10))
        trials.append({"alpha": alpha, "beta": beta, "sparse_weight": sw, "dev_mrr@10": float(np.mean(rr))})
    best = max(trials, key=lambda t: t["dev_mrr@10"])
    return StructureParams(alpha=best["alpha"], beta=best["beta"], sparse_weight=best["sparse_weight"]), trials

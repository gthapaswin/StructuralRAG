from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.schema import RetrievedSection


class Retriever(ABC):
    """Scores every section in the corpus; search() restricts and ranks."""

    name: str = "base"

    def __init__(self, corpus: SectionCorpus):
        self.corpus = corpus

    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """Return a score for every section (higher is better)."""

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries])

    def rank(self, scores: np.ndarray, top_k: int, document_id: str | None = None) -> list[tuple[int, float]]:
        idx = self.corpus.doc_indices(document_id) if document_id else np.arange(len(self.corpus))
        sub = scores[idx]
        keep = np.isfinite(sub)  # -inf marks "not predicted" (generative retrieval)
        idx, sub = idx[keep], sub[keep]
        k = min(top_k, len(idx))
        if k == 0:
            return []
        top = np.argpartition(-sub, k - 1)[:k]
        top = top[np.argsort(-sub[top], kind="stable")]
        return [(int(idx[t]), float(sub[t])) for t in top]

    def search(self, query: str, top_k: int = 5, document_id: str | None = None) -> list[RetrievedSection]:
        return self._to_results(self.rank(self.score(query), top_k, document_id))

    def _to_results(self, ranked: list[tuple[int, float]]) -> list[RetrievedSection]:
        out = []
        for i, sc in ranked:
            s = self.corpus.sections[i]
            out.append(
                RetrievedSection(
                    document_id=s.document_id,
                    section_id=s.section_id,
                    title=s.title,
                    hierarchy_path=s.path,
                    score=sc,
                    text=s.text,
                )
            )
        return out


def rrf(score_lists: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal Rank Fusion over full score vectors."""
    weights = weights or [1.0] * len(score_lists)
    fused = np.zeros_like(score_lists[0], dtype=np.float64)
    for w, s in zip(weights, score_lists, strict=True):
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        fused += w / (k + ranks)
    return fused

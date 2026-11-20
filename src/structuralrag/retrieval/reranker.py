"""Cross-encoder reranking of a base retriever's top sections within one report.

The cross-encoder reads the question and each candidate together (hierarchy path +
section text), which is slower than comparing embeddings but more accurate. Only the
base retriever's top-N sections are rescored; the rest are dropped (score -inf).

A cross-encoder reads at most 512 tokens, so a long section is cut off and evidence near
its end is never seen. With ``window_words`` set, each section is split into overlapping
windows (each keeping the hierarchy path) and scored by its best window ("MaxP").
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from structuralrag.retrieval.base import Retriever
from structuralrag.schema import Section
from structuralrag.utils.reproducibility import best_device


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str, max_length: int):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device=best_device(), max_length=max_length)


def passage(section: Section) -> str:
    """What the cross-encoder sees for a section: where it sits, then what it says."""
    return f"{section.path_str}\n{section.text}"


def windows(section: Section, size: int, stride: int) -> list[str]:
    """Overlapping word windows of the section text, each prefixed with the hierarchy path."""
    words = section.text.split()
    if len(words) <= size:
        return [passage(section)]
    starts = range(0, len(words) - size + stride, stride)
    return [f"{section.path_str}\n{' '.join(words[s : s + size])}" for s in starts]


class RerankRetriever(Retriever):
    requires_document = True

    def __init__(
        self,
        base: Retriever,
        model_name: str,
        top_n: int = 20,
        max_length: int = 384,
        label: str = "",
        window_words: int | None = None,
        window_stride: int | None = None,
    ):
        super().__init__(base.corpus)
        self.base, self.model_name, self.top_n, self.max_length = base, model_name, top_n, max_length
        self.window_words = window_words
        self.window_stride = window_stride or (window_words // 2 if window_words else None)
        self.name = f"{base.name}+rerank{f'[{label}]' if label else ''}{'[maxp]' if window_words else ''}"

    @property
    def model(self):
        return load_cross_encoder(self.model_name, self.max_length)

    def score_for_document(self, query: str, document_id: str) -> np.ndarray:
        cand = [i for i, _ in self.base.rank(self.base.score_batch([query])[0], self.top_n, document_id)]
        scores = np.full(len(self.corpus), -np.inf)
        if not cand:
            return scores
        texts, owner = [], []
        for i in cand:
            sec = self.corpus.sections[i]
            parts = windows(sec, self.window_words, self.window_stride) if self.window_words else [passage(sec)]
            texts += parts
            owner += [i] * len(parts)
        raw = np.asarray(
            self.model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False), dtype=float
        )
        for i, s in zip(owner, raw, strict=True):
            scores[i] = max(scores[i], s)  # a section scores as its best window
        return scores

    def score(self, query: str) -> np.ndarray:
        raise NotImplementedError("Reranking needs a document_id; use score_for_document().")

    def search(self, query: str, top_k: int = 5, document_id: str | None = None):
        if document_id is None:
            raise ValueError("Reranking requires a document_id")
        return self._to_results(self.rank(self.score_for_document(query, document_id), top_k, document_id))

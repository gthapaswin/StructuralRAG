"""Cross-encoder reranking of a base retriever's top sections within one report.

The cross-encoder reads the question and each candidate together (hierarchy path +
section text), which is slower than comparing embeddings but more accurate. Only the
base retriever's top-N sections are rescored; the rest are dropped (score -inf).
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


class RerankRetriever(Retriever):
    requires_document = True

    def __init__(self, base: Retriever, model_name: str, top_n: int = 20, max_length: int = 384, label: str = ""):
        super().__init__(base.corpus)
        self.base, self.model_name, self.top_n, self.max_length = base, model_name, top_n, max_length
        self.name = f"{base.name}+rerank{f'[{label}]' if label else ''}"

    @property
    def model(self):
        return load_cross_encoder(self.model_name, self.max_length)

    def score_for_document(self, query: str, document_id: str) -> np.ndarray:
        cand = [i for i, _ in self.base.rank(self.base.score_batch([query])[0], self.top_n, document_id)]
        scores = np.full(len(self.corpus), -np.inf)
        if cand:
            pairs = [(query, passage(self.corpus.sections[i])) for i in cand]
            scores[cand] = np.asarray(self.model.predict(pairs, batch_size=32, show_progress_bar=False), dtype=float)
        return scores

    def score(self, query: str) -> np.ndarray:
        raise NotImplementedError("Reranking needs a document_id; use score_for_document().")

    def search(self, query: str, top_k: int = 5, document_id: str | None = None):
        if document_id is None:
            raise ValueError("Reranking requires a document_id")
        return self._to_results(self.rank(self.score_for_document(query, document_id), top_k, document_id))

from __future__ import annotations

import numpy as np

from structuralrag.retrieval.base import Retriever, rrf
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.dense import DenseRetriever


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion of a lexical and a dense retriever."""

    def __init__(self, sparse: BM25Retriever, dense: DenseRetriever, k: int = 60, sparse_weight: float = 1.0):
        super().__init__(sparse.corpus)
        self.sparse, self.dense, self.k, self.sparse_weight = sparse, dense, k, sparse_weight
        self.name = f"hybrid[{sparse.view}]"

    def score(self, query: str) -> np.ndarray:
        return self.score_batch([query])[0]

    def score_batch(self, queries: list[str]) -> np.ndarray:
        dense = self.dense.score_batch(queries)
        return np.stack(
            [
                rrf([self.sparse.score(q), d], k=self.k, weights=[self.sparse_weight, 1.0])
                for q, d in zip(queries, dense, strict=True)
            ]
        )

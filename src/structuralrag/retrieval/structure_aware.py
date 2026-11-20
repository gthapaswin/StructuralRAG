"""Structure-aware retrieval.

Dense score is a multi-field combination over the hierarchy:

    s(q, n) = cos(q, content_n) + alpha * cos(q, path_n) + beta * cos(q, content_parent(n))

* ``path_n`` embeds only the titles from the root to n ("Background > Oversight > FDA")
  so a query can match *where* a section lives, not just what it says.
* the parent term propagates evidence down the tree: sections whose parent is
  relevant get a boost, which disambiguates siblings with overlapping vocabulary.

The dense structural score is then fused (RRF) with BM25 over the structurally
enriched text view. alpha / beta / sparse weight are tuned on a dev split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structuralrag.retrieval.base import Retriever, rrf
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import DenseRetriever, Encoder


@dataclass
class StructureParams:
    alpha: float = 0.3  # weight of hierarchy-path similarity
    beta: float = 0.3  # weight of parent-content similarity
    sparse_weight: float = 1.0  # RRF weight of the BM25 component (0 disables it)
    rrf_k: int = 60


class StructureAwareRetriever(Retriever):
    def __init__(
        self,
        corpus: SectionCorpus,
        encoder: Encoder,
        params: StructureParams | None = None,
        content_dense: DenseRetriever | None = None,
        sparse: BM25Retriever | None = None,
    ):
        super().__init__(corpus)
        self.params = params or StructureParams()
        self.name = "structure_aware"
        self.encoder = encoder
        self.content = content_dense or DenseRetriever(corpus, encoder, View.TEXT)
        self.sparse = sparse or BM25Retriever(corpus, View.FULL)
        self.path_emb = encoder.encode_passages(corpus.path_texts())
        # For root sections, fall back to their own content score (neutral propagation).
        self._parent_or_self = np.where(corpus.parent_idx >= 0, corpus.parent_idx, np.arange(len(corpus)))

    def components(self, queries: list[str]) -> dict[str, np.ndarray]:
        q = self.encoder.encode_queries(queries)
        content = (q @ self.content.emb.T).astype(np.float64)
        return {
            "content": content,
            "path": (q @ self.path_emb.T).astype(np.float64),
            "parent": content[:, self._parent_or_self],
            "sparse": np.stack([self.sparse.score(x) for x in queries]),
        }

    @staticmethod
    def combine(comp: dict[str, np.ndarray], p: StructureParams) -> np.ndarray:
        dense = comp["content"] + p.alpha * comp["path"] + p.beta * comp["parent"]
        if p.sparse_weight <= 0:
            return dense
        return np.stack(
            [rrf([s, d], k=p.rrf_k, weights=[p.sparse_weight, 1.0]) for s, d in zip(comp["sparse"], dense, strict=True)]
        )

    def score(self, query: str) -> np.ndarray:
        return self.score_batch([query])[0]

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return self.combine(self.components(queries), self.params)

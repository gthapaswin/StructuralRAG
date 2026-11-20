"""Build every retriever over a corpus, sharing indexes and embeddings."""

from __future__ import annotations

from pathlib import Path

from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import DEFAULT_MODEL, DenseRetriever, Encoder
from structuralrag.retrieval.hybrid import HybridRetriever
from structuralrag.retrieval.structure_aware import StructureAwareRetriever, StructureParams

METHODS = ("bm25", "dense", "hybrid", "structure_aware")


class RetrieverSuite:
    """Lazily constructs and caches retrievers so components are built once."""

    def __init__(
        self,
        corpus: SectionCorpus,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        structure_params: StructureParams | None = None,
    ):
        self.corpus = corpus
        self.encoder = Encoder(model_name, cache_dir=cache_dir)
        self.structure_params = structure_params or StructureParams()
        self._bm25: dict[View, BM25Retriever] = {}
        self._dense: dict[View, DenseRetriever] = {}
        self._structure: StructureAwareRetriever | None = None

    def bm25(self, view: View = View.TEXT) -> BM25Retriever:
        if view not in self._bm25:
            self._bm25[view] = BM25Retriever(self.corpus, view)
        return self._bm25[view]

    def dense(self, view: View = View.TEXT) -> DenseRetriever:
        if view not in self._dense:
            self._dense[view] = DenseRetriever(self.corpus, self.encoder, view)
        return self._dense[view]

    def hybrid(self, view: View = View.TEXT) -> HybridRetriever:
        return HybridRetriever(self.bm25(view), self.dense(view))

    def structure_aware(self) -> StructureAwareRetriever:
        if self._structure is None:
            self._structure = StructureAwareRetriever(
                self.corpus,
                self.encoder,
                self.structure_params,
                content_dense=self.dense(View.TEXT),
                sparse=self.bm25(View.FULL),
            )
        return self._structure

    def get(self, method: str) -> Retriever:
        if method == "bm25":
            return self.bm25()
        if method == "dense":
            return self.dense()
        if method == "hybrid":
            return self.hybrid()
        if method == "structure_aware":
            return self.structure_aware()
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from structuralrag.preprocessing.text_cleaner import tokenize
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus, View


class BM25Retriever(Retriever):
    def __init__(self, corpus: SectionCorpus, view: View = View.TEXT, k1: float = 1.5, b: float = 0.75):
        super().__init__(corpus)
        self.view = view
        self.name = f"bm25[{view}]"
        tokenized = [tokenize(t) or ["<empty>"] for t in corpus.views(view)]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)

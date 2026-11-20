from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.utils.reproducibility import best_device

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


@lru_cache(maxsize=2)
def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=best_device())


class Encoder:
    """Sentence-transformer wrapper with an on-disk embedding cache."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    @property
    def model(self):
        return load_encoder(self.model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        key = hashlib.sha1((self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb_{key}.npy" if self.cache_dir else None
        if path and path.exists():
            return np.load(path)
        log.info("Encoding %d passages with %s", len(texts), self.model_name)
        emb = self.model.encode(
            [t or " " for t in texts], batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefix = QUERY_PREFIX.get(self.model_name, "")
        return self.model.encode(
            [prefix + q for q in queries], batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)


class DenseRetriever(Retriever):
    """Cosine similarity between normalized query and section embeddings."""

    def __init__(self, corpus: SectionCorpus, encoder: Encoder, view: View = View.TEXT):
        super().__init__(corpus)
        self.encoder = encoder
        self.view = view
        self.name = f"dense[{view}]"
        self.emb = encoder.encode_passages(corpus.views(view))

    def score(self, query: str) -> np.ndarray:
        return self.score_batch([query])[0]

    def score_batch(self, queries: list[str]) -> np.ndarray:
        q = self.encoder.encode_queries(queries)
        return (q @ self.emb.T).astype(np.float64)

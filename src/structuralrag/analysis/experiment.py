"""Score components for the heading study, computed once per corpus variant.

Everything is computed within the query's document (the task is routing a question to a
section of a known document), including the RRF ranks.

flat            = RRF(BM25 over section text, dense over section text): no headings anywhere.
structure-aware = RRF(BM25 over the full structural view, cos(content) + alpha*cos(path)
                  + beta*cos(parent)), the project's method. Only the heading-dependent parts
                  (path embedding, BM25 full view) change between heading variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from structuralrag.analysis.informativeness import heading_specificity
from structuralrag.preprocessing.text_cleaner import tokenize
from structuralrag.retrieval.base import rrf
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import Encoder
from structuralrag.schema import Query

TOP_K = 10


@dataclass
class Components:
    """Per-query scores over the sections of the query's own document only."""

    content: list[np.ndarray]  # cosine to section text
    path: list[np.ndarray]  # cosine to the hierarchy path of titles
    parent: list[np.ndarray]  # cosine to the parent's text (self for roots)
    sparse_full: list[np.ndarray]  # BM25 over the structural view (titles, path, parent intro, text)
    sparse_text: list[np.ndarray]  # BM25 over section text only
    gold: list[np.ndarray]  # boolean mask of gold sections within the document
    specificity: list[np.ndarray]  # heading specificity of each section (see informativeness.py)


def _bm25(retriever: BM25Retriever, text: str, idx: np.ndarray) -> np.ndarray:
    """BM25 with corpus-level statistics, scored only for the document's sections."""
    return np.asarray(retriever.bm25.get_batch_scores(tokenize(text), idx.tolist()), dtype=np.float64)


def compute(
    corpus: SectionCorpus,
    encoder: Encoder,
    queries: list[Query],
    q_emb: np.ndarray | None = None,
    sparse_text: list[np.ndarray] | None = None,
) -> Components:
    """q_emb and sparse_text do not depend on headings and can be reused across variants
    (section order and ids are the same in every heading variant)."""
    texts = [q.question for q in queries]
    q_all = encoder.encode_queries(texts) if q_emb is None else q_emb
    content_emb = encoder.encode_passages(corpus.views(View.TEXT))
    path_emb = encoder.encode_passages(corpus.path_texts())
    parent_or_self = np.where(corpus.parent_idx >= 0, corpus.parent_idx, np.arange(len(corpus)))
    full = BM25Retriever(corpus, View.FULL)
    text_bm25 = BM25Retriever(corpus, View.TEXT) if sparse_text is None else None
    spec = heading_specificity(corpus)
    comp = Components([], [], [], [], [], [], [])
    for i, q in enumerate(queries):
        idx = corpus.doc_indices(q.document_id)
        qe = q_all[i]
        comp.content.append((content_emb[idx] @ qe).astype(np.float64))
        comp.path.append((path_emb[idx] @ qe).astype(np.float64))
        comp.parent.append((content_emb[parent_or_self[idx]] @ qe).astype(np.float64))
        comp.sparse_full.append(_bm25(full, texts[i], idx))
        comp.sparse_text.append(sparse_text[i] if sparse_text is not None else _bm25(text_bm25, texts[i], idx))
        gold = {corpus.uid_to_idx[u] for u in q.target_uids}
        comp.gold.append(np.array([j in gold for j in idx]))
        comp.specificity.append(spec[idx])
    return comp


def text_bm25_scores(corpus: SectionCorpus, queries: list[Query]) -> list[np.ndarray]:
    bt = BM25Retriever(corpus, View.TEXT)
    return [_bm25(bt, q.question, corpus.doc_indices(q.document_id)) for q in queries]


def _rr(s: np.ndarray, gold: np.ndarray, k: int) -> float:
    order = np.argsort(-s, kind="stable")[:k]
    hits = np.flatnonzero(gold[order])
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def reciprocal_ranks(scores: list[np.ndarray], comp: Components, k: int = TOP_K) -> np.ndarray:
    """MRR@k contribution per query (scores are over the query's document)."""
    return np.array([_rr(s, g, k) for s, g in zip(scores, comp.gold, strict=True)])


def flat_scores(comp: Components, rrf_k: int = 60) -> list[np.ndarray]:
    return [rrf([s, d], k=rrf_k) for s, d in zip(comp.sparse_text, comp.content, strict=True)]


def structure_scores(
    comp: Components,
    alpha: float,
    beta: float = 0.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
    fallback: bool = False,
) -> list[np.ndarray]:
    """fallback=True: the path term of a section with a generic heading falls back to its content
    score, path' = w * cos(q, path) + (1 - w) * cos(q, content), w = heading specificity."""
    paths = comp.path
    if fallback:
        paths = [w * p + (1 - w) * c for w, p, c in zip(comp.specificity, comp.path, comp.content, strict=True)]
    dense = [c + alpha * p + beta * pa for c, p, pa in zip(comp.content, paths, comp.parent, strict=True)]
    if sparse_weight <= 0:
        return dense
    return [rrf([s, d], k=rrf_k, weights=[sparse_weight, 1.0]) for s, d in zip(comp.sparse_full, dense, strict=True)]


def structure_rr_per_alpha(
    comp: Components, alphas, beta: float = 0.0, sparse_weight: float = 1.0, fallback: bool = False
) -> np.ndarray:
    """(n_alphas, n_queries) reciprocal ranks, for oracle and gating analyses."""
    return np.stack(
        [reciprocal_ranks(structure_scores(comp, a, beta, sparse_weight, fallback=fallback), comp) for a in alphas]
    )

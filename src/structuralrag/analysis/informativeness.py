"""Heading informativeness: how well a document's headings identify their own sections.

All features use only the document (no questions), so they can be computed before
retrieval and used to predict, or set, how much weight headings should get.

self_mrr      Treat each heading path as a query and rank the document's sections by
              cosine to their text: does the heading point to its own section? (main score)
self_nrank    Same, as a size-normalized rank in [0, 1] (comparable across document sizes).
title_mrr     Same with the section's own title only (no ancestors).
sibling_sim   Mean cosine between titles of sibling sections (high = hard to tell apart).
generic_rate  Share of titles that are common across documents ("Background", "Results").
title_idf     Mean IDF of title words over all titles in the pool (rare words = specific).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from structuralrag.preprocessing.text_cleaner import tokenize
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import Encoder
from structuralrag.schema import Document


def _norm(t: str) -> str:
    return " ".join(tokenize(t))


def title_stats(pool: list[Document]) -> tuple[Counter, dict[str, float], int]:
    """Document frequency of normalized titles and IDF of title words, over a pool of documents."""
    title_df: Counter = Counter()
    word_df: Counter = Counter()
    for d in pool:
        title_df.update({_norm(s.title) for s in d.sections})
        word_df.update({w for s in d.sections for w in tokenize(s.title)})
    n = len(pool)
    idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in word_df.items()}
    return title_df, idf, n


def heading_specificity(corpus: SectionCorpus) -> np.ndarray:
    """Per-section heading specificity in [0, 1] from the collection itself (no labels).

    w = log(N / df) / log(N), where df is the number of the collection's N documents that
    contain the same (normalized) heading. A heading unique to its document gets 1; one that
    every document has ("Introduction") gets 0.
    """
    df: Counter = Counter()
    for d in corpus.docs.values():
        df.update({_norm(s.title) for s in d.sections})
    n = max(len(corpus.docs), 2)
    return np.array([np.log(n / max(df[_norm(s.title)], 1)) / np.log(n) for s in corpus.sections])


def _self_rank(q: np.ndarray, p: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank of section i when its query vector q[i] ranks the document's sections p.

    Returns the reciprocal rank and a size-normalized rank score in [0, 1]
    (1 = ranked first, 0 = ranked last), which does not shrink with the number of sections.
    """
    sims = q @ p.T
    rr = np.full(len(q), np.nan)
    nr = np.full(len(q), np.nan)
    n = int(keep.sum())
    for i in np.flatnonzero(keep):
        others = keep.copy()
        others[i] = False
        rank = 1 + int(np.sum(sims[i, others] >= sims[i, i]))  # ties count against the heading
        rr[i] = 1.0 / rank
        nr[i] = 1.0 - (rank - 1) / (n - 1)
    return rr, nr


def section_features(corpus: SectionCorpus, encoder: Encoder) -> pd.DataFrame:
    """Per-section heading features (self rank of path and title)."""
    content = encoder.encode_passages(corpus.views(View.TEXT))
    path_q = encoder.encode_queries(corpus.path_texts())
    title_q = encoder.encode_queries([s.title for s in corpus.sections])
    rows = []
    for doc_id in corpus.docs:
        idx = corpus.doc_indices(doc_id)
        keep = np.array([len(corpus.sections[i].text.split()) >= 5 for i in idx])
        if keep.sum() < 2:
            continue
        p_rr, p_nr = _self_rank(path_q[idx], content[idx], keep)
        t_rr, t_nr = _self_rank(title_q[idx], content[idx], keep)
        for j, i in enumerate(idx):
            rows.append(
                {
                    "uid": corpus.sections[i].uid,
                    "document_id": doc_id,
                    "path_rr": p_rr[j],
                    "path_nrank": p_nr[j],
                    "title_rr": t_rr[j],
                    "title_nrank": t_nr[j],
                }
            )
    return pd.DataFrame(rows)


def document_features(
    corpus: SectionCorpus, encoder: Encoder, pool: list[Document], common_frac: float = 0.02
) -> pd.DataFrame:
    title_df, idf, n_pool = title_stats(pool)
    sec = section_features(corpus, encoder)
    title_emb = encoder.encode_passages([s.title for s in corpus.sections])
    rows = []
    for doc_id, d in corpus.docs.items():
        idx = corpus.doc_indices(doc_id)
        s = sec[sec.document_id == doc_id]
        groups = defaultdict(list)
        for i in idx:
            groups[corpus.sections[i].parent_section_id].append(i)
        sib = [
            float((title_emb[g] @ title_emb[g].T)[np.triu_indices(len(g), 1)].mean())
            for g in groups.values()
            if len(g) > 1
        ]
        titles = [x.title for x in d.sections]
        words = [w for t in titles for w in tokenize(t)]
        rows.append(
            {
                "document_id": doc_id,
                "n_sections": len(idx),
                "max_depth": max(x.level for x in d.sections),
                "self_mrr": float(s.path_rr.mean()) if len(s) else np.nan,
                "self_nrank": float(s.path_nrank.mean()) if len(s) else np.nan,
                "title_mrr": float(s.title_rr.mean()) if len(s) else np.nan,
                "sibling_sim": float(np.mean(sib)) if sib else np.nan,
                "generic_rate": float(np.mean([title_df[_norm(t)] / n_pool >= common_frac for t in titles])),
                "title_idf": float(np.mean([idf.get(w, 0.0) for w in words])) if words else 0.0,
            }
        )
    return pd.DataFrame(rows)

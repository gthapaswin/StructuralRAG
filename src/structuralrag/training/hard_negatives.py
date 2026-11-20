"""Hard negatives from the document tree.

A question's negatives are other sections *of the same report* that the base model
finds most similar. Sections that are ancestors or descendants of the gold section are
skipped: their text overlaps with the answer, so treating them as wrong would teach the
model to punish correct-but-broader matches (false negatives).
"""

from __future__ import annotations

import numpy as np

from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.schema import Query


def related(a: str, b: str) -> bool:
    """Same section, or one is an ancestor of the other (dotted ids)."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def mine_hard_negatives(corpus: SectionCorpus, queries: list[Query], scores: np.ndarray, n: int) -> list[list[int]]:
    """For each query, the indices of the n highest-scoring unrelated sections in its report.

    ``scores`` is (n_queries, n_sections) from the base retriever. Sections with no text are
    never used. Queries whose report has fewer than n candidates get fewer negatives.
    """
    out = []
    for q, s in zip(queries, scores, strict=True):
        gold = q.target_section_ids[0]
        idx = corpus.doc_indices(q.document_id)
        order = idx[np.argsort(-s[idx], kind="stable")]
        negs = [int(i) for i in order if corpus.sections[i].text and not related(corpus.sections[i].section_id, gold)][
            :n
        ]
        out.append(negs)
    return out

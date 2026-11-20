"""Inverse-Cloze-Task (ICT) query generation.

A sentence is sampled from a section and *removed* from that section's indexed
text; the sentence becomes the query and the section is the retrieval target.
This needs no LLM, has an unambiguous gold label, and avoids the trivial
exact-match leakage that would occur if the sentence stayed in the index.
"""

from __future__ import annotations

import random

from structuralrag.preprocessing.text_cleaner import split_sentences
from structuralrag.schema import Document, Query

MIN_SECTION_SENTENCES = 4
MIN_QUERY_WORDS = 8
MAX_QUERY_WORDS = 45


def eligible_sentences(text: str) -> list[int]:
    sents = split_sentences(text)
    if len(sents) < MIN_SECTION_SENTENCES:
        return []
    return [i for i, s in enumerate(sents) if MIN_QUERY_WORDS <= len(s.split()) <= MAX_QUERY_WORDS]


def generate_ict_queries(
    docs: list[Document],
    split: str,
    queries_per_doc: int,
    seed: int,
) -> tuple[list[Document], list[Query]]:
    """Return (documents with query sentences removed, queries)."""
    rng = random.Random(seed)
    out_docs: list[Document] = []
    queries: list[Query] = []
    for doc in docs:
        doc = doc.model_copy(deep=True)
        candidates = [s for s in doc.sections if eligible_sentences(s.text)]
        rng.shuffle(candidates)
        for sec in candidates[:queries_per_doc]:
            sents = split_sentences(sec.text)
            idx = rng.choice(eligible_sentences(sec.text))
            question = sents[idx]
            sec.text = " ".join(s for i, s in enumerate(sents) if i != idx)
            queries.append(
                Query(
                    query_id=f"{doc.document_id}::q{len(queries)}",
                    dataset=doc.dataset,
                    document_id=doc.document_id,
                    question=question,
                    target_section_ids=[sec.section_id],
                    split=split,
                    generator="ict",
                    metadata={"sentence_index": idx, "target_level": sec.level},
                )
            )
        out_docs.append(doc)
    return out_docs, queries

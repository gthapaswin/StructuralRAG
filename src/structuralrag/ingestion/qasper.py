"""Adapter for QASPER (allenai/qasper): NLP research papers with human-written questions.

A second domain for the project. Papers have nested sections ("Model ::: Response
Generation"), and each question was written by an NLP practitioner who saw only the title
and abstract; other annotators answered it from the full paper and marked the evidence
paragraphs. The gold sections of a question are the sections holding that evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.schema import Document, Query

log = logging.getLogger(__name__)

HF_DATASET = "allenai/qasper"
PARQUET_BASE = f"hf://datasets/{HF_DATASET}@refs%2Fconvert%2Fparquet/qasper"
SEP = " ::: "
FLOAT = "FLOAT SELECTED"  # evidence that is a table or figure, not section text


def row_to_document(row: dict) -> tuple[Document, dict[str, str]]:
    """Return the paper as a section tree, plus a map paragraph text -> section id."""
    headings = [FlatHeading(title="Abstract", depth=1, text=row["abstract"] or "")]
    para_owner: list[list[str]] = [[row["abstract"] or ""]]
    ft = row["full_text"]
    open_path: list[str] = []
    for name, paras in zip(ft["section_name"], ft["paragraphs"], strict=True):
        parts = [p.strip() for p in (name or "").split(SEP) if p.strip()] or ["Untitled"]
        paras = [p for p in paras if p and p.strip()]
        # "Model ::: Background" may appear without a "Model" section before it: add the missing
        # (empty) parent headings, or the child would attach to whatever chapter came before.
        for depth in range(1, len(parts)):
            if open_path[:depth] != parts[:depth]:
                headings.append(FlatHeading(title=parts[depth - 1], depth=depth, text=""))
                para_owner.append([])
                open_path = parts[:depth]
        headings.append(FlatHeading(title=parts[-1], depth=len(parts), text=" ".join(paras)))
        para_owner.append(paras)
        open_path = parts
    doc = build_document(
        dataset="qasper",
        document_id=f"qasper_{row['id']}",
        title=row["title"],
        headings=headings,
        metadata={"source": HF_DATASET},
    )
    if len(doc.sections) != len(headings):
        raise ValueError(f"{row['id']}: section count changed while building the tree")
    owner = {p: sec.section_id for sec, paras in zip(doc.sections, para_owner, strict=True) for p in paras}
    return doc, owner


def row_to_queries(row: dict, doc: Document, owner: dict[str, str], split: str) -> tuple[list[Query], dict]:
    """One query per answerable question whose evidence is section text.

    Gold = sections containing the evidence of the first annotator who answered with text
    evidence (questions have 1-3 annotators; using one keeps the gold set from growing with
    the number of annotators).
    """
    stats = {"questions": 0, "unanswerable": 0, "no_text_evidence": 0, "unmatched_evidence": 0}
    queries = []
    qas = row["qas"]
    for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"], strict=True):
        stats["questions"] += 1
        gold, reason = None, "unanswerable"
        for ans in answers["answer"]:
            if ans["unanswerable"]:
                continue
            ev = [e for e in ans["evidence"] if e and not e.startswith(FLOAT)]
            if not ev:
                reason = "no_text_evidence"
                continue
            ids = [owner.get(e) for e in ev]
            if any(i is None for i in ids):
                reason = "unmatched_evidence"
                continue
            gold = list(dict.fromkeys(ids))
            break
        if gold is None:
            stats[reason] += 1
            continue
        level = max(doc.section(g).level for g in gold)
        queries.append(
            Query(
                query_id=f"{doc.document_id}::{qid}",
                dataset="qasper",
                document_id=doc.document_id,
                question=question.strip(),
                target_section_ids=gold,
                split=split,
                generator="human",
                metadata={"target_level": level, "n_gold": len(gold)},
            )
        )
    return queries, stats


def load_qasper(split: str, limit: int | None = None, seed: int = 13) -> Iterator[tuple[Document, list[Query], dict]]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={split: f"{PARQUET_BASE}/{split}/*.parquet"}, split=split)
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    for row in ds:
        try:
            doc, owner = row_to_document(row)
        except ValueError as e:
            log.warning("Skipping %s: %s", row.get("id"), e)
            continue
        queries, stats = row_to_queries(row, doc, owner, split)
        yield doc, queries, stats

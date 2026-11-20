"""Adapter for GovReport (launch/gov_report, 'structure' config).

Each report comes with an ordered list of section titles, their nesting depth and
their paragraphs, so the hierarchy is taken from the dataset rather than guessed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.schema import Document

log = logging.getLogger(__name__)

HF_DATASET = "launch/gov_report"
HF_CONFIG = "structure"


def row_to_document(row: dict) -> Document:
    secs = row["document_sections"]
    titles, paragraphs, depths = secs["title"], secs["paragraphs"], secs["depth"]
    if not titles:
        raise ValueError(f"{row['id']} has no sections")
    base = min(depths)
    headings = [
        FlatHeading(title=t or "", depth=d - base + 1, text=(p or "").replace("\n", " "))
        for t, p, d in zip(titles, paragraphs, depths, strict=True)
    ]
    doc_id = row["id"]
    return build_document(
        dataset="govreport",
        document_id=doc_id,
        title=headings[0].title if headings else doc_id,
        headings=headings,
        metadata={"source": f"{HF_DATASET}/{HF_CONFIG}", "agency": doc_id.split("_")[0]},
    )


def load_govreport(split: str, limit: int | None = None, seed: int = 13) -> Iterator[Document]:
    """Yield GovReport documents for an HF split ('train' | 'validation' | 'test')."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, HF_CONFIG, split=split)
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    for row in ds:
        try:
            yield row_to_document(row)
        except ValueError as e:  # malformed report: skip, but say so
            log.warning("Skipping %s: %s", row.get("id"), e)

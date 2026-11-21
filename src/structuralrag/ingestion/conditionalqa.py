"""Adapter for ConditionalQA (Sun et al., ACL 2022): UK government guidance pages (gov.uk) with
questions written by crowd workers from a personal scenario, and evidence sentences marked in the
page. Held-out fourth domain for the heading study.

Pages are lists of HTML elements; <h1> starts a part ("Eligibility", "How to claim"), <h2> and
<h3> start nested sections. The gold sections of a question are the sections containing its
evidence elements. The query is the question alone (the scenario is not used).
"""

from __future__ import annotations

import html
import re

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.schema import Document, Query

HEADING = re.compile(r"^<h([1-4])>(.*)</h\1>$", re.S)
TAG = re.compile(r"<[^>]+>")


def _text(element: str) -> str:
    return " ".join(html.unescape(TAG.sub(" ", element)).split())


def page_to_document(page: dict, doc_id: str) -> tuple[Document, dict[str, str]]:
    """Section tree of a page, plus a map evidence element (raw html) -> section id."""
    heads: list[FlatHeading] = []
    owned: list[list[str]] = []
    for el in page["contents"]:
        m = HEADING.match(el.strip())
        if m:
            heads.append(FlatHeading(title=_text(m.group(2)) or "Untitled", depth=int(m.group(1)), text=""))
            owned.append([])
            continue
        if not heads:  # text before any heading
            heads.append(FlatHeading(title="Overview", depth=1, text=""))
            owned.append([])
        owned[-1].append(el)
    heads = [
        FlatHeading(title=h.title, depth=h.depth, text=" ".join(_text(e) for e in o))
        for h, o in zip(heads, owned, strict=True)
    ]
    doc = build_document(
        dataset="conditionalqa",
        document_id=doc_id,
        title=page["title"],
        headings=heads,
        metadata={"source": "ConditionalQA", "url": page["url"]},
    )
    if len(doc.sections) != len(heads):
        raise ValueError(f"{doc_id}: section count changed while building the tree")
    owner = {e.strip(): s.section_id for s, o in zip(doc.sections, owned, strict=True) for e in o}
    return doc, owner


def question_to_query(q: dict, doc: Document, owner: dict[str, str]) -> Query | None:
    if q["not_answerable"] or not q["evidences"]:
        return None
    ids = [owner.get(e.strip()) for e in q["evidences"]]
    if any(i is None for i in ids):
        return None
    gold = list(dict.fromkeys(ids))
    return Query(
        query_id=f"{doc.document_id}::{q['id']}",
        dataset="conditionalqa",
        document_id=doc.document_id,
        question=q["question"].strip(),
        target_section_ids=gold,
        split=q["id"].split("-")[0],
        generator="human",
        metadata={"target_level": max(doc.section(g).level for g in gold), "n_gold": len(gold)},
    )

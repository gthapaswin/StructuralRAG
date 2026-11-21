"""Controlled changes to a document's headings. Section text is never touched.

Every transform returns new Documents whose titles (and therefore hierarchy paths) differ,
so any retriever that reads headings sees the change while content-only retrievers do not.
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict

from structuralrag.schema import Document


def retitle(doc: Document, titles: dict[str, str]) -> Document:
    """Replace section titles (section_id -> title) and rebuild every hierarchy path."""
    new = {s.section_id: titles.get(s.section_id, s.title) for s in doc.sections}
    by_id = {s.section_id: s for s in doc.sections}
    sections = []
    for s in doc.sections:
        chain, cur = [], s
        while cur is not None:
            chain.append(new[cur.section_id])
            cur = by_id.get(cur.parent_section_id) if cur.parent_section_id else None
        sections.append(s.model_copy(update={"title": new[s.section_id], "path": chain[::-1]}))
    return doc.model_copy(update={"sections": sections})


# Sections that summarise the document; their headings are generic and they lose to body sections
# (see reports/heading_study/FINDINGS.md). Matched on the original heading.
SUMMARY_HEADING = re.compile(r"^(introduction|abstract|overview|summary|conclusions?|concluding|background)\b", re.I)


def is_summary_heading(title: str) -> bool:
    return bool(SUMMARY_HEADING.match(title.strip()))


def generic_title(section_id: str) -> str:
    return f"Section {section_id}"


def genericize(docs: list[Document], frac: float, seed: int = 0) -> list[Document]:
    """Replace a random fraction of headings with content-free labels ("Section 3.2")."""
    rng = random.Random(seed)
    out = []
    for d in docs:
        titles = {s.section_id: generic_title(s.section_id) for s in d.sections if rng.random() < frac}
        out.append(retitle(d, titles))
    return out


def placebo(docs: list[Document], seed: int = 0) -> list[Document]:
    """Give every section a real heading from a *different* document at the same depth.

    Keeps heading length, style and depth statistics; removes the link to the content.
    """
    rng = random.Random(seed)
    pool: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for d in docs:
        for s in d.sections:
            pool[s.level].append((d.document_id, s.title))
    out = []
    for d in docs:
        titles = {}
        for s in d.sections:
            cands = pool[s.level] if len(pool[s.level]) > 1 else pool[1]
            for _ in range(50):
                src, t = rng.choice(cands)
                if src != d.document_id:
                    break
            titles[s.section_id] = t
        out.append(retitle(d, titles))
    return out


def ocr_noise(docs: list[Document], rate: float, seed: int = 0) -> list[Document]:
    """Corrupt heading characters (substitute / delete / insert) with probability `rate` each."""
    rng = random.Random(seed)
    letters = string.ascii_lowercase

    def corrupt(t: str) -> str:
        out = []
        for ch in t:
            r = rng.random()
            if r < rate / 3:
                out.append(rng.choice(letters))
            elif r < 2 * rate / 3:
                continue
            elif r < rate:
                out += [ch, rng.choice(letters)]
            else:
                out.append(ch)
        return "".join(out) or t

    return [retitle(d, {s.section_id: corrupt(s.title) for s in d.sections}) for d in docs]


def flatten(docs: list[Document]) -> list[Document]:
    """Keep each section's own heading but drop its ancestors from the path (no hierarchy)."""
    out = []
    for d in docs:
        sections = [s.model_copy(update={"path": [s.title]}) for s in d.sections]
        out.append(d.model_copy(update={"sections": sections}))
    return out

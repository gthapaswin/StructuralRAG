"""Parse an arbitrary PDF into the unified schema using its embedded outline (ToC).

If the PDF has no outline, falls back to font-size heuristics: lines whose font is
noticeably larger than the body text are treated as headings.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.schema import Document


def _page_texts(pdf: fitz.Document) -> list[str]:
    return [page.get_text("text") for page in pdf]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _headings_from_outline(pdf: fitz.Document) -> list[FlatHeading]:
    toc = pdf.get_toc(simple=True)  # [[level, title, page], ...]
    if not toc:
        return []
    pages = _page_texts(pdf)
    headings: list[FlatHeading] = []
    for i, (level, title, page) in enumerate(toc):
        start = max(page - 1, 0)
        nxt_page = toc[i + 1][2] - 1 if i + 1 < len(toc) else len(pages) - 1
        end = max(start, nxt_page)
        chunk = "\n".join(pages[start : end + 1])
        # Trim to the region between this title and the next title where possible.
        low = chunk.lower()
        pos = low.find(_norm(title)[:40])
        if pos >= 0:
            chunk = chunk[pos + len(title) :]
        if i + 1 < len(toc):
            npos = chunk.lower().find(_norm(toc[i + 1][1])[:40])
            if npos > 0:
                chunk = chunk[:npos]
        headings.append(FlatHeading(title=title, depth=level, text=chunk, page_start=start + 1, page_end=end + 1))
    return headings


def _headings_from_fonts(pdf: fitz.Document, max_levels: int = 3) -> list[FlatHeading]:
    spans = []
    for pno, page in enumerate(pdf):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if text:
                    size = round(max(s["size"] for s in line["spans"]), 1)
                    spans.append((pno + 1, size, text))
    if not spans:
        return []
    body = Counter(s for _, s, _ in spans).most_common(1)[0][0]
    heading_sizes = sorted({s for _, s, t in spans if s >= body + 1.5 and len(t) < 120}, reverse=True)[:max_levels]
    level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}
    headings: list[FlatHeading] = []
    for page, size, text in spans:
        if size in level_of:
            headings.append(FlatHeading(title=text, depth=level_of[size], page_start=page, page_end=page))
        elif headings:
            headings[-1].text += " " + text
            headings[-1].page_end = page
    return headings


def parse_pdf(path: str | Path, document_id: str | None = None) -> Document:
    path = Path(path)
    with fitz.open(path) as pdf:
        headings = _headings_from_outline(pdf) or _headings_from_fonts(pdf)
        if not headings:
            text = "\n".join(_page_texts(pdf))
            headings = [FlatHeading(title=path.stem, depth=1, text=text, page_start=1, page_end=len(pdf))]
        title = (pdf.metadata or {}).get("title") or path.stem
    return build_document(
        dataset="pdf",
        document_id=document_id or path.stem,
        title=title,
        headings=headings,
        metadata={"source": str(path.name)},
    )

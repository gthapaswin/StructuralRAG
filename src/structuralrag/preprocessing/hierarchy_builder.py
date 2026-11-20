"""Turn a flat list of (title, depth, text) headings into a validated section tree."""

from __future__ import annotations

from dataclasses import dataclass

from structuralrag.preprocessing.text_cleaner import normalize
from structuralrag.schema import Document, Section


@dataclass
class FlatHeading:
    title: str
    depth: int  # 1 = top level
    text: str = ""
    page_start: int | None = None
    page_end: int | None = None


class HierarchyError(ValueError):
    pass


def build_document(
    dataset: str,
    document_id: str,
    title: str,
    headings: list[FlatHeading],
    metadata: dict | None = None,
) -> Document:
    """Build a Document from headings given in reading order.

    A heading's parent is the closest earlier heading with a smaller *raw* depth, so
    depth jumps (1 -> 3) still yield a valid child at level 2, and headings that share
    a raw depth are always siblings. Section ids are dotted counters: 1, 1.1, 2, 2.1.1 ...
    """
    sections: list[Section] = []
    stack: list[tuple[int, Section]] = []  # (raw depth, section) for the current ancestor chain
    counters: list[int] = []

    for h in headings:
        while stack and stack[-1][0] >= h.depth:
            stack.pop()
        depth = len(stack) + 1
        counters = counters[:depth]
        if len(counters) < depth:
            counters.append(0)
        counters[depth - 1] += 1
        section_id = ".".join(str(c) for c in counters)
        parent = stack[-1][1] if stack else None
        sec = Section(
            dataset=dataset,
            document_id=document_id,
            section_id=section_id,
            parent_section_id=parent.section_id if parent else None,
            title=normalize(h.title) or f"Untitled {section_id}",
            text=normalize(h.text),
            level=depth,
            path=[*(parent.path if parent else []), normalize(h.title) or f"Untitled {section_id}"],
            page_start=h.page_start,
            page_end=h.page_end,
        )
        if parent:
            parent.children.append(section_id)
        sections.append(sec)
        stack.append((h.depth, sec))

    doc = Document(dataset=dataset, document_id=document_id, title=title, sections=sections, metadata=metadata or {})
    validate_hierarchy(doc)
    return doc


def validate_hierarchy(doc: Document) -> None:
    """Raise HierarchyError if parent/child links or levels are inconsistent."""
    by_id = {s.section_id: s for s in doc.sections}
    for s in doc.sections:
        if s.parent_section_id is None:
            if s.level != 1:
                raise HierarchyError(f"{doc.document_id}:{s.section_id} is a root but has level {s.level}")
            continue
        parent = by_id.get(s.parent_section_id)
        if parent is None:
            raise HierarchyError(f"{doc.document_id}:{s.section_id} has orphan parent {s.parent_section_id}")
        if parent.level != s.level - 1:
            raise HierarchyError(f"{doc.document_id}:{s.section_id} level does not follow parent")
        if s.section_id not in parent.children:
            raise HierarchyError(f"{doc.document_id}:{s.section_id} missing from parent's children")
        if s.path[:-1] != parent.path:
            raise HierarchyError(f"{doc.document_id}:{s.section_id} path inconsistent with parent")
    for s in doc.sections:
        for c in s.children:
            if c not in by_id:
                raise HierarchyError(f"{doc.document_id}:{s.section_id} references missing child {c}")

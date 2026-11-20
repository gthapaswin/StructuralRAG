"""Flattened section corpus + structural text views used by every retriever."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from structuralrag.schema import Document, Section

PARENT_SNIPPET_CHARS = 400


class View(StrEnum):
    """Text representation of a section (ablations A1-A5)."""

    TEXT = "text"  # A1: content only
    TITLE = "title"  # A2: title + content
    PATH = "path"  # A3: full hierarchy path + content
    PARENT = "parent"  # A4: title + parent title/intro + content
    FULL = "full"  # A5: document title + path + parent intro + content


class SectionCorpus:
    def __init__(self, docs: list[Document]):
        self.docs = {d.document_id: d for d in docs}
        self.sections: list[Section] = [s for d in docs for s in d.sections]
        self.uid_to_idx = {s.uid: i for i, s in enumerate(self.sections)}
        self._doc_idx: dict[str, np.ndarray] = {}
        for i, s in enumerate(self.sections):
            self._doc_idx.setdefault(s.document_id, [])  # type: ignore[arg-type]
            self._doc_idx[s.document_id].append(i)  # type: ignore[attr-defined]
        self._doc_idx = {k: np.asarray(v) for k, v in self._doc_idx.items()}
        self.parent_idx = np.array(
            [
                self.uid_to_idx.get(f"{s.document_id}::{s.parent_section_id}", -1) if s.parent_section_id else -1
                for s in self.sections
            ]
        )

    def __len__(self) -> int:
        return len(self.sections)

    def doc_indices(self, document_id: str) -> np.ndarray:
        if document_id not in self._doc_idx:
            raise KeyError(f"Unknown document_id {document_id!r}")
        return self._doc_idx[document_id]

    def parent(self, i: int) -> Section | None:
        p = self.parent_idx[i]
        return self.sections[p] if p >= 0 else None

    def view(self, i: int, view: View) -> str:
        s = self.sections[i]
        if view is View.TEXT:
            return s.text
        if view is View.TITLE:
            return f"{s.title}. {s.text}"
        if view is View.PATH:
            return f"{s.path_str}. {s.text}"
        parent = self.parent(i)
        parent_ctx = f"{parent.title}: {parent.text[:PARENT_SNIPPET_CHARS]}" if parent else ""
        if view is View.PARENT:
            return f"{s.title}. {parent_ctx} {s.text}".strip()
        doc_title = self.docs[s.document_id].title
        return f"{doc_title}. {s.path_str}. {parent_ctx} {s.text}".strip()

    def views(self, view: View) -> list[str]:
        return [self.view(i, view) for i in range(len(self))]

    def path_texts(self) -> list[str]:
        """Structure-only field: hierarchy path of titles (no body text)."""
        return [s.path_str for s in self.sections]

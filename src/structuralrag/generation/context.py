"""Assemble the evidence given to the answer generator from retrieved sections.

Two modes, both capped at the same word budget so a difference in answer quality
comes from *structure*, not from giving the model more text:

* ``plain``: the retrieved sections' body text, like flat chunk RAG.
* ``hierarchical``: each section with its breadcrumb path, a short intro from its
  parent, and, for heading-only sections (no body text), the text of its children.
  Sections already shown inside another block are not repeated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from structuralrag.retrieval.corpus import SectionCorpus

PARENT_INTRO_WORDS = 50


@dataclass
class EvidenceBlock:
    section_id: str
    header: str
    body: str

    def render(self) -> str:
        return f"[{self.section_id}] {self.header}\n{self.body}".strip()


@dataclass
class Context:
    mode: str
    blocks: list[EvidenceBlock] = field(default_factory=list)

    @property
    def citable_ids(self) -> list[str]:
        return [b.section_id for b in self.blocks]

    @property
    def text(self) -> str:
        return "\n\n".join(b.render() for b in self.blocks)

    @property
    def n_words(self) -> int:
        return sum(len(b.body.split()) + len(b.header.split()) for b in self.blocks)

    def body(self, section_id: str) -> str:
        return next((b.body for b in self.blocks if b.section_id == section_id), "")


def _words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _fill(blocks: list[EvidenceBlock], budget: int) -> list[EvidenceBlock]:
    """Keep blocks in order until the word budget is spent; truncate the last one."""
    out, used = [], 0
    for b in blocks:
        cost = len(b.header.split())
        room = budget - used - cost
        if room <= 20:
            break
        body = _words(b.body, room)
        out.append(EvidenceBlock(b.section_id, b.header, body))
        used += cost + len(body.split())
    return out


def build_context(corpus: SectionCorpus, section_idx: list[int], mode: str, budget: int = 900) -> Context:
    if mode not in {"plain", "hierarchical"}:
        raise ValueError(f"Unknown context mode {mode!r}")
    sections = [corpus.sections[i] for i in section_idx]
    if len({s.document_id for s in sections}) > 1:
        raise ValueError("Context must come from a single document")

    if mode == "plain":
        blocks = [EvidenceBlock(s.section_id, "", s.text) for s in sections if s.text]
        return Context(mode, _fill(blocks, budget))

    doc = corpus.docs[sections[0].document_id] if sections else None
    shown: set[str] = set()
    blocks: list[EvidenceBlock] = []
    for s in sections:
        if s.section_id in shown:
            continue
        parent = doc.section(s.parent_section_id) if s.parent_section_id else None
        header = " › ".join(s.path)
        parts = []
        if parent and parent.text and parent.section_id not in shown:
            parts.append(
                f"(Context from {parent.section_id} {parent.title}: {_words(parent.text, PARENT_INTRO_WORDS)})"
            )
        if s.text:
            parts.append(s.text)
            blocks.append(EvidenceBlock(s.section_id, header, "\n".join(parts)))
            shown.add(s.section_id)
            continue
        # Heading-only section: its content lives in its children; make each child citable.
        if parts:
            blocks.append(EvidenceBlock(s.section_id, header, parts[0]))
        shown.add(s.section_id)
        for cid in s.children:
            child = doc.section(cid)
            if child.text and cid not in shown:
                blocks.append(EvidenceBlock(cid, " › ".join(child.path), child.text))
                shown.add(cid)
    return Context(mode, _fill(blocks, budget))

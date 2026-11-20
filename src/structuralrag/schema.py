"""Unified document representation shared by every dataset adapter and retriever."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Section(BaseModel):
    """One node in a document's hierarchy (chapter, section, subsection...)."""

    dataset: str
    document_id: str
    section_id: str = Field(description="Dotted hierarchical id, e.g. '3.2.1'")
    parent_section_id: str | None = None
    title: str
    text: str = ""
    level: int = Field(ge=1)
    path: list[str] = Field(default_factory=list, description="Titles from root to this node")
    children: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Corpus-wide unique identifier."""
        return f"{self.document_id}::{self.section_id}"

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)


class Document(BaseModel):
    dataset: str
    document_id: str
    title: str
    sections: list[Section]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ids_unique(self) -> Document:
        ids = [s.section_id for s in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate section ids in document {self.document_id}")
        return self

    def section(self, section_id: str) -> Section:
        for s in self.sections:
            if s.section_id == section_id:
                return s
        raise KeyError(f"{self.document_id} has no section {section_id}")

    def toc(self) -> str:
        """Indented table of contents, used for display and ToC-conditioned methods."""
        return "\n".join(f"{'  ' * (s.level - 1)}{s.section_id} {s.title}" for s in self.sections)


class Query(BaseModel):
    query_id: str
    dataset: str
    document_id: str
    question: str
    target_section_ids: list[str] = Field(min_length=1)
    split: str
    generator: str = Field(description="How the query was produced, e.g. 'ict', 'llm'")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def target_uids(self) -> set[str]:
        return {f"{self.document_id}::{sid}" for sid in self.target_section_ids}


class RetrievedSection(BaseModel):
    document_id: str
    section_id: str
    title: str
    hierarchy_path: list[str]
    score: float
    text: str

    @property
    def uid(self) -> str:
        return f"{self.document_id}::{self.section_id}"

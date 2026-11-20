from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from structuralrag.schema import RetrievedSection

Method = Literal["bm25", "dense", "hybrid", "structure_aware", "generative", "generative+structure"]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: str | None = Field(default=None, description="Restrict search to one document")
    method: Method = "structure_aware"
    top_k: int = Field(default=5, ge=1, le=50)


class RetrieveResponse(BaseModel):
    query: str
    method: Method
    latency_ms: float
    results: list[RetrievedSection]


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    num_sections: int


class SectionNode(BaseModel):
    section_id: str
    parent_section_id: str | None
    title: str
    level: int
    text: str

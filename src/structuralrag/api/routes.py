"""FastAPI service.  uv run uvicorn structuralrag.api.routes:app --reload"""

from __future__ import annotations

import json
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from structuralrag.api.schemas import DocumentSummary, RetrieveRequest, RetrieveResponse, SectionNode
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents

app = FastAPI(title="StructuralRAG", version="0.1.0", description="Structure-aware section retrieval")


@lru_cache(maxsize=1)
def get_suite() -> RetrieverSuite:
    path = DATA_DIR / "test_documents.jsonl"
    if not path.exists():
        raise HTTPException(503, "No processed data. Run scripts/prepare_data.py first.")
    params = StructureParams()
    results = REPORTS_DIR / "results.json"
    if results.exists():
        params = StructureParams(**json.loads(results.read_text())["tuned_structure_params"])
    cfg = load_config()
    return RetrieverSuite(SectionCorpus(load_documents(path)), cfg["retrieval"]["embedding_model"], CACHE_DIR, params)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    return [
        DocumentSummary(document_id=d.document_id, title=d.title, num_sections=len(d.sections))
        for d in get_suite().corpus.docs.values()
    ]


@app.get("/documents/{document_id}", response_model=list[SectionNode])
def document(document_id: str) -> list[SectionNode]:
    doc = get_suite().corpus.docs.get(document_id)
    if doc is None:
        raise HTTPException(404, f"Unknown document {document_id}")
    return [SectionNode(**s.model_dump(include=set(SectionNode.model_fields))) for s in doc.sections]


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    suite = get_suite()
    if req.document_id and req.document_id not in suite.corpus.docs:
        raise HTTPException(404, f"Unknown document {req.document_id}")
    retriever = suite.get(req.method)
    t0 = time.perf_counter()
    results = retriever.search(req.query, top_k=req.top_k, document_id=req.document_id)
    return RetrieveResponse(
        query=req.query, method=req.method, latency_ms=(time.perf_counter() - t0) * 1000, results=results
    )


@app.get("/experiments")
def experiments() -> dict:
    path = REPORTS_DIR / "results.json"
    if not path.exists():
        raise HTTPException(404, "No results yet. Run scripts/run_evaluation.py.")
    data = json.loads(path.read_text())
    return {k: data[k] for k in ("tuned_structure_params", "results", "significance", "ablation_components")}

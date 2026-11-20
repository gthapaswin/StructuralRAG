"""FastAPI service.  uv run uvicorn structuralrag.api.routes:app --reload"""

from __future__ import annotations

import json
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from structuralrag.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    CitedSentence,
    DocumentSummary,
    RetrieveRequest,
    RetrieveResponse,
    SectionNode,
)
from structuralrag.generation.answerer import Answerer
from structuralrag.generation.context import build_context
from structuralrag.llm.ollama_client import OllamaClient, OllamaError
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.generative import GenerativeFusionRetriever, GenerativeRetriever
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


@lru_cache(maxsize=1)
def get_generative() -> dict:
    """LLM retrievers backed by Ollama; raises 503 if Ollama or the model is unavailable."""
    llm = load_config()["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite")
    client.check()
    gen = GenerativeRetriever(get_suite().corpus, client, llm["max_ids"], snippets=True)
    return {"generative": gen, "generative+structure": GenerativeFusionRetriever(gen, get_suite().structure_aware())}


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
    if req.method.startswith("generative"):
        if not req.document_id:
            raise HTTPException(422, "LLM methods need document_id: they read that document's table of contents.")
        try:
            retriever = get_generative()[req.method]
        except OllamaError as e:
            raise HTTPException(503, str(e)) from e
    else:
        retriever = suite.get(req.method)
    t0 = time.perf_counter()
    results = retriever.search(req.query, top_k=req.top_k, document_id=req.document_id)
    return RetrieveResponse(
        query=req.query, method=req.method, latency_ms=(time.perf_counter() - t0) * 1000, results=results
    )


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    cfg = load_config()
    llm, cache = cfg["llm"], CACHE_DIR / "ollama_cache.sqlite"
    gen = OllamaClient(llm["model"], llm["host"], cache_path=cache)
    gen.check()
    checker = OllamaClient(cfg["answers"]["checker_model"], llm["host"], cache_path=cache)
    checker.check()
    return Answerer(gen, checker)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """Cited answer from structure-aware retrieval + hierarchical context, with claim checking."""
    suite = get_suite()
    if req.document_id not in suite.corpus.docs:
        raise HTTPException(404, f"Unknown document {req.document_id}")
    try:
        answerer = get_answerer()
    except OllamaError as e:
        raise HTTPException(503, str(e)) from e
    a_cfg = load_config()["answers"]
    t0 = time.perf_counter()
    sa = suite.structure_aware()
    idx = [i for i, _ in sa.rank(sa.score(req.query), a_cfg["top_k"], req.document_id)]
    ctx = build_context(suite.corpus, idx, "hierarchical", a_cfg["context_words"])
    ans = answerer.answer(req.query, ctx, check=req.check_claims)
    return AnswerResponse(
        query=req.query,
        answer=ans.text,
        abstained=ans.abstained,
        abstain_reason=ans.abstain_reason,
        sentences=[CitedSentence(text=s.text, citations=s.citations) for s in ans.sentences],
        removed_claims=[CitedSentence(text=s.text, citations=s.citations) for s in ans.removed],
        evidence_sections=ctx.citable_ids,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


@app.get("/experiments")
def experiments() -> dict:
    path = REPORTS_DIR / "results.json"
    if not path.exists():
        raise HTTPException(404, "No results yet. Run scripts/run_evaluation.py.")
    data = json.loads(path.read_text())
    return {k: data[k] for k in ("tuned_structure_params", "results", "significance", "ablation_components")}

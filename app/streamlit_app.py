"""StructuralRAG dashboard.  uv run streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from structuralrag.ingestion.pdf_parser import parse_pdf
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import METHODS, RetrieverSuite
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents

st.set_page_config(page_title="StructuralRAG", layout="wide")
CFG = load_config()
RESULTS = REPORTS_DIR / "results.json"


@st.cache_data
def load_results() -> dict | None:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None


def tuned_params() -> StructureParams:
    res = load_results()
    return StructureParams(**res["tuned_structure_params"]) if res else StructureParams()


@st.cache_resource(show_spinner="Building indexes (first run only)...")
def benchmark_suite() -> RetrieverSuite:
    docs = load_documents(DATA_DIR / "test_documents.jsonl")
    return RetrieverSuite(SectionCorpus(docs), CFG["retrieval"]["embedding_model"], CACHE_DIR, tuned_params())


@st.cache_resource(show_spinner="Parsing PDF and building indexes...")
def pdf_suite(pdf_bytes: bytes, name: str) -> RetrieverSuite:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(pdf_bytes)
        doc = parse_pdf(path)
    return RetrieverSuite(SectionCorpus([doc]), CFG["retrieval"]["embedding_model"], None, tuned_params())


st.title("StructuralRAG")
st.caption("Beyond chunking: structure-aware retrieval over long, hierarchical documents.")

with st.sidebar:
    source = st.radio("Corpus", ["GovReport benchmark", "Upload a PDF"])
    suite: RetrieverSuite | None = None
    if source == "Upload a PDF":
        up = st.file_uploader("PDF", type="pdf")
        if up:
            suite = pdf_suite(up.getvalue(), up.name)
    elif (DATA_DIR / "test_documents.jsonl").exists():
        suite = benchmark_suite()
    else:
        st.error("Run `uv run python scripts/prepare_data.py` first.")

tab_retrieve, tab_explore, tab_bench, tab_analytics = st.tabs(
    ["Retrieval playground", "Document explorer", "Benchmarks", "Analytics"]
)

if suite:
    docs = list(suite.corpus.docs.values())
    labels = {f"{d.title[:70]}  ({d.document_id})": d.document_id for d in docs}

    with tab_retrieve:
        c1, c2, c3 = st.columns([3, 1, 1])
        doc_label = c1.selectbox("Document", ["All documents", *labels])
        method = c2.selectbox("Method", METHODS, index=METHODS.index("structure_aware"))
        top_k = c3.slider("Top K", 1, 20, 5)
        query = st.text_input("Question", placeholder="e.g. How do FDA and USDA collaborate on oversight?")
        compare = st.checkbox("Compare all methods side by side")
        if query:
            doc_id = labels.get(doc_label)
            methods = METHODS if compare else (method,)
            cols = st.columns(len(methods))
            for col, m in zip(cols, methods, strict=True):
                col.subheader(m)
                for r in suite.get(m).search(query, top_k=top_k, document_id=doc_id):
                    with col.expander(f"{r.section_id} · {r.title}  —  {r.score:.3f}"):
                        st.caption(" › ".join(r.hierarchy_path) + f"  ·  {r.document_id}")
                        st.write(r.text[:1500] or "_(heading only — no body text)_")

    with tab_explore:
        doc = suite.corpus.docs[labels[st.selectbox("Document ", list(labels))]]
        st.markdown(f"**{doc.title}** — {len(doc.sections)} sections, max depth {max(s.level for s in doc.sections)}")
        for s in doc.sections:
            with st.expander(f"{'  ' * (s.level - 1)}{s.section_id}  {s.title}"):
                st.caption(s.path_str)
                st.write(s.text or "_(heading only)_")

res = load_results()
with tab_bench:
    if not res:
        st.info("Run `uv run python scripts/run_evaluation.py` to produce benchmark results.")
    else:
        p = res["tuned_structure_params"]
        st.markdown(f"Tuned on dev split: α={p['alpha']}, β={p['beta']}, BM25 weight={p['sparse_weight']}")
        df = pd.DataFrame(res["results"])
        scope = st.radio("Scope", sorted(df.scope.unique()), horizontal=True)
        view = df[df.scope == scope].set_index("method")
        metrics = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
        st.dataframe(view[metrics + ["same_branch"]].style.format("{:.3f}"), use_container_width=True)
        st.bar_chart(view[metrics])
        st.subheader("Significance (structure-aware minus baseline)")
        st.dataframe(pd.DataFrame(res["significance"]).query("scope == @scope"), use_container_width=True)

with tab_analytics:
    if res:
        df = pd.DataFrame(res["results"])
        st.subheader("Latency (ms, single query, end-to-end)")
        st.bar_chart(df[df.scope == "document"].set_index("method")[["latency_p50_ms", "latency_p95_ms"]])
        st.subheader("Ablation: what text does each section get indexed with?")
        abl = pd.DataFrame(res["ablation_views"]).pivot(index="view", columns="method", values="mrr@10")
        st.dataframe(abl.style.format("{:.3f}"), use_container_width=True)
        st.subheader("Ablation: structure-aware score components")
        st.dataframe(pd.DataFrame(res["ablation_components"]).set_index("configuration"), use_container_width=True)
        st.subheader("MRR@10 by depth of the target section")
        st.dataframe(pd.DataFrame(res["mrr_by_target_level"]).set_index("method"), use_container_width=True)
        st.subheader("Efficiency")
        st.json(res["efficiency"])

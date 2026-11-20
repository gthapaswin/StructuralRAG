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


def _warm(suite: RetrieverSuite) -> RetrieverSuite:
    """Build every index now (behind the spinner) instead of on the first query."""
    for m in METHODS:
        suite.get(m)
    return suite


@st.cache_resource(show_spinner="Building indexes (first run only, ~30 s)...")
def benchmark_suite() -> RetrieverSuite:
    docs = load_documents(DATA_DIR / "test_documents.jsonl")
    return _warm(RetrieverSuite(SectionCorpus(docs), CFG["retrieval"]["embedding_model"], CACHE_DIR, tuned_params()))


@st.cache_resource(show_spinner="Parsing PDF and building indexes...")
def pdf_suite(pdf_bytes: bytes, name: str) -> RetrieverSuite:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(pdf_bytes)
        doc = parse_pdf(path)
    return _warm(RetrieverSuite(SectionCorpus([doc]), CFG["retrieval"]["embedding_model"], None, tuned_params()))


GENERIC_HEADINGS = {
    "background",
    "introduction",
    "scope and methodology",
    "objectives, scope, and methodology",
    "letter",
    "conclusions",
    "conclusion",
    "agency comments",
    "results in brief",
}


def doc_label_for(doc) -> str:
    """GovReport has no report titles; label by id plus the first informative top-level heading."""
    tops = [s.title for s in doc.sections if s.level == 1]
    hint = next((t for t in tops if t.lower() not in GENERIC_HEADINGS), tops[0] if tops else "")
    return f"{doc.document_id.removeprefix('GAO_')} — {hint[:70]}"


def show_hit(r) -> None:
    with st.expander(f"{r.section_id} · {r.title}  —  score {r.score:.3f}"):
        st.caption(" › ".join(r.hierarchy_path) + f"  ·  {r.document_id}")
        st.write(r.text[:1500] or "_(heading only — no body text)_")


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
    labels = {doc_label_for(d): d.document_id for d in docs}

    with tab_retrieve:
        doc_label = st.selectbox("Document", ["All documents", *labels])
        c1, c2 = st.columns([2, 1])
        method = c1.selectbox("Method", METHODS, index=METHODS.index("structure_aware"))
        top_k = c2.slider("Top K", 1, 20, 5)
        query = st.text_input(
            "Question", placeholder="e.g. What did agencies say about cybersecurity workforce shortages?"
        )
        compare = st.checkbox("Compare all methods")
        if query:
            doc_id = labels.get(doc_label)
            hits = {m: suite.get(m).search(query, top_k=top_k, document_id=doc_id) for m in METHODS}
            if compare:
                st.markdown("**Top results per method** — the same section in several columns means the methods agree")
                short = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid", "structure_aware": "Structure"}
                table = pd.DataFrame(
                    {short[m]: [f"{r.section_id} {r.title[:22]}…" for r in rs] for m, rs in hits.items()},
                    index=pd.RangeIndex(1, top_k + 1, name="#"),
                )
                st.dataframe(table, width="stretch")
                for tab, m in zip(st.tabs([short[m] for m in METHODS]), METHODS, strict=True):
                    with tab:
                        for r in hits[m]:
                            show_hit(r)
            else:
                for r in hits[method]:
                    show_hit(r)

    with tab_explore:
        doc = suite.corpus.docs[labels[st.selectbox("Document ", list(labels))]]
        depth = max(s.level for s in doc.sections)
        st.markdown(f"**{doc_label_for(doc)}** — {len(doc.sections)} sections, max depth {depth}")
        for s in doc.sections:
            # Indent with em-spaces: leading ASCII spaces would render as a Markdown code block.
            with st.expander(f"{'\u2003' * 2 * (s.level - 1)}{s.section_id}  {s.title}"):
                st.caption(s.path_str)
                st.write(s.text or "_(heading only)_")

res = load_results()


def clean_method(name: str) -> str:
    return name.split("[")[0]


with tab_bench:
    if not res:
        st.info("Run `uv run python scripts/run_evaluation.py` to produce benchmark results.")
    else:
        p = res["tuned_structure_params"]
        st.markdown(
            f"GovReport test set: **{res['results'][0]['n_queries']} queries**. "
            f"Structure-aware weights tuned on a separate dev split: "
            f"path α={p['alpha']}, parent β={p['beta']}, BM25 weight={p['sparse_weight']}."
        )
        df = pd.DataFrame(res["results"])
        df["method"] = df["method"].map(clean_method)
        scope = st.radio(
            "Scope",
            ["document", "corpus"],
            horizontal=True,
            format_func=lambda s: {"document": "Within one report", "corpus": "Across all reports"}[s],
        )
        view = df[df.scope == scope].set_index("method")
        metrics = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
        st.table(view[metrics + ["same_branch"]].round(3))
        st.caption("same_branch: top result is in the same top-level chapter as the correct section.")
        st.bar_chart(view[["recall@1", "mrr@10", "ndcg@10"]], stack=False)
        st.subheader("Is structure-aware significantly better?")
        sig = pd.DataFrame(res["significance"]).query("scope == @scope").drop(columns="scope")
        sig["significant (p<0.05, both tests)"] = (sig.p_permutation < 0.05) & (sig.p_wilcoxon < 0.05)
        st.table(sig.round(4).set_index(["baseline", "metric"]))

with tab_analytics:
    if res:
        df = pd.DataFrame(res["results"])
        df["method"] = df["method"].map(clean_method)
        st.subheader("Latency per query (ms, end-to-end)")
        lat = df[df.scope == "document"].set_index("method")[["latency_p50_ms", "latency_p95_ms"]]
        st.bar_chart(lat.rename(columns={"latency_p50_ms": "P50", "latency_p95_ms": "P95"}), stack=False)
        st.subheader("Ablation: what text is each section indexed with? (MRR@10)")
        abl = pd.DataFrame(res["ablation_views"]).pivot(index="view", columns="method", values="mrr@10")
        st.table(abl.reindex(["text", "title", "path", "parent", "full"]).round(3))
        st.caption("text → +title → +hierarchy path → +parent context → all metadata (A1–A5).")
        st.subheader("Ablation: structure-aware score components")
        comp = pd.DataFrame(res["ablation_components"])
        comp["configuration"] = comp["configuration"].str.replace("+ ", "+", regex=False)  # avoid Markdown bullets
        st.table(comp.set_index("configuration").round(3))
        st.subheader("MRR@10 by depth of the correct section")
        depth = pd.DataFrame(res["mrr_by_target_level"]).set_index("method")
        st.table(depth.rename(columns=lambda c: f"depth {c}").round(3))
        st.subheader("Efficiency")
        st.json(res["efficiency"])

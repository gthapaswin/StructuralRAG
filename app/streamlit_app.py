"""StructuralRAG dashboard.  uv run streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import streamlit as st

from structuralrag.generation.answerer import Answerer
from structuralrag.generation.context import build_context
from structuralrag.ingestion.pdf_parser import parse_pdf
from structuralrag.llm.ollama_client import OllamaClient, OllamaError
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import METHODS, RetrieverSuite
from structuralrag.retrieval.generative import GenerativeFusionRetriever, GenerativeRetriever
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents

st.set_page_config(page_title="StructuralRAG", layout="wide")
CFG = load_config()
RESULTS = REPORTS_DIR / "results.json"
GEN_RESULTS = REPORTS_DIR / "generative_results.json"
GEN_METHODS = ("generative", "generative+structure")
SHORT = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "structure_aware": "Structure",
    "generative": "LLM",
    "generative+structure": "LLM+Struct",
}


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


@st.cache_resource(show_spinner=False)
def generative_retrievers(_suite: RetrieverSuite, key: str) -> dict | None:
    """LLM retrievers if Ollama is running with the configured model, else None."""
    llm = CFG["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=CFG["seed"])
    try:
        client.check()
    except OllamaError:
        return None
    gen = GenerativeRetriever(_suite.corpus, client, llm["max_ids"], snippets=True)
    out = {"generative": gen, "generative+structure": GenerativeFusionRetriever(gen, _suite.structure_aware())}
    checker = OllamaClient(CFG["answers"]["checker_model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite")
    try:
        checker.check()
    except OllamaError:
        checker = None
    out["answerer"] = Answerer(client, checker)
    return out


def show_answer(question: str, doc_id: str) -> None:
    """Grounded answer from structure-aware top sections + hierarchical context, with claim check."""
    a_cfg = CFG["answers"]
    sa = suite.structure_aware()
    idx = [i for i, _ in sa.rank(sa.score(question), a_cfg["top_k"], doc_id)]
    ctx = build_context(suite.corpus, idx, "hierarchical", a_cfg["context_words"])
    answerer = gen_retrievers["answerer"]
    with st.spinner("Writing answer and checking each claim..."):
        ans = answerer.answer(question, ctx, check=answerer.checker is not None)
    with st.container(border=True):
        st.markdown("**Answer**")
        if ans.abstained:
            st.warning(ans.text)
        else:
            for sent in ans.sentences:
                st.markdown(f"{sent.text} `{'` `'.join(sent.citations)}`")
        if ans.removed:
            with st.expander(f"{len(ans.removed)} claim(s) removed by the checker (not supported by cited text)"):
                for sent in ans.removed:
                    st.markdown(f"~~{sent.text}~~ `{'` `'.join(sent.citations)}`")
        with st.expander(f"Evidence given to the model ({ctx.n_words} words, sections {', '.join(ctx.citable_ids)})"):
            st.text(ctx.text)
        if answerer.checker is None:
            st.caption(f"Claim check off: `{a_cfg['checker_model']}` not found in Ollama.")


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
    tops = [s.title for s in doc.sections if s.level == 1 and not s.title.startswith("Untitled")]
    hint = next((t for t in tops if t.lower() not in GENERIC_HEADINGS), tops[0] if tops else "(no headings)")
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
    suite_key = source
    if source == "Upload a PDF":
        up = st.file_uploader("PDF", type="pdf")
        if up:
            suite = pdf_suite(up.getvalue(), up.name)
            suite_key = f"pdf:{up.name}:{up.size}"
    elif (DATA_DIR / "test_documents.jsonl").exists():
        suite = benchmark_suite()
    else:
        st.error("Run `uv run python scripts/prepare_data.py` first.")
    gen_retrievers = generative_retrievers(suite, suite_key) if suite else None
    if suite:
        if gen_retrievers:
            st.success(f"LLM methods on: `{CFG['llm']['model']}` via Ollama")
        else:
            st.caption(f"LLM methods off — start Ollama with `{CFG['llm']['model']}` to enable them.")

tab_retrieve, tab_explore, tab_bench, tab_gen, tab_analytics = st.tabs(
    ["Retrieval playground", "Document explorer", "Benchmarks", "Generative", "Analytics"]
)

if suite:
    docs = list(suite.corpus.docs.values())
    labels = {doc_label_for(d): d.document_id for d in docs}

    with tab_retrieve:
        doc_label = st.selectbox("Document", ["All documents", *labels])
        doc_id = labels.get(doc_label)
        available = list(METHODS) + (list(GEN_METHODS) if gen_retrievers else [])
        c1, c2 = st.columns([2, 1])
        method = c1.selectbox("Method", available, index=available.index("structure_aware"))
        top_k = c2.slider("Top K", 1, 20, 5)
        query = st.text_input(
            "Question", placeholder="e.g. What did agencies say about cybersecurity workforce shortages?"
        )
        compare = st.checkbox("Compare all methods")
        want_answer = bool(gen_retrievers) and st.checkbox(
            "Generate a cited answer", disabled=not doc_id, help="Pick a single report to enable."
        )
        if method in GEN_METHODS and not doc_id:
            st.info("LLM methods read one report's table of contents: pick a document above.")
        elif query:
            if want_answer and doc_id:
                show_answer(query, doc_id)
            run = [m for m in available if compare or m == method]
            run = [m for m in run if m not in GEN_METHODS or doc_id]  # LLM methods need a document
            hits = {}
            for m in run:
                retriever = gen_retrievers[m] if m in GEN_METHODS else suite.get(m)
                ctx = st.spinner(f"Asking {CFG['llm']['model']}...") if m in GEN_METHODS else nullcontext()
                with ctx:
                    hits[m] = retriever.search(query, top_k=top_k, document_id=doc_id)
            if compare:
                st.markdown("**Top results per method** — the same section in several columns means the methods agree")
                table = pd.DataFrame(
                    {SHORT[m]: pd.Series([f"{r.section_id} {r.title[:22]}…" for r in rs]) for m, rs in hits.items()}
                ).fillna("")
                table.index = pd.RangeIndex(1, len(table) + 1, name="#")
                st.dataframe(table, width="stretch")
                if gen_retrievers and not doc_id:
                    st.caption("LLM methods are skipped across all documents; pick one report to include them.")
                for tab, m in zip(st.tabs([SHORT[m] for m in hits]), hits, strict=True):
                    with tab:
                        for r in hits[m]:
                            show_hit(r)
            else:
                if method == "generative" and len(hits[method]) < top_k:
                    st.caption("The LLM lists at most a few sections; it does not rank the rest.")
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

with tab_gen:
    if not GEN_RESULTS.exists():
        st.info(
            "Run `uv run python scripts/generate_questions.py` and `scripts/run_generative_eval.py` (needs Ollama)."
        )
    else:
        g = json.loads(GEN_RESULTS.read_text())
        st.markdown(
            f"Zero-shot generative retrieval with `{g['model']}`: the model reads only the table of contents "
            "and returns section ids. Output is constrained to the report's real ids, so every prediction is valid."
        )
        names = {"ict": "Sentence-from-section (ICT)", "llm": "LLM-written questions"}
        qset = st.radio("Question set", list(g["query_sets"]), horizontal=True, format_func=names.get)
        data = g["query_sets"][qset]
        st.caption(f"{data['n_queries']} questions, document scope.")
        tbl = pd.DataFrame(data["results"]).set_index("method")
        st.table(tbl[["recall@1", "recall@5", "mrr@10", "ndcg@10", "mean_returned"]].round(3))
        st.bar_chart(tbl[["recall@1", "mrr@10"]], stack=False)
        st.subheader("Paired significance tests")
        sig = pd.DataFrame(data["significance"])
        sig["significant (p<0.05, both tests)"] = (sig.p_permutation < 0.05) & (sig.p_wilcoxon < 0.05)
        st.table(sig.round(4).set_index(["method", "vs", "metric"]))
        st.subheader("MRR@10 by depth of the correct section")
        st.table(pd.DataFrame(data["mrr_by_target_level"]).set_index("method").rename(columns=lambda c: f"depth {c}"))
        if "llm_question_stats" in g:
            st.subheader("How the LLM questions were filtered")
            st.json(g["llm_question_stats"])
        st.caption(f"LLM latency per query (uncached): {g['generative_latency_ms']}")

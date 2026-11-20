"""Evaluate generative (ToC-conditioned) retrieval and all baselines on two query sets:
ICT queries (sentence taken from the section) and LLM-written paraphrased questions.

Needs Ollama running with the model in configs/experiment.yaml (llm.model).
uv run python scripts/run_generative_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np
import pandas as pd

from structuralrag.evaluation.evaluator import RunResult, evaluate
from structuralrag.evaluation.statistical_tests import paired_test
from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import METHODS, RetrieverSuite
from structuralrag.retrieval.generative import GenerativeFusionRetriever, GenerativeRetriever
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.schema import Query
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_generative_eval")
METRICS = ("recall@1", "recall@5", "mrr@10", "ndcg@10")


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return out


def llm_latency(gen: GenerativeRetriever, queries: list[Query], n: int = 20) -> dict[str, float]:
    """Uncached end-to-end latency on fresh LLM calls."""
    fresh = GenerativeRetriever(gen.corpus, OllamaClient(gen.client.model, gen.client.host), gen.max_ids, gen.snippets)
    times = []
    for q in queries[:n]:
        t0 = time.perf_counter()
        fresh.predict_ids(q.question, q.document_id)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    llm = cfg["llm"]

    params = StructureParams(**json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"])
    corpus = SectionCorpus(load_documents(DATA_DIR / "test_documents.jsonl"))
    suite = RetrieverSuite(corpus, cfg["retrieval"]["embedding_model"], CACHE_DIR, params)
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    client.check()

    gen_toc = GenerativeRetriever(corpus, client, llm["max_ids"], snippets=False)
    gen_snip = GenerativeRetriever(corpus, client, llm["max_ids"], snippets=True)
    fusion = GenerativeFusionRetriever(gen_snip, suite.structure_aware())

    ict = load_queries(DATA_DIR / "test_queries.jsonl")
    ict = random.Random(cfg["seed"]).sample(ict, min(llm["generative_ict_sample"], len(ict)))
    query_sets = {"ict": ict, "llm": load_queries(DATA_DIR / "test_queries_llm.jsonl")}

    out: dict = {"model": llm["model"], "structure_params": params.__dict__, "query_sets": {}}
    md = [
        "# Generative retrieval and LLM-written questions\n",
        f"LLM: `{llm['model']}` via Ollama (temperature 0, output constrained to the report's section ids). "
        f"Document scope. Structure-aware weights from the main evaluation (α={params.alpha}, β={params.beta}).\n",
    ]
    per_query = []
    for name, queries in query_sets.items():
        log.info("Query set %s: %d queries", name, len(queries))
        runs: dict[str, RunResult] = {}
        for m in METHODS:
            runs[m] = evaluate(suite.get(m), queries, scope="document", latency_sample=0)
        for key, r in (
            ("generative_toc", gen_toc),
            ("generative_toc_snippets", gen_snip),
            ("generative+structure", fusion),
        ):
            t0 = time.perf_counter()
            runs[key] = evaluate(r, queries, scope="document", batch_size=16, latency_sample=0)
            log.info("%s on %s: %.0fs", key, name, time.perf_counter() - t0)

        rows = []
        for key, r in runs.items():
            s = r.summary()
            s["method"] = key
            s["mean_returned"] = float(r.metric_array("n_returned").mean())
            rows.append(s)
            df = pd.DataFrame(r.per_query)
            df["method"], df["query_set"] = key, name
            per_query.append(df)
        sig = []
        for a, b in [
            ("generative_toc_snippets", "structure_aware"),
            ("generative+structure", "structure_aware"),
            ("generative+structure", "hybrid"),
            ("structure_aware", "hybrid"),
            ("structure_aware", "bm25"),
            ("structure_aware", "dense"),
        ]:
            for metric in ("mrr@10", "recall@1"):
                t = paired_test(runs[a].metric_array(metric), runs[b].metric_array(metric))
                sig.append({"method": a, "vs": b, "metric": metric, **t})
        depth = (
            pd.concat([pd.DataFrame(r.per_query).assign(method=k) for k, r in runs.items()])
            .groupby(["method", "target_level"])["mrr@10"]
            .mean()
            .unstack()
            .round(3)
        )
        out["query_sets"][name] = {
            "n_queries": len(queries),
            "results": rows,
            "significance": sig,
            "mrr_by_target_level": depth.reset_index().to_dict(orient="records"),
        }
        title = {"ict": "ICT queries (sentence from the section)", "llm": "LLM-written paraphrased questions"}[name]
        md.append(f"\n## {title} — {len(queries)} queries\n")
        md.append(fmt_table(rows, ["method", *METRICS, "same_branch", "mean_returned"]))
        md.append("\n**Paired tests**\n")
        md.append(fmt_table(sig, ["method", "vs", "metric", "mean_diff", "p_permutation", "p_wilcoxon"]))
        depth_rows = [{"method": i, **{f"depth {c}": v for c, v in r.items()}} for i, r in depth.iterrows()]
        md.append("\n**MRR@10 by target depth**\n")
        md.append(fmt_table(depth_rows, list(depth_rows[0])))

    out["generative_latency_ms"] = llm_latency(gen_snip, query_sets["llm"])
    q_stats = DATA_DIR / "llm_question_stats.json"
    if q_stats.exists():
        out["llm_question_stats"] = json.loads(q_stats.read_text())
        md.append(f"\n## Question generation\n\n```\n{out['llm_question_stats']}\n```\n")
    md.append(f"\nGenerative latency (uncached, ToC+snippets): {out['generative_latency_ms']}\n")
    write_json(REPORTS_DIR / "generative_results.json", out)
    pd.concat(per_query).to_csv(REPORTS_DIR / "generative_per_query.csv", index=False)
    (REPORTS_DIR / "generative_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

"""Tune on dev, evaluate all methods + ablations on test, write reports/.

uv run python scripts/run_evaluation.py [--config configs/experiment.yaml]
"""

from __future__ import annotations

import argparse
import logging
import resource
import time

import pandas as pd

from structuralrag.evaluation.evaluator import evaluate
from structuralrag.evaluation.statistical_tests import paired_test
from structuralrag.evaluation.tuning import tune_structure
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.factory import METHODS, RetrieverSuite
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import environment_info, set_seed

log = logging.getLogger("run_evaluation")


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    model = cfg["retrieval"]["embedding_model"]
    REPORTS_DIR.mkdir(exist_ok=True)

    # ---- Dev: tune structure-aware weights -------------------------------------------
    dev_suite = RetrieverSuite(SectionCorpus(load_documents(DATA_DIR / "dev_documents.jsonl")), model, CACHE_DIR)
    dev_q = load_queries(DATA_DIR / "dev_queries.jsonl")
    params, trials = tune_structure(dev_suite, dev_q, cfg["tuning"])
    log.info("Best structure params on dev: %s", params)

    # ---- Test -----------------------------------------------------------------------
    test_docs = load_documents(DATA_DIR / "test_documents.jsonl")
    test_q = load_queries(DATA_DIR / "test_queries.jsonl")
    corpus = SectionCorpus(test_docs)
    t0 = time.perf_counter()
    suite = RetrieverSuite(corpus, model, cache_dir=None, structure_params=params)  # no cache: honest build time
    build = {}
    for m in METHODS:
        t = time.perf_counter()
        suite.get(m)
        build[m] = time.perf_counter() - t
    build["total_s"] = time.perf_counter() - t0

    results, per_query_frames, comparisons = [], [], []
    for scope in cfg["evaluation"]["scopes"]:
        runs = {m: evaluate(suite.get(m), test_q, scope=scope) for m in METHODS}
        for m, r in runs.items():
            results.append(r.summary())
            df = pd.DataFrame(r.per_query)
            df["method"], df["scope"] = m, scope
            per_query_frames.append(df)
        sa = runs["structure_aware"]
        for base in ("bm25", "dense", "hybrid"):
            for metric in ("mrr@10", "recall@1"):
                t = paired_test(sa.metric_array(metric), runs[base].metric_array(metric))
                comparisons.append({"scope": scope, "baseline": base, "metric": metric, **t})

    # ---- Ablations A1-A5: does structure help, or just more text? -------------------
    ablations = []
    for view in [View(v) for v in cfg["evaluation"]["ablation_views"]]:
        for kind in ("bm25", "dense", "hybrid"):
            r = evaluate(getattr(suite, kind)(view), test_q, scope="document", latency_sample=0)
            s = r.summary()
            ablations.append(
                {"view": view.value, "method": kind, **{k: s[k] for k in ("recall@1", "recall@5", "mrr@10", "ndcg@10")}}
            )
    # Component ablation of the structure-aware score
    comp_abl = []
    sa_r = suite.structure_aware()
    for label, p in {
        "content only": StructureParams(0, 0, 0),
        "+ path": StructureParams(params.alpha, 0, 0),
        "+ parent": StructureParams(0, params.beta, 0),
        "+ path + parent": StructureParams(params.alpha, params.beta, 0),
        "+ path + parent + BM25(full)": params,
    }.items():
        sa_r.params = p
        s = evaluate(sa_r, test_q, scope="document", latency_sample=0).summary()
        comp_abl.append({"configuration": label, **{k: s[k] for k in ("recall@1", "recall@5", "mrr@10", "ndcg@10")}})
    sa_r.params = params

    # ---- By target depth --------------------------------------------------------------
    pq = pd.concat(per_query_frames)
    by_level = (
        pq[pq.scope == "document"].groupby(["method", "target_level"])["mrr@10"].mean().unstack().round(3).reset_index()
    )

    emb_bytes = sum(d.emb.nbytes for d in suite._dense.values()) + suite.structure_aware().path_emb.nbytes
    efficiency = {
        "index_build_seconds": build,
        "embedding_storage_mb": emb_bytes / 1e6,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,  # bytes on macOS
        "corpus_sections": len(corpus),
    }

    out = {
        "config": cfg,
        "environment": environment_info(),
        "tuned_structure_params": params.__dict__,
        "tuning_trials": trials,
        "results": results,
        "significance": comparisons,
        "ablation_views": ablations,
        "ablation_components": comp_abl,
        "mrr_by_target_level": by_level.to_dict(orient="records"),
        "efficiency": efficiency,
    }
    write_json(REPORTS_DIR / "results.json", out)
    pq.to_csv(REPORTS_DIR / "per_query.csv", index=False)

    cols = [
        "method",
        "n_queries",
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "same_branch",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    md = [
        "# StructuralRAG benchmark results\n",
        f"Dataset: GovReport (structure), test docs={len(test_docs)}, "
        f"sections={len(corpus)}, queries={len(test_q)} (ICT). Embedding model: `{model}`.\n",
        f"Tuned on dev: alpha={params.alpha}, beta={params.beta}, sparse_weight={params.sparse_weight}\n",
    ]
    for scope in cfg["evaluation"]["scopes"]:
        md.append(f"\n## Scope: {scope}\n")
        md.append(fmt_table([r for r in results if r["scope"] == scope], cols))
    md.append("\n## Structure-aware vs baselines (paired, per query)\n")
    md.append(fmt_table(comparisons, ["scope", "baseline", "metric", "mean_diff", "p_permutation", "p_wilcoxon"]))
    md.append("\n## Ablation: text views (document scope)\n")
    md.append(fmt_table(ablations, ["view", "method", "recall@1", "recall@5", "mrr@10", "ndcg@10"]))
    md.append("\n## Ablation: structure-aware score components (document scope)\n")
    md.append(fmt_table(comp_abl, ["configuration", "recall@1", "recall@5", "mrr@10", "ndcg@10"]))
    md.append("\n## MRR@10 by target section depth (document scope)\n")
    level_rows = [{str(k): v for k, v in r.items()} for r in by_level.to_dict(orient="records")]
    md.append(fmt_table(level_rows, [str(c) for c in by_level.columns]))
    md.append(f"\n\n## Efficiency\n\n```\n{efficiency}\n```\n")
    (REPORTS_DIR / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

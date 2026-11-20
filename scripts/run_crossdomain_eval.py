"""Second domain: every retriever on QASPER research papers with human-written questions.

Zero-shot transfer: all settings come from GovReport (structure weights tuned on GovReport
dev, models fine-tuned on GovReport train). Nothing is tuned on QASPER. Document scope.

uv run python scripts/run_crossdomain_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from structuralrag.evaluation.evaluator import RunResult, evaluate
from structuralrag.evaluation.statistical_tests import paired_test
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.reranker import RerankRetriever
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_crossdomain_eval")
METRICS = ("recall@1", "recall@5", "mrr@10", "ndcg@10")


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-rerank", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ft, rr = cfg["finetune"], cfg["finetune"]["reranker"]

    corpus = SectionCorpus(load_documents(DATA_DIR / "qasper_documents.jsonl"))
    queries = load_queries(DATA_DIR / "qasper_queries.jsonl")
    tuned = json.loads((REPORTS_DIR / "finetune_results.json").read_text())["structure_params"]
    encoders = {"base": (cfg["retrieval"]["embedding_model"], tuned["base"])}
    for label, key in (("fine-tuned ICT", "output"), ("fine-tuned ICT+LLM", "output_mixed")):
        path = PROJECT_ROOT / ft["embedding"].get(key, "")
        if key in ft["embedding"] and path.exists():
            encoders[label] = (str(path), tuned.get(label.replace("fine-tuned ", "finetuned_"), tuned["finetuned"]))

    systems: dict[str, Retriever] = {}
    suites = {}
    for label, (model, params) in encoders.items():
        suite = RetrieverSuite(corpus, model, CACHE_DIR, StructureParams(**params))
        suites[label] = suite
        if label == "base":
            systems["bm25"] = suite.bm25()
        systems[f"dense ({label})"] = suite.dense()
        systems[f"hybrid ({label})"] = suite.hybrid()
        systems[f"structure-aware ({label})"] = suite.structure_aware()

    if not args.skip_rerank:
        first = suites["base"].structure_aware()
        common = {
            "top_n": rr["rerank_top"],
            "max_length": rr["inference_max_length"],
            "window_words": rr["window_words"],
            "window_stride": rr["window_stride"],
        }
        systems["+ rerank MaxP (zero-shot)"] = RerankRetriever(first, rr["base_model"], label="zs", **common)
        for label, key in (("ICT", "output"), ("ICT+LLM", "output_mixed")):
            path = PROJECT_ROOT / rr.get(key, "")
            if key in rr and path.exists():
                systems[f"+ rerank MaxP (fine-tuned {label})"] = RerankRetriever(
                    first, str(path), label=label, **common
                )

    runs: dict[str, RunResult] = {}
    frames = []
    for name, r in systems.items():
        runs[name] = evaluate(r, queries, scope="document", latency_sample=20)
        log.info("%s: MRR@10 %.3f", name, runs[name].summary()["mrr@10"])
        frames.append(pd.DataFrame(runs[name].per_query).assign(system=name))

    rows = [
        {"system": n, **{m: r.summary()[m] for m in METRICS}, "p50_ms": r.latency_ms["p50_ms"]} for n, r in runs.items()
    ]
    base_sa = "structure-aware (base)"
    sig = []
    for name in runs:
        for ref in ("bm25", "dense (base)", base_sa):
            if name != ref and not (ref == base_sa and name in ("bm25", "dense (base)")):
                for m in ("mrr@10",):
                    sig.append(
                        {
                            "system": name,
                            "vs": ref,
                            "metric": m,
                            **paired_test(runs[name].metric_array(m), runs[ref].metric_array(m)),
                        }
                    )
    pq = pd.concat(frames)
    depth = pq.groupby(["system", "target_level"])["mrr@10"].mean().unstack().round(3)
    stats = json.loads((DATA_DIR / "qasper_stats.json").read_text())
    out = {
        "dataset": "allenai/qasper (test)",
        "stats": stats,
        "results": rows,
        "significance": sig,
        "mrr_by_target_level": depth.reset_index().to_dict(orient="records"),
    }
    write_json(REPORTS_DIR / "crossdomain_results.json", out)
    pq.to_csv(REPORTS_DIR / "crossdomain_per_query.csv", index=False)
    depth_rows = [{"system": i, **{f"depth {c}": v for c, v in r.items()}} for i, r in depth.iterrows()]
    md = [
        "# Second domain: QASPER research papers\n",
        f"{stats['documents']} NLP papers, {stats['sections']} sections, {stats['queries']} human-written questions "
        "(gold = sections holding the annotator's evidence). Zero-shot transfer: settings and fine-tuned models "
        "come from GovReport; nothing is tuned on QASPER. Document scope.\n",
        fmt_table(rows, ["system", *METRICS, "p50_ms"]),
        "\n## Paired tests (MRR@10)\n",
        fmt_table(sig, ["system", "vs", "mean_diff", "p_permutation", "p_wilcoxon"]),
        "\n## MRR@10 by gold-section depth\n",
        fmt_table(depth_rows, list(depth_rows[0])),
    ]
    (REPORTS_DIR / "crossdomain_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

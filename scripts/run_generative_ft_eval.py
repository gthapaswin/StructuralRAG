"""Zero-shot vs LoRA fine-tuned generative retrieval (same small model, MLX).

Same test questions as scripts/run_generative_eval.py (the fixed 300-query ICT sample and
all LLM-written questions), same prompt (ToC + snippets), document scope. The fine-tuned
model is also fused with structure-aware ranking. The earlier qwen3:8b zero-shot numbers
are copied in from reports/generative_results.json for reference.

uv run --group finetune python scripts/run_generative_ft_eval.py
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
from structuralrag.llm.mlx_client import MLXClient
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.generative import GenerativeFusionRetriever, GenerativeRetriever
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_generative_ft_eval")
METRICS = ("recall@1", "recall@5", "mrr@10", "ndcg@10")


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return out


def latency(gen: GenerativeRetriever, queries, n: int = 20) -> dict[str, float]:
    """Uncached latency: a fresh client without the response cache, model already loaded."""
    c = gen.client
    fresh = MLXClient(c.model_path, c.adapter_path, cache_path=None, label=c.model)
    fresh._loaded = c._load()
    g = GenerativeRetriever(gen.corpus, fresh, gen.max_ids, gen.snippets)
    times = []
    for q in queries[:n]:
        t0 = time.perf_counter()
        g.predict_ids(q.question, q.document_id)
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    llm, g = cfg["llm"], cfg["finetune"]["llm"]
    adapter = str(PROJECT_ROOT / g["output"])

    params = StructureParams(**json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"])
    corpus = SectionCorpus(load_documents(DATA_DIR / "test_documents.jsonl"))
    suite = RetrieverSuite(corpus, cfg["retrieval"]["embedding_model"], CACHE_DIR, params)
    cache = CACHE_DIR / "mlx_cache.sqlite"
    zs_client = MLXClient(g["base_model"], None, cache, label="qwen3-1.7b zero-shot")
    ft_client = MLXClient(g["base_model"], adapter, cache, label="qwen3-1.7b fine-tuned")
    zs = GenerativeRetriever(corpus, zs_client, llm["max_ids"], snippets=True)
    ft = GenerativeRetriever(corpus, ft_client, llm["max_ids"], snippets=True)
    systems: dict[str, Retriever] = {
        "structure-aware": suite.structure_aware(),
        "generative 1.7B zero-shot": zs,
        "generative 1.7B fine-tuned": ft,
        "generative 1.7B fine-tuned + structure": GenerativeFusionRetriever(ft, suite.structure_aware()),
    }
    pairs = [
        ("generative 1.7B fine-tuned", "generative 1.7B zero-shot"),
        ("generative 1.7B fine-tuned", "structure-aware"),
        ("generative 1.7B fine-tuned + structure", "structure-aware"),
    ]

    ict = load_queries(DATA_DIR / "test_queries.jsonl")
    ict = random.Random(cfg["seed"]).sample(ict, min(llm["generative_ict_sample"], len(ict)))
    query_sets = {"ict": ict, "llm": load_queries(DATA_DIR / "test_queries_llm.jsonl")}
    prev = json.loads((REPORTS_DIR / "generative_results.json").read_text())

    out: dict = {"base_model": g["base_model"], "adapter": g["output"], "training": g, "query_sets": {}}
    md = [
        "# Fine-tuned generative retrieval\n",
        f"`{g['base_model']}` with a LoRA adapter trained on GovReport train-split questions (MLX), "
        "vs the same model zero-shot. Prompt: table of contents with snippets. Document scope. "
        "No constrained decoding with MLX: invalid ids are dropped (see `valid_output`).\n",
    ]
    frames = []
    for qs_name, qs in query_sets.items():
        runs: dict[str, RunResult] = {}
        for name, r in systems.items():
            t0 = time.perf_counter()
            runs[name] = evaluate(r, qs, scope="document", batch_size=16, latency_sample=0)
            log.info(
                "%s / %s: MRR %.3f (%.0fs)", qs_name, name, runs[name].summary()["mrr@10"], time.perf_counter() - t0
            )
            frames.append(pd.DataFrame(runs[name].per_query).assign(system=name, query_set=qs_name))
        rows = []
        for name, r in runs.items():
            n_ret = r.metric_array("n_returned")
            rows.append(
                {
                    "system": name,
                    **{m: r.summary()[m] for m in METRICS},
                    "valid_output": float((n_ret > 0).mean()),
                    "mean_returned": float(n_ret.mean()),
                }
            )
        ref = {r["method"]: r for r in prev["query_sets"][qs_name]["results"]}
        for key, label in (
            ("generative_toc_snippets", "generative qwen3:8b zero-shot (earlier run)"),
            ("generative+structure", "generative qwen3:8b + structure (earlier run)"),
        ):
            if key in ref:
                rows.append(
                    {
                        "system": label,
                        **{m: ref[key][m] for m in METRICS},
                        "valid_output": 1.0,
                        "mean_returned": ref[key].get("mean_returned", float("nan")),
                    }
                )
        sig = []
        for a, b in pairs:
            for m in ("mrr@10", "recall@1"):
                sig.append(
                    {
                        "comparison": f"{a} vs {b}",
                        "metric": m,
                        **paired_test(runs[a].metric_array(m), runs[b].metric_array(m)),
                    }
                )
        out["query_sets"][qs_name] = {"n": len(qs), "results": rows, "significance": sig}
        md += [
            f"\n## {'ICT' if qs_name == 'ict' else 'LLM-written'} questions (n = {len(qs)})\n",
            fmt_table(rows, ["system", *METRICS, "valid_output", "mean_returned"]),
            "\n",
            fmt_table(sig, ["comparison", "metric", "mean_diff", "p_permutation", "p_wilcoxon"]),
        ]
    out["latency_ms"] = {"zero-shot": latency(zs, query_sets["llm"]), "fine-tuned": latency(ft, query_sets["llm"])}
    md.append(f"\nUncached latency per query: {out['latency_ms']}\n")
    info = PROJECT_ROOT / g["output"] / "training_log.json"
    if info.exists():
        out["training_log"] = json.loads(info.read_text())
    pd.concat(frames).to_csv(REPORTS_DIR / "generative_ft_per_query.csv", index=False)
    write_json(REPORTS_DIR / "generative_ft_results.json", out)
    (REPORTS_DIR / "generative_ft_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

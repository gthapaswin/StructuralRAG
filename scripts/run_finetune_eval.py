"""Compare base vs fine-tuned retrievers on the test set (document scope).

Embedding: dense, hybrid and structure-aware with the base encoder vs the fine-tuned one.
The structure-aware weights are re-tuned on dev for the fine-tuned encoder (same grid).
Reranker (if trained): structure-aware top-N reranked by the zero-shot vs fine-tuned
cross-encoder. Both question sets: ICT and LLM-written paraphrases.

uv run python scripts/run_finetune_eval.py [--skip-rerank]
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from structuralrag.evaluation.evaluator import RunResult, evaluate
from structuralrag.evaluation.statistical_tests import paired_test
from structuralrag.evaluation.tuning import tune_structure
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.reranker import RerankRetriever
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_finetune_eval")
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
    ft = cfg["finetune"]
    base_model = cfg["retrieval"]["embedding_model"]
    ft_model = str(PROJECT_ROOT / ft["embedding"]["output"])
    rr = ft["reranker"]

    # Structure weights: base from the main evaluation; fine-tuned re-tuned on dev, same grid.
    base_params = StructureParams(**json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"])
    dev_corpus = SectionCorpus(load_documents(DATA_DIR / "dev_documents.jsonl"))
    ft_params, _ = tune_structure(
        RetrieverSuite(dev_corpus, ft_model, CACHE_DIR), load_queries(DATA_DIR / "dev_queries.jsonl"), cfg["tuning"]
    )
    log.info("Fine-tuned encoder, dev-tuned structure params: %s", ft_params)

    corpus = SectionCorpus(load_documents(DATA_DIR / "test_documents.jsonl"))
    base = RetrieverSuite(corpus, base_model, CACHE_DIR, base_params)
    tuned = RetrieverSuite(corpus, ft_model, CACHE_DIR, ft_params)

    systems: dict[str, Retriever] = {
        "dense (base)": base.dense(),
        "dense (fine-tuned)": tuned.dense(),
        "hybrid (base)": base.hybrid(),
        "hybrid (fine-tuned)": tuned.hybrid(),
        "structure-aware (base)": base.structure_aware(),
        "structure-aware (fine-tuned)": tuned.structure_aware(),
    }
    pairs = [
        ("dense (fine-tuned)", "dense (base)"),
        ("hybrid (fine-tuned)", "hybrid (base)"),
        ("structure-aware (fine-tuned)", "structure-aware (base)"),
    ]
    rr_path = PROJECT_ROOT / rr["output"]
    if not args.skip_rerank and rr_path.exists():
        sa = tuned.structure_aware()
        systems["+ rerank (zero-shot)"] = RerankRetriever(
            sa, rr["base_model"], rr["rerank_top"], rr["inference_max_length"], "zs"
        )
        systems["+ rerank (fine-tuned)"] = RerankRetriever(
            sa, str(rr_path), rr["rerank_top"], rr["inference_max_length"], "ft"
        )
        pairs += [
            ("+ rerank (zero-shot)", "structure-aware (fine-tuned)"),
            ("+ rerank (fine-tuned)", "structure-aware (fine-tuned)"),
            ("+ rerank (fine-tuned)", "+ rerank (zero-shot)"),
        ]

    query_sets = {
        "ict": load_queries(DATA_DIR / "test_queries.jsonl"),
        "llm": load_queries(DATA_DIR / "test_queries_llm.jsonl"),
    }
    out: dict = {
        "base_model": base_model,
        "finetuned_model": ft["embedding"]["output"],
        "structure_params": {"base": base_params.__dict__, "finetuned": ft_params.__dict__},
        "query_sets": {},
    }
    md = [
        "# Fine-tuning results\n",
        f"Document scope (route a question to a section of a known report). Base encoder `{base_model}`; "
        f"fine-tuned on {ft['train_documents']} train-split reports (ICT questions, hard negatives from the "
        "same report). Structure-aware weights re-tuned on dev for each encoder: "
        f"base {base_params.__dict__}, fine-tuned {ft_params.__dict__}.\n",
    ]
    frames = []
    for qs_name, qs in query_sets.items():
        runs: dict[str, RunResult] = {}
        for name, r in systems.items():
            runs[name] = evaluate(r, qs, scope="document", latency_sample=20)
            df = pd.DataFrame(runs[name].per_query)
            df["system"], df["query_set"] = name, qs_name
            frames.append(df)
            log.info("%s / %s: MRR@10 %.3f", qs_name, name, runs[name].summary()["mrr@10"])
        rows = [
            {"system": n, **{m: r.summary()[m] for m in METRICS}, "p50_ms": r.latency_ms["p50_ms"]}
            for n, r in runs.items()
        ]
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
            fmt_table(rows, ["system", *METRICS, "p50_ms"]),
            "\n",
            fmt_table(sig, ["comparison", "metric", "mean_diff", "p_permutation", "p_wilcoxon"]),
        ]
    pd.concat(frames).to_csv(REPORTS_DIR / "finetune_per_query.csv", index=False)
    write_json(REPORTS_DIR / "finetune_results.json", out)
    (REPORTS_DIR / "finetune_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

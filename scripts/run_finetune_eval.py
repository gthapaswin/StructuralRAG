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
    ft, rr = cfg["finetune"], cfg["finetune"]["reranker"]
    base_model = cfg["retrieval"]["embedding_model"]

    # Encoders: base, fine-tuned on ICT, fine-tuned on ICT + LLM paraphrases (when trained).
    # Structure weights: base from the main evaluation; each fine-tuned encoder re-tuned on dev.
    base_params = StructureParams(**json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"])
    encoders: dict[str, tuple[str, StructureParams]] = {"base": (base_model, base_params)}
    dev_corpus = SectionCorpus(load_documents(DATA_DIR / "dev_documents.jsonl"))
    dev_q = load_queries(DATA_DIR / "dev_queries.jsonl")
    for label, key in (("fine-tuned", "output"), ("fine-tuned ICT+LLM", "output_mixed")):
        path = PROJECT_ROOT / ft["embedding"][key]
        if path.exists():
            params, _ = tune_structure(RetrieverSuite(dev_corpus, str(path), CACHE_DIR), dev_q, cfg["tuning"])
            encoders[label] = (str(path), params)
            log.info("%s: dev-tuned structure params %s", label, params)

    corpus = SectionCorpus(load_documents(DATA_DIR / "test_documents.jsonl"))
    suites = {k: RetrieverSuite(corpus, m, CACHE_DIR, p) for k, (m, p) in encoders.items()}
    systems: dict[str, Retriever] = {}
    pairs = []
    for kind in ("dense", "hybrid", "structure_aware"):
        name = kind.replace("_", "-")
        for label, suite in suites.items():
            systems[f"{name} ({label})"] = suite.get(kind)
            if label != "base":
                pairs.append((f"{name} ({label})", f"{name} (base)"))

    if not args.skip_rerank:
        sa = suites["fine-tuned"].structure_aware() if "fine-tuned" in suites else suites["base"].structure_aware()
        first = "structure-aware (fine-tuned)" if "fine-tuned" in suites else "structure-aware (base)"
        common = {"top_n": rr["rerank_top"], "max_length": rr["inference_max_length"]}
        maxp = {"window_words": rr["window_words"], "window_stride": rr["window_stride"]}
        rerankers = {"zero-shot": rr["base_model"]}
        for label, key in (("fine-tuned", "output"), ("fine-tuned ICT+LLM", "output_mixed")):
            if (PROJECT_ROOT / rr[key]).exists():
                rerankers[label] = str(PROJECT_ROOT / rr[key])
        for label, model in rerankers.items():
            systems[f"+ rerank ({label})"] = RerankRetriever(sa, model, label=label, **common)
            systems[f"+ rerank MaxP ({label})"] = RerankRetriever(sa, model, label=label, **common, **maxp)
            pairs += [
                (f"+ rerank MaxP ({label})", f"+ rerank ({label})"),
                (f"+ rerank MaxP ({label})", first),
            ]
            if label != "zero-shot":
                pairs.append((f"+ rerank MaxP ({label})", "+ rerank MaxP (zero-shot)"))

    query_sets = {
        "ict": load_queries(DATA_DIR / "test_queries.jsonl"),
        "llm": load_queries(DATA_DIR / "test_queries_llm.jsonl"),
    }
    params_out = {"base": base_params.__dict__}
    for label, (_, p) in encoders.items():
        if label != "base":
            params_out["finetuned" if label == "fine-tuned" else "finetuned_ICT+LLM"] = p.__dict__
    out: dict = {"base_model": base_model, "structure_params": params_out, "query_sets": {}}
    md = [
        "# Fine-tuning results\n",
        f"Document scope (route a question to a section of a known report). Base encoder `{base_model}`; "
        f"fine-tuned on {ft['train_documents']} train-split reports with hard negatives from the same report, "
        "on ICT questions only or on ICT + LLM-written paraphrases. Rerankers rescore the top "
        f"{rr['rerank_top']} of fine-tuned structure-aware; MaxP = best of overlapping {rr['window_words']}-word "
        f"windows. Structure weights re-tuned on dev per encoder: {params_out}.\n",
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

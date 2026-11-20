"""Evaluate grounded answer generation.

Configurations (same questions, same generator, same word budget):
  A  BM25 retrieval            + plain context
  B  structure-aware retrieval + plain context          (B vs A: effect of retrieval)
  C  structure-aware retrieval + hierarchical context   (C vs B: effect of structure in the context)
each with and without the claim check (effect of the checker).
Plus unanswerable questions (asked against a different report) to measure abstention.

Faithfulness is scored by an NLI model that plays no part in producing answers.
Needs Ollama with the generator (llm.model) and checker (answers.checker_model).

uv run python scripts/run_answer_eval.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time

import pandas as pd

from structuralrag.evaluation.answer_metrics import NLI_MODEL, NLIJudge, answer_metrics
from structuralrag.evaluation.statistical_tests import bootstrap_ci, paired_test
from structuralrag.generation.answerer import Answer, Answerer
from structuralrag.generation.context import Context, build_context
from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.factory import RetrieverSuite
from structuralrag.retrieval.structure_aware import StructureParams
from structuralrag.schema import Query
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries, write_json
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_answer_eval")
CONFIGS = {"A": ("bm25", "plain"), "B": ("structure_aware", "plain"), "C": ("structure_aware", "hierarchical")}
METRICS = ("abstained", "faithfulness", "citation_hit", "citation_hit_branch", "citation_precision", "gold_in_context")


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        cells = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols]
        out += "| " + " | ".join(cells) + " |\n"
    return out


def summarize(df: pd.DataFrame) -> dict:
    out = {"n": len(df)}
    for m in METRICS:
        vals = df[m].dropna().to_numpy()
        out[m] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{m}_ci"] = list(bootstrap_ci(vals)) if len(vals) > 1 else None
        out[f"{m}_n"] = len(vals)
    out["mean_sentences"] = float(df.loc[df.abstained == 0, "n_sentences"].mean())
    out["removed_per_answer"] = float(df["n_removed"].mean())
    return out


def paired(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict:
    """Paired test on questions where both configurations have a value."""
    j = df_a[["query_id", metric]].merge(df_b[["query_id", metric]], on="query_id").dropna()
    if len(j) < 5:
        return {"n_pairs": len(j)}
    return {"n_pairs": len(j), **paired_test(j[f"{metric}_x"].to_numpy(), j[f"{metric}_y"].to_numpy())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    a_cfg, llm = cfg["answers"], cfg["llm"]
    rng = random.Random(cfg["seed"])

    params = StructureParams(**json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"])
    corpus = SectionCorpus(load_documents(DATA_DIR / "test_documents.jsonl"))
    suite = RetrieverSuite(corpus, cfg["retrieval"]["embedding_model"], CACHE_DIR, params)
    cache = CACHE_DIR / "ollama_cache.sqlite"
    gen_client = OllamaClient(llm["model"], llm["host"], cache_path=cache, seed=cfg["seed"])
    chk_client = OllamaClient(a_cfg["checker_model"], llm["host"], cache_path=cache, seed=cfg["seed"])
    gen_client.check()
    chk_client.check()
    answerer = Answerer(gen_client, chk_client)
    judge = NLIJudge()

    pool = load_queries(DATA_DIR / "test_queries_llm.jsonl")
    questions = rng.sample(pool, a_cfg["questions"])
    doc_ids = sorted(corpus.docs)
    unanswerable = []
    for q in rng.sample(pool, a_cfg["unanswerable"]):
        other = rng.choice([d for d in doc_ids if d != q.document_id])
        unanswerable.append(q.model_copy(update={"query_id": q.query_id + "::neg", "document_id": other}))

    def contexts(qs: list[Query], method: str, mode: str) -> list[Context]:
        r = suite.get(method)
        return [
            build_context(
                corpus,
                [i for i, _ in r.rank(r.score(q.question), a_cfg["top_k"], q.document_id)],
                mode,
                a_cfg["context_words"],
            )
            for q in qs
        ]

    # Phase 1: all generator calls, then phase 2: all checker calls (avoids swapping models in memory).
    runs: dict[str, tuple[list[Query], list[Context], list[Answer]]] = {}
    jobs = {k: (questions, *v) for k, v in CONFIGS.items()}
    jobs["C_unanswerable"] = (unanswerable, *CONFIGS["C"])
    for name, (qs, method, mode) in jobs.items():
        t0 = time.perf_counter()
        ctxs = contexts(qs, method, mode)
        drafts = [answerer.draft(q.question, c) for q, c in zip(qs, ctxs, strict=True)]
        runs[name] = (qs, ctxs, drafts)
        log.info("drafted %s (%d) in %.0fs", name, len(qs), time.perf_counter() - t0)

    rows = []
    examples = []
    for name, (qs, ctxs, drafts) in runs.items():
        t0 = time.perf_counter()
        for q, ctx, draft in zip(qs, ctxs, drafts, strict=True):
            checked = answerer.check(copy.deepcopy(draft), ctx)
            answerable = not name.endswith("unanswerable")
            gold = q.target_section_ids[0] if answerable else None
            for variant, ans in (("no_check", draft), ("checked", checked)):
                m = answer_metrics(ans, ctx, gold, judge)
                rows.append({"config": name, "variant": variant, "query_id": q.query_id, **m})
            if name == "C" and len(examples) < 6 and (checked.removed or len(examples) < 3):
                examples.append({"question": q.question, "gold": gold, "answer": checked.to_dict()})
        log.info("checked + judged %s in %.0fs", name, time.perf_counter() - t0)

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "answer_per_query.csv", index=False)
    results = []
    for (name, variant), g in df.groupby(["config", "variant"], sort=False):
        results.append({"config": name, "variant": variant, **summarize(g)})

    def sel(name: str, variant: str) -> pd.DataFrame:
        return df[(df.config == name) & (df.variant == variant)]

    comparisons = []
    for label, a, b in [
        ("structure-aware vs BM25 retrieval (plain ctx)", ("B", "no_check"), ("A", "no_check")),
        ("hierarchical vs plain context", ("C", "no_check"), ("B", "no_check")),
        ("claim check vs none (config C)", ("C", "checked"), ("C", "no_check")),
        ("full pipeline vs BM25 baseline", ("C", "checked"), ("A", "no_check")),
    ]:
        for metric in ("faithfulness", "citation_hit", "abstained"):
            comparisons.append({"comparison": label, "metric": metric, **paired(sel(*a), sel(*b), metric)})

    out = {
        "generator": llm["model"],
        "checker": a_cfg["checker_model"],
        "faithfulness_judge": NLI_MODEL,
        "settings": a_cfg,
        "results": results,
        "comparisons": comparisons,
        "examples": examples,
    }
    write_json(REPORTS_DIR / "answer_results.json", out)

    names = {
        "A": "A: BM25 + plain",
        "B": "B: structure + plain",
        "C": "C: structure + hierarchical",
        "C_unanswerable": "C on unanswerable questions",
    }
    table = [{**r, "setup": names[r["config"]]} for r in results]
    md = [
        "# Grounded answer generation\n",
        f"Generator `{llm['model']}`, claim checker `{a_cfg['checker_model']}`, faithfulness judged by "
        f"`{NLI_MODEL}` (NLI, independent of both). {a_cfg['questions']} LLM-written questions, top-"
        f"{a_cfg['top_k']} sections, {a_cfg['context_words']}-word context budget; "
        f"{a_cfg['unanswerable']} unanswerable questions (asked against a different report).\n",
        fmt_table(
            table,
            [
                "setup",
                "variant",
                "abstained",
                "faithfulness",
                "citation_hit",
                "citation_hit_branch",
                "citation_precision",
                "gold_in_context",
                "mean_sentences",
                "removed_per_answer",
            ],
        ),
        "\n## Paired comparisons\n",
        fmt_table(comparisons, ["comparison", "metric", "n_pairs", "mean_diff", "p_permutation", "p_wilcoxon"]),
        "\n## Example answers (config C, checked)\n",
    ]
    for ex in examples:
        a = ex["answer"]
        md.append(f"**Q:** {ex['question']}  \n**Gold section:** {ex['gold']}  \n**A:** {a['text']}")
        for s in a["removed"]:
            md.append(f"  \n*Removed by claim check:* {s['text']} [{', '.join(s['citations'])}]")
        md.append("\n")
    (REPORTS_DIR / "answer_results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

"""Heading study: does the retrieval difference change the final answer?

For a fixed random sample of questions with reference answers (NQ short answers; ConditionalQA
span answers, yes/no answers excluded), each retrieval system (flat, structure-aware, summary-role
neutral) picks the top-3 sections of the question's document; the same LLM (qwen3:8b) writes a
cited answer from them (plain context, 900-word budget, no claim check). An answer is correct if
it contains a reference answer string (normalized). Abstentions count as wrong.

uv run python scripts/run_heading_answers.py [--n 200]
Output: reports/heading_study/answers.{md,json}; per-question CSV (gitignored)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re

import numpy as np
import pandas as pd

from structuralrag.analysis import experiment as ex
from structuralrag.analysis.informativeness import role_weights
from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test
from structuralrag.generation.answerer import Answerer
from structuralrag.generation.context import build_context
from structuralrag.ingestion.natural_questions import short_answers
from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries

log = logging.getLogger("run_heading_answers")
OUT = REPORTS_DIR / "heading_study"
TOP = 3


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def references(ds: str, queries) -> dict[str, list[str]]:
    if ds == "nq":
        ids = {q.document_id.removeprefix("nq_") for q in queries}
        sa = short_answers(sorted((PROJECT_ROOT / "data/raw/nq").glob("validation_*.parquet")), ids)
        return {q.query_id: sa.get(q.document_id.removeprefix("nq_"), []) for q in queries}
    raw = {}
    for split in ("train", "dev"):
        for q in json.loads((PROJECT_ROOT / f"data/raw/conditionalqa/{split}.json").read_text()):
            raw[q["id"]] = [norm(a[0]) for a in q["answers"] if norm(a[0]) not in {"yes", "no", ""}]
    return {q.query_id: raw.get(q.query_id.split("::")[-1], []) for q in queries}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config()
    llm = cfg["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    client.check()
    answerer = Answerer(client)
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    rows = []
    for ds, dfile, qfile in (
        ("nq", "nq_documents.jsonl", "nq_queries.jsonl"),
        ("conditionalqa", "conditionalqa_documents.jsonl", "conditionalqa_queries.jsonl"),
    ):
        corpus = SectionCorpus(load_documents(DATA_DIR / dfile))
        queries = load_queries(DATA_DIR / qfile)
        refs = references(ds, queries)
        pool = [q for q in queries if refs[q.query_id]]
        sample = random.Random(cfg["seed"]).sample(pool, min(args.n, len(pool)))
        comp = ex.compute(corpus, encoder, sample)
        rw = role_weights(corpus)
        from dataclasses import replace

        comp_role = replace(comp, specificity=[rw[corpus.doc_indices(q.document_id)] for q in sample])
        systems = {
            "flat": ex.flat_scores(comp),
            "structure-aware": ex.structure_scores(comp, 0.2),
            "role-neutral": ex.structure_scores(comp_role, 0.2, fallback="mean"),
        }
        for name, scores in systems.items():
            for k, (q, s, gold) in enumerate(zip(sample, scores, comp.gold, strict=True)):
                idx = corpus.doc_indices(q.document_id)
                order = np.argsort(-s, kind="stable")[:TOP]
                ans = answerer.draft(q.question, build_context(corpus, [int(idx[o]) for o in order], "plain"))
                text = norm(ans.text) if not ans.abstained else ""
                rows.append(
                    {
                        "dataset": ds,
                        "system": name,
                        "query_id": q.query_id,
                        "document_id": q.document_id,
                        "gold_in_top3": float(gold[order].any()),
                        "correct": float(any(r and r in text for r in refs[q.query_id])),
                        "abstained": float(ans.abstained),
                    }
                )
                if k % 50 == 0:
                    log.info("%s / %s: %d/%d", ds, name, k, len(sample))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "answers_per_query.csv", index=False)
    res, md = (
        {},
        [
            "# Heading study: effect on final answers\n",
            f"Fixed random sample per dataset (questions with a reference answer; ConditionalQA yes/no answers "
            f"excluded). Top-{TOP} sections → qwen3:8b cited answer (plain context, 900 words, no claim check). "
            "Correct = the answer contains a reference answer string. Document-clustered tests vs structure-aware.\n",
            "| dataset | system | n | gold section in top-3 | answer correct | abstained |",
            "|---|---|---|---|---|---|",
        ],
    )
    tests = []
    for ds, g in df.groupby("dataset", sort=False):
        w = g.pivot_table(index="query_id", columns="system", values=["correct", "gold_in_top3", "abstained"])
        docs = g.drop_duplicates("query_id").set_index("query_id").loc[w.index].document_id.values
        for name in ("flat", "structure-aware", "role-neutral"):
            md.append(
                f"| {ds} | {name} | {len(w)} | {w[('gold_in_top3', name)].mean():.3f} | "
                f"{w[('correct', name)].mean():.3f} | {w[('abstained', name)].mean():.3f} |"
            )
            res[f"{ds}/{name}"] = {m: float(w[(m, name)].mean()) for m in ("gold_in_top3", "correct", "abstained")}
        for a, b in (("role-neutral", "structure-aware"), ("flat", "structure-aware"), ("role-neutral", "flat")):
            d = w[("correct", a)].values - w[("correct", b)].values
            r = clustered_paired_test(w[("correct", a)].values, w[("correct", b)].values, docs)
            tests.append(
                {
                    "dataset": ds,
                    "comparison": f"{a} vs {b}",
                    "diff": float(d.mean()),
                    "ci": bootstrap_ci(d),
                    "p_cluster": r["p_cluster"],
                }
            )
    md += ["\n| dataset | comparison (answer correct) | difference | 95% CI | p (clustered) |", "|---|---|---|---|---|"]
    md += [
        f"| {t['dataset']} | {t['comparison']} | {t['diff']:+.3f} | [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | "
        f"{t['p_cluster']:.3f} |"
        for t in tests
    ]
    (OUT / "answers.json").write_text(json.dumps({"results": res, "tests": tests}, indent=2, default=float))
    (OUT / "answers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

"""Heading study: is the NQ loss partly an evaluation artifact?

NQ marks one long answer per question (most often in the lead section). When structure-aware
retrieval ranks a body section first (e.g. "Release" for "when did the movie come out"), that
section may also contain the answer. Using NQ's short answers (exact answer strings), this checks
how often each method's top-1 section contains a short answer, in addition to whether it is the
annotated gold section.

uv run python scripts/nq_answer_check.py
Output: reports/heading_study/nq_answer_check.json (+ printed table)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from structuralrag.analysis import experiment as ex
from structuralrag.evaluation.statistical_tests import paired_test
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT, load_documents, load_queries


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def short_answers(files: list[Path], wanted: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=["id", "document", "annotations"]).to_pylist():
                if row["id"] not in wanted:
                    continue
                toks, html = row["document"]["tokens"]["token"], row["document"]["tokens"]["is_html"]
                answers = set()
                for sa in row["annotations"]["short_answers"]:
                    for s, e in zip(sa["start_token"], sa["end_token"], strict=True):
                        text = " ".join(t for t, h in zip(toks[s:e], html[s:e], strict=True) if not h)
                        if norm(text):
                            answers.add(norm(text))
                out[row["id"]] = sorted(answers)
    return out


def main() -> None:
    cfg = load_config()
    alpha = json.loads((REPORTS_DIR / "heading_study" / "settings.json").read_text())["base_params"]["alpha"]
    corpus = SectionCorpus(load_documents(DATA_DIR / "nq_documents.jsonl"))
    queries = load_queries(DATA_DIR / "nq_queries.jsonl")
    ids = {q.document_id.removeprefix("nq_") for q in queries}
    answers = short_answers(sorted((PROJECT_ROOT / "data/raw/nq").glob("validation_*.parquet")), ids)
    qs = [q for q in queries if answers.get(q.document_id.removeprefix("nq_"))]
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    comp = ex.compute(corpus, encoder, qs)
    systems = {"flat": ex.flat_scores(comp), "structure-aware": ex.structure_scores(comp, alpha)}
    res: dict = {"questions_with_short_answer": len(qs), "of": len(queries)}
    hit, gold = {}, {}
    for name, scores in systems.items():
        h, g = [], []
        for q, s, gm in zip(qs, scores, comp.gold, strict=True):
            idx = corpus.doc_indices(q.document_id)
            top = int(np.argmax(s))
            text = norm(corpus.sections[idx[top]].text)
            h.append(float(any(a in text for a in answers[q.document_id.removeprefix("nq_")])))
            g.append(float(gm[top]))
        hit[name], gold[name] = np.array(h), np.array(g)
        res[name] = {"top1_is_gold": float(np.mean(g)), "top1_contains_short_answer": float(np.mean(h))}
    res["gold_vs_answer_test"] = {
        "top1_is_gold (SA - flat)": paired_test(gold["structure-aware"], gold["flat"]),
        "top1_contains_answer (SA - flat)": paired_test(hit["structure-aware"], hit["flat"]),
    }
    lead = np.array([q.metadata.get("lead", False) for q in qs])
    res["lead_gold_only"] = {
        n: {"top1_is_gold": float(gold[n][lead].mean()), "top1_contains_short_answer": float(hit[n][lead].mean())}
        for n in systems
    }
    out = REPORTS_DIR / "heading_study" / "nq_answer_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

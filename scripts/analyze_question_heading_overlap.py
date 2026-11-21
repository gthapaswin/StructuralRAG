"""Heading study: do headings help only when questions use the heading's words?

Tests the explanation offered (untested) for the held-out ConditionalQA result: its questions
describe personal situations instead of using the vocabulary of section headings, so headings have
little to match. For each question and its target section (the first gold section, as in the main
analyses), two measures of question-heading match:

- lexical: share of the target heading's content words (stop words removed) that occur in the question
- semantic: is the target's heading path the closest heading path in the document to the question
  (cosine, bge-small)? ("heading points to the target")

Reported per question set: how often questions match their target heading, and whether the
structure-aware gain over flat rises with the match (Spearman; and the gain among questions with
and without a match), also within unique target headings only (separating match from specificity).

uv run python scripts/analyze_question_heading_overlap.py
Output: reports/heading_study/question_heading_overlap.{md,json}
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from structuralrag.analysis.informativeness import heading_specificity
from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test
from structuralrag.preprocessing.text_cleaner import tokenize
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries

OUT = REPORTS_DIR / "heading_study"
SETS = [
    ("GovReport (ICT)", "govreport", "ict", "test_documents.jsonl", "test_queries.jsonl"),
    ("GovReport (LLM-written)", "govreport", "llm", "test_documents.jsonl", "test_queries_llm.jsonl"),
    ("QASPER (human)", "qasper", "human", "qasper_documents.jsonl", "qasper_queries.jsonl"),
    ("NQ (human)", "nq", "human", "nq_documents.jsonl", "nq_queries.jsonl"),
    ("ConditionalQA (human)", "conditionalqa", "human", "conditionalqa_documents.jsonl", "conditionalqa_queries.jsonl"),
]


def content_words(t: str) -> set[str]:
    return {w for w in tokenize(t) if w not in ENGLISH_STOP_WORDS and len(w) > 1}


def main() -> None:
    cfg = load_config()
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    pq = pd.read_csv(OUT / "per_query.csv.gz")
    pq = pq[(pq.variant == "real") & pq.system.isin(["flat", "sa_a0.2"])]
    res, corpora = {}, {}
    for name, ds, qset, dfile, qfile in SETS:
        if dfile not in corpora:
            c = SectionCorpus(load_documents(DATA_DIR / dfile))
            corpora[dfile] = (c, encoder.encode_passages(c.path_texts()), heading_specificity(c))
        corpus, path_emb, spec = corpora[dfile]
        qs = load_queries(DATA_DIR / qfile)
        w = pq[(pq.dataset == ds) & (pq.query_set == qset)].pivot_table(index="query_id", columns="system", values="rr")
        meta = pq[(pq.dataset == ds) & (pq.query_set == qset)].drop_duplicates("query_id").set_index("query_id")
        q_emb = encoder.encode_queries([q.question for q in qs])
        rows = []
        for q, qe in zip(qs, q_emb, strict=True):
            if q.query_id not in w.index:
                continue
            t = corpus.uid_to_idx[meta.loc[q.query_id, "gold_uid"]]
            title = content_words(corpus.sections[t].title)
            qw = content_words(q.question)
            idx = corpus.doc_indices(q.document_id)
            sims = path_emb[idx] @ qe
            rows.append(
                {
                    "query_id": q.query_id,
                    "document_id": q.document_id,
                    "lexical": len(title & qw) / len(title) if title else 0.0,
                    "points": float(idx[int(np.argmax(sims))] == t),
                    "unique": spec[t] >= 0.9,
                    "gain": w.loc[q.query_id, "sa_a0.2"] - w.loc[q.query_id, "flat"],
                }
            )
        df = pd.DataFrame(rows)
        rho, p = stats.spearmanr(df.lexical, df.gain)
        any_lex = df.lexical > 0

        def split(mask, df=df):
            a, b = df.gain[mask].values, df.gain[~mask].values
            return {"with": float(a.mean()), "without": float(b.mean()), "n_with": int(mask.sum())}

        u = df[df.unique]
        rho_u, p_u = stats.spearmanr(u.lexical, u.gain)
        sa_vs_flat = clustered_paired_test(df.gain.values, np.zeros(len(df)), df.document_id.values)
        res[name] = {
            "n": len(df),
            "share_any_heading_word": float(any_lex.mean()),
            "mean_lexical": float(df.lexical.mean()),
            "share_heading_points_to_target": float(df.points.mean()),
            "spearman_lexical_gain": float(rho),
            "p": float(p),
            "spearman_lexical_gain_unique_targets": float(rho_u),
            "p_unique": float(p_u),
            "gain_by_lexical": split(any_lex),
            "gain_by_points": split(df.points.astype(bool)),
            "gain_unique_by_lexical": split(u.lexical > 0, u),
            "mean_gain": float(df.gain.mean()),
            "mean_gain_ci": bootstrap_ci(df.gain.values),
            "p_gain": sa_vs_flat["p_cluster"],
        }
    md = [
        "# Heading study: do questions use their target heading's words?\n",
        "Target = first gold section. 'Heading word in question' = at least one content word of the target's "
        "own heading occurs in the question. 'Heading points to target' = the target's heading path is the "
        "closest in its document to the question (bge-small cosine). Gain = structure-aware minus flat "
        "(MRR@10, alpha 0.2).\n",
        "| set | n | heading word in question | heading points to target | SA gain overall | "
        "gain: with / without heading word | gain: points / does not | Spearman(lexical, gain) | "
        "same, unique targets only |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        gl, gp = r["gain_by_lexical"], r["gain_by_points"]
        md.append(
            f"| {name} | {r['n']} | {r['share_any_heading_word']:.0%} | {r['share_heading_points_to_target']:.0%} | "
            f"{r['mean_gain']:+.3f} | {gl['with']:+.3f} / {gl['without']:+.3f} | {gp['with']:+.3f} / "
            f"{gp['without']:+.3f} | {r['spearman_lexical_gain']:.2f} (p = {r['p']:.2g}) | "
            f"{r['spearman_lexical_gain_unique_targets']:.2f} (p = {r['p_unique']:.2g}) |"
        )
    (OUT / "question_heading_overlap.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / "question_heading_overlap.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Heading study: does a cross-encoder reranker show the same summary-section loss?

The cross-encoder (bge-reranker-base, zero-shot: not fine-tuned here) rescores the top-20 sections
of flat retrieval within the question's document, reading the question and a section together.
Each section is read in MaxP windows (200 words, stride 100), scored by its best window, as in the
project's reranker. Two inputs per section:

- text only (no headings)
- heading path + text (as in the project's reranker)

and two role-neutral combinations that need no extra scoring: summary-role sections are read as
text only, all others with their heading path (summary role from the heading list, or from the
label-free position rule: first and last top-level section).

Reported: MRR@10 per input, the heading gain (path + text minus text only) for summary-only targets
vs other targets, and document-clustered tests.

uv run python scripts/run_heading_reranker.py [--limit N]
Output: reports/heading_study/reranker_headings.{md,json}
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from structuralrag.analysis import experiment as ex
from structuralrag.analysis.informativeness import position_weights, role_weights
from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.retrieval.reranker import load_cross_encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries

log = logging.getLogger("run_heading_reranker")
OUT = REPORTS_DIR / "heading_study"
TOP_N, WIN, STRIDE = 20, 200, 100
SETS = [
    ("GovReport (ICT)", "test_documents.jsonl", "test_queries.jsonl"),
    ("GovReport (LLM-written)", "test_documents.jsonl", "test_queries_llm.jsonl"),
    ("QASPER (human)", "qasper_documents.jsonl", "qasper_queries.jsonl"),
    ("NQ (human)", "nq_documents.jsonl", "nq_queries.jsonl"),
    ("ConditionalQA (human)", "conditionalqa_documents.jsonl", "conditionalqa_queries.jsonl"),
]


def windows(text: str) -> list[str]:
    words = text.split()
    if len(words) <= WIN:
        return [text]
    return [" ".join(words[s : s + WIN]) for s in range(0, len(words) - WIN + STRIDE, STRIDE)]


def rr_at(scores: np.ndarray, gold: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    hits = np.flatnonzero(gold[order])
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per set (for a timing check)")
    args = ap.parse_args()
    cfg = load_config()
    rcfg = cfg["finetune"]["reranker"] if "finetune" in cfg else {}
    model_name = rcfg.get("base_model", "BAAI/bge-reranker-base")
    model = load_cross_encoder(model_name, rcfg.get("inference_max_length", 512))
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    res, corpora = {}, {}
    for name, dfile, qfile in SETS:
        t0 = time.time()
        if dfile not in corpora:
            c = SectionCorpus(load_documents(DATA_DIR / dfile))
            corpora[dfile] = (c, role_weights(c), position_weights(c, last=True))
        corpus, w_list, w_pos = corpora[dfile]
        qs = load_queries(DATA_DIR / qfile)[: args.limit]
        comp = ex.compute(corpus, encoder, qs)
        flat = ex.flat_scores(comp)
        pairs, owner = [], []  # owner: (query k, local section j, with_path)
        cands = []
        for k, (q, s) in enumerate(zip(qs, flat, strict=True)):
            idx = corpus.doc_indices(q.document_id)
            cand = np.argsort(-s, kind="stable")[:TOP_N]
            cands.append(cand)
            for j in cand:
                sec = corpus.sections[idx[j]]
                for wtext in windows(sec.text):
                    pairs += [(q.question, wtext), (q.question, f"{sec.path_str}\n{wtext}")]
                    owner += [(k, j, False), (k, j, True)]
        log.info("%s: %d questions, %d pairs", name, len(qs), len(pairs))
        raw = np.asarray(model.predict(pairs, batch_size=64, show_progress_bar=False), dtype=float)
        best = {}
        for (k, j, wp), sc in zip(owner, raw, strict=True):
            best[(k, j, wp)] = max(best.get((k, j, wp), -np.inf), sc)
        rr = {"flat": [], "text only": [], "path + text": [], "role-neutral (list)": [], "role-neutral (position)": []}
        summary_only = []
        for k, q in enumerate(qs):
            idx = corpus.doc_indices(q.document_id)
            gold = comp.gold[k]
            rr["flat"].append(rr_at(flat[k], gold))
            for label, choose in (
                ("text only", lambda j: False),
                ("path + text", lambda j: True),
                ("role-neutral (list)", lambda j, idx=idx, w=w_list: bool(w[idx[j]])),
                ("role-neutral (position)", lambda j, idx=idx, w=w_pos: bool(w[idx[j]])),
            ):
                s = np.full(len(idx), -np.inf)
                for j in cands[k]:
                    s[j] = best[(k, j, choose(j))]
                rr[label].append(rr_at(s, gold))
            summary_only.append(all(w_list[corpus.uid_to_idx[u]] == 0 for u in q.target_uids))
        rr = {k: np.array(v) for k, v in rr.items()}
        summary_only = np.array(summary_only)
        docs = np.array([q.document_id for q in qs])
        gain = rr["path + text"] - rr["text only"]

        def test(a, b, m=None, docs=docs):
            m = np.ones(len(a), bool) if m is None else m
            t = clustered_paired_test(a[m], b[m], docs[m])
            return {"diff": float(np.mean(a[m] - b[m])), "ci": bootstrap_ci(a[m] - b[m]), "p": t["p_cluster"]}

        res[name] = {
            "n": len(qs),
            "mrr": {k: float(v.mean()) for k, v in rr.items()},
            "n_summary_only": int(summary_only.sum()),
            "heading_gain_summary_targets": float(gain[summary_only].mean()) if summary_only.any() else None,
            "heading_gain_other_targets": float(gain[~summary_only].mean()),
            "tests": {
                "path + text vs text only": test(rr["path + text"], rr["text only"]),
                "role-neutral (list) vs path + text": test(rr["role-neutral (list)"], rr["path + text"]),
                "role-neutral (position) vs path + text": test(rr["role-neutral (position)"], rr["path + text"]),
                "path + text vs text only, summary targets": test(rr["path + text"], rr["text only"], summary_only)
                if summary_only.sum() > 1
                else None,
            },
        }
        log.info("%s: %s (%.0fs)", name, {k: round(v, 3) for k, v in res[name]["mrr"].items()}, time.time() - t0)
    md = [
        "# Heading study: the summary-section loss in a cross-encoder reranker\n",
        f"{model_name} (zero-shot) rescores the top-{TOP_N} sections of flat retrieval within the document; MaxP "
        f"windows of {WIN} words (stride {STRIDE}). Role-neutral = summary-role sections read without their heading "
        "path (heading list, or label-free position rule). MRR@10; document-clustered tests.\n",
        "| set | n | flat (first stage) | rerank: text only | rerank: path + text | role-neutral (list) | "
        "role-neutral (position) | heading gain: summary-only targets | heading gain: other targets |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        m = r["mrr"]
        hs = r["heading_gain_summary_targets"]
        md.append(
            f"| {name} | {r['n']} | {m['flat']:.3f} | {m['text only']:.3f} | {m['path + text']:.3f} | "
            f"{m['role-neutral (list)']:.3f} | {m['role-neutral (position)']:.3f} | "
            f"{'' if hs is None else f'{hs:+.3f}'} (n = {r['n_summary_only']}) | "
            f"{r['heading_gain_other_targets']:+.3f} |"
        )
    md += ["\n| set | comparison | difference | 95% CI | p |", "|---|---|---|---|---|"]
    for name, r in res.items():
        for key, t in r["tests"].items():
            if t:
                md.append(
                    f"| {name} | {key} | {t['diff']:+.3f} | [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | {t['p']:.3f} |"
                )
    suffix = f"_limit{args.limit}" if args.limit else ""
    (OUT / f"reranker_headings{suffix}.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / f"reranker_headings{suffix}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

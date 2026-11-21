"""Heading study: real examples for the paper (where headings hurt and where they help).

For each question set it picks questions whose target section has a generic heading and where
structure-aware ranks it lower than flat retrieval, plus questions with a specific target heading
where structure-aware wins, and shows the question, the target heading and each method's top-1.

uv run python scripts/heading_examples.py
Output: reports/heading_study/examples.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from structuralrag.analysis import experiment as ex
from structuralrag.analysis.informativeness import heading_specificity
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries

DIR = REPORTS_DIR / "heading_study"
SETS = [
    ("govreport", "ict", "test_documents.jsonl", "test_queries.jsonl", "GovReport (ICT)"),
    ("govreport", "llm", "test_documents.jsonl", "test_queries_llm.jsonl", "GovReport (LLM-written)"),
    ("qasper", "human", "qasper_documents.jsonl", "qasper_queries.jsonl", "QASPER"),
    ("nq", "human", "nq_documents.jsonl", "nq_queries.jsonl", "Natural Questions"),
]
PER_KIND = 3


def main() -> None:
    cfg = load_config()
    alpha = json.loads((DIR / "settings.json").read_text())["base_params"]["alpha"]
    pq = pd.read_csv(DIR / "per_query.csv.gz")
    pq = pq[pq.variant == "real"]
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    md = [
        "# Heading study: examples\n",
        f"Within-document retrieval. SA = structure-aware (alpha = {alpha}); flat = BM25 + dense over text only. "
        "w = heading specificity (0 = heading shared by every document, 1 = unique). Questions are shown as "
        "written; ICT questions are sentences removed from the target section.\n",
    ]
    rng = np.random.default_rng(13)
    for ds, qs_name, dfile, qfile, label in SETS:
        sub = pq[(pq.dataset == ds) & (pq.query_set == qs_name)]
        w = sub.pivot_table(index="query_id", columns="system", values="rr")
        corpus = SectionCorpus(load_documents(DATA_DIR / dfile))
        spec = dict(zip([x.uid for x in corpus.sections], heading_specificity(corpus), strict=True))
        queries = {q.query_id: q for q in load_queries(DATA_DIR / qfile)}
        gold_w = pd.Series({qid: spec[sorted(queries[qid].target_uids)[0]] for qid in w.index})
        diff = w[f"sa_a{alpha}"] - w["flat"]
        kinds = {
            "Generic target heading, structure-aware loses": w.index[(gold_w < 0.5) & (diff < 0)],
            "Specific target heading, structure-aware wins": w.index[(gold_w >= 0.9) & (diff > 0)],
        }
        md.append(f"\n## {label}\n")
        for kind, ids in kinds.items():
            if len(ids) == 0:
                continue
            pick = list(rng.choice(np.asarray(ids), size=min(PER_KIND, len(ids)), replace=False))
            qs = [queries[i] for i in pick]
            comp = ex.compute(corpus, encoder, qs)
            sa = ex.structure_scores(comp, alpha)
            flat = ex.flat_scores(comp)
            md.append(f"\n### {kind}\n")
            for q, s_sa, s_flat in zip(qs, sa, flat, strict=True):
                idx = corpus.doc_indices(q.document_id)
                gold = corpus.sections[corpus.uid_to_idx[sorted(q.target_uids)[0]]]

                def top(scores, idx=idx, corpus=corpus, spec=spec):
                    sec = corpus.sections[idx[int(np.argmax(scores))]]
                    return f"{sec.path_str} (w = {spec[sec.uid]:.2f})"

                md += [
                    f"- **Q:** {q.question}",
                    f"  - Target: {gold.path_str} (w = {spec[gold.uid]:.2f})",
                    f"  - Flat top-1: {top(s_flat)}",
                    f"  - Structure-aware top-1: {top(s_sa)}",
                ]
    (DIR / "examples.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

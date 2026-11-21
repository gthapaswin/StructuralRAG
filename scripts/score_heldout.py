"""Score the pre-registered ConditionalQA predictions (reports/heading_study/heldout_predictions.md).

uv run python scripts/score_heldout.py
Output: reports/heading_study/heldout_results.md (+ .json)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from structuralrag.analysis.headings import is_summary_heading
from structuralrag.analysis.informativeness import heading_specificity
from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.utils.config import DATA_DIR, REPORTS_DIR
from structuralrag.utils.io import load_documents, load_queries

DIR = REPORTS_DIR / "heading_study"


def main() -> None:
    pq = pd.read_csv(DIR / "per_query.csv.gz")
    pq = pq[pq.dataset == "conditionalqa"]
    w = pq.pivot_table(index="query_id", columns=["variant", "system"], values="rr")
    corpus = SectionCorpus(load_documents(DATA_DIR / "conditionalqa_documents.jsonl"))
    spec = dict(zip([s.uid for s in corpus.sections], heading_specificity(corpus), strict=True))
    title = {s.uid: s.title for s in corpus.sections}
    queries = {q.query_id: q for q in load_queries(DATA_DIR / "conditionalqa_queries.jsonl")}
    qs = [queries[i] for i in w.index]
    docs = np.array([q.document_id for q in qs])
    w_max = np.array([max(spec[u] for u in q.target_uids) for q in qs])
    summary = np.array([all(is_summary_heading(title[u]) for u in q.target_uids) for q in qs])

    flat = w[("real", "flat")].values
    sa = w[("real", "sa_a0.2")].values
    placebo = np.mean([w[(f"placebo_s{s}", "sa_a0.2")].values for s in range(3)], axis=0)
    role = w[("real", "sasum_a0.2")].values
    gain = sa - flat

    def test(a, b, groups=docs):
        r = clustered_paired_test(a, b, groups)
        return {"diff": float(np.mean(a - b)), "ci": bootstrap_ci(a - b), "p": r["p_cluster"]}

    p1, p2 = test(sa, flat), test(sa, placebo)
    rho, p3p = stats.spearmanr(w_max, gain)
    uni, gen = gain[w_max >= 0.9], gain[w_max < 0.5]
    p6 = test(role, sa)
    p7 = test(role[summary], sa[summary], docs[summary]) if summary.sum() else None
    res = {
        "n": len(qs),
        "means": {"flat": flat.mean(), "SA": sa.mean(), "SA placebo": placebo.mean(), "role-neutral": role.mean()},
        "P1": {**p1, "holds": p1["diff"] > 0 and p1["p"] < 0.05},
        "P2": {**p2, "holds": p2["diff"] > 0 and p2["p"] < 0.05},
        "P3": {"spearman": float(rho), "p": float(p3p), "holds": rho > 0 and p3p < 0.05},
        "P4": {"n": int(summary.sum()), "mean_gain": float(gain[summary].mean()), "holds": gain[summary].mean() < 0},
        "P5": {
            "unique": float(uni.mean()),
            "generic": float(gen.mean()),
            "n_unique": len(uni),
            "n_generic": len(gen),
            "holds": uni.mean() > gen.mean(),
        },
        "P6": {**p6, "holds": p6["diff"] >= -0.005},
        "P7": {**p7, "n": int(summary.sum()), "holds": p7["diff"] > 0} if p7 else None,
    }
    (DIR / "heldout_results.json").write_text(json.dumps(res, indent=2, default=float))

    def ok(x):
        return "**holds**" if x["holds"] else "**fails**"

    md = [
        "# Held-out test: ConditionalQA results against the pre-registered predictions\n",
        "Predictions: [`heldout_predictions.md`](heldout_predictions.md) (committed and pushed before this run). "
        f"n = {len(qs)} questions; tests are document-clustered sign-flip tests; 95% bootstrap CIs.\n",
        f"MRR@10: flat {flat.mean():.3f}, structure-aware {sa.mean():.3f}, placebo {placebo.mean():.3f}, "
        f"role-neutral {role.mean():.3f}.\n",
        "| | Prediction | Result | Verdict |",
        "|---|---|---|---|",
        f"| P1 | SA beats flat (+0.02 to +0.03) | {p1['diff']:+.3f} [{p1['ci'][0]:+.3f}, {p1['ci'][1]:+.3f}], "
        f"p = {p1['p']:.3f} | {ok(res['P1'])} |",
        f"| P2 | Real beats shuffled headings | {p2['diff']:+.3f} [{p2['ci'][0]:+.3f}, {p2['ci'][1]:+.3f}], "
        f"p = {p2['p']:.3f} | {ok(res['P2'])} |",
        f"| P3 | Gain rises with target specificity | Spearman {rho:.3f}, p = {p3p:.3g} | {ok(res['P3'])} |",
        f"| P4 | Summary-only targets lose | mean gain {gain[summary].mean():+.3f} (n = {summary.sum()}) "
        f"| {ok(res['P4'])} |",
        f"| P5 | Unique > generic targets | unique {uni.mean():+.3f} (n = {len(uni)}) vs generic {gen.mean():+.3f} "
        f"(n = {len(gen)}) | {ok(res['P5'])} |",
        f"| P6 | Role-neutral ≥ SA − 0.005 | {p6['diff']:+.3f} [{p6['ci'][0]:+.3f}, {p6['ci'][1]:+.3f}], "
        f"p = {p6['p']:.3f} | {ok(res['P6'])} |",
    ]
    if p7:
        md.append(
            f"| P7 | Role-neutral > SA on summary-only targets | {p7['diff']:+.3f} (n = {summary.sum()}), "
            f"p = {p7['p']:.3f} | {ok(res['P7'])} |"
        )
    (DIR / "heldout_results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

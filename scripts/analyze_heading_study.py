"""Heading study, part 3: analysis of the per-query results.

A. Do headings help? flat vs structure-aware with real, placebo, generic, OCR-noised,
   flattened and LLM-rewritten headings (paired tests).
B. Dose-response: gain over flat as headings are progressively made generic or noisy.
C. Does heading informativeness predict the gain? (per document and per question)
D. Headroom: best fixed path weight vs per-document and per-question oracles.
E. Gating: path weight set per document from the informativeness score vs fixed weights
   and a learned per-query gate, with document-level cross-validation.
F. Fine-tuning absorption: does a fine-tuned encoder remove the gain only where headings
   are informative? (needs run_heading_study.py --encoder ... --tag ft)

uv run python scripts/analyze_heading_study.py
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test, paired_test  # noqa: E402
from structuralrag.retrieval.dense import Encoder  # noqa: E402
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config  # noqa: E402
from structuralrag.utils.io import load_queries  # noqa: E402

log = logging.getLogger("analyze_heading_study")
DIR = REPORTS_DIR / "heading_study"
FIG = DIR / "figures"
SETS = [("govreport", "ict"), ("govreport", "llm"), ("qasper", "human"), ("nq", "human")]
LABEL = {
    ("govreport", "ict"): "GovReport (ICT)",
    ("govreport", "llm"): "GovReport (LLM-written)",
    ("qasper", "human"): "QASPER (human)",
    ("nq", "human"): "NQ (human)",
}
QFILES = {
    ("govreport", "ict"): "test_queries.jsonl",
    ("govreport", "llm"): "test_queries_llm.jsonl",
    ("qasper", "human"): "qasper_queries.jsonl",
    ("nq", "human"): "nq_queries.jsonl",
}
DOCFILES = {"govreport": "test_documents.jsonl", "qasper": "qasper_documents.jsonl", "nq": "nq_documents.jsonl"}
FEATURES = ["self_nrank", "self_mrr", "title_mrr", "sibling_sim", "generic_rate", "title_idf"]
SEED = 13


def fmt(rows: list[dict], cols: list[str]) -> str:
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return out


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return float(rho), float(p), float(lo), float(hi)


def section_controls() -> dict[str, dict]:
    """Per-section covariates for the robustness regression."""
    from structuralrag.utils.io import load_documents

    out = {}
    for f in DOCFILES.values():
        for d in load_documents(DATA_DIR / f):
            n = len(d.sections)
            for k, x in enumerate(d.sections):
                out[x.uid] = {
                    "log_words": np.log1p(len(x.text.split())),
                    "depth": x.level,
                    "position": k / max(n - 1, 1),
                    "log_sections": np.log(n),
                }
    return out


def clustered_ols(
    y: np.ndarray, X: np.ndarray, groups: np.ndarray, n_boot: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients with a bootstrap over documents (clusters) for 95% CIs."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        boots.append(np.linalg.lstsq(X[pick], y[pick], rcond=None)[0])
    return beta, np.quantile(np.array(boots), [0.025, 0.975], axis=0)


def ptest(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> dict:
    """Per-question permutation test plus the document-clustered version (the primary test)."""
    return {**paired_test(a, b), **clustered_paired_test(a, b, groups)}


def holm(tests: list[dict], key: str = "p_cluster") -> list[dict]:
    """Holm-Bonferroni adjusted p-values within one family of tests (added as p_holm)."""
    order = np.argsort([t[key] for t in tests])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i][key]))
        tests[i]["p_holm"] = running
    return tests


def specificity_by_uid() -> dict[str, float]:
    from structuralrag.analysis.informativeness import heading_specificity
    from structuralrag.retrieval.corpus import SectionCorpus
    from structuralrag.utils.io import load_documents

    out = {}
    for f in DOCFILES.values():
        corpus = SectionCorpus(load_documents(DATA_DIR / f))
        out.update(zip([x.uid for x in corpus.sections], heading_specificity(corpus), strict=True))
    return out


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    pq = pd.read_csv(DIR / "per_query.csv.gz")
    docf = pd.read_csv(DIR / "doc_features.csv")
    secf = pd.read_csv(DIR / "section_features.csv")
    settings = json.loads((DIR / "settings.json").read_text())
    return pq, docf, secf, settings


def wide(pq: pd.DataFrame, ds: str, qs: str) -> pd.DataFrame:
    sub = pq[(pq.dataset == ds) & (pq.query_set == qs)]
    w = sub.pivot_table(index="query_id", columns=["variant", "system"], values="rr")
    meta = sub.drop_duplicates("query_id").set_index("query_id")[["document_id", "gold_uid", "target_level", "lead"]]
    meta = meta.loc[w.index]  # same row order as the score table
    return w, meta


def main() -> None:
    cfg = load_config()
    pq, docf, secf, settings = load()
    a0 = settings["base_params"]["alpha"]
    SA = f"sa_a{a0}"
    alphas = settings["alphas"]
    FIG.mkdir(parents=True, exist_ok=True)
    sets = [s for s in SETS if ((pq.dataset == s[0]) & (pq.query_set == s[1])).any()]
    out: dict = {"alpha_fixed": a0, "alphas": alphas}
    md = [
        "# Heading study\n",
        f"Within-document section retrieval. Structure-aware (SA) uses the GovReport-dev settings (path weight "
        f"alpha = {a0}, BM25 over the structural view fused by RRF); flat = RRF of BM25 and dense over section "
        "text only. Only headings change between variants; section text and questions are identical.\n",
    ]

    # ---------- A. Do headings help? ----------
    rows, sig, per_set = [], [], {}
    for ds, qs in sets:
        w, meta = wide(pq, ds, qs)
        per_set[(ds, qs)] = (w, meta)
        flat = w[("real", "flat")].values
        real = w[("real", SA)].values
        plac = np.mean([w[(f"placebo_s{s}", SA)].values for s in range(3) if (f"placebo_s{s}", SA) in w], axis=0)
        r = {
            "set": LABEL[(ds, qs)],
            "n": len(w),
            "flat": flat.mean(),
            "SA real": real.mean(),
            "SA placebo": plac.mean(),
        }
        for v, name in (("generic_1.0", "SA generic"), ("flatten", "SA flattened"), ("ocr_0.2", "SA OCR 0.2")):
            r[name] = w[(v, SA)].mean() if (v, SA) in w else float("nan")
        rows.append(r)
        for a, b, an, bn in ((real, flat, "SA real", "flat"), (real, plac, "SA real", "SA placebo")):
            sig.append({"set": LABEL[(ds, qs)], "comparison": f"{an} vs {bn}", **ptest(a, b, meta.document_id.values)})
        if ("generic_1.0", SA) in w:
            sig.append(
                {
                    "set": LABEL[(ds, qs)],
                    "comparison": "SA real vs SA generic",
                    **ptest(real, w[("generic_1.0", SA)].values, meta.document_id.values),
                }
            )
    for name in {t["set"] for t in sig}:
        holm([t for t in sig if t["set"] == name])
    cols = ["set", "n", "flat", "SA real", "SA placebo", "SA generic", "SA flattened", "SA OCR 0.2"]
    md += [
        "\n## A. Do headings help? (MRR@10)\n",
        fmt(rows, cols),
        "\n",
        fmt(sig, ["set", "comparison", "mean_diff", "p_permutation", "p_cluster", "p_holm"]),
    ]
    out["A"] = {"table": rows, "tests": sig}

    # ---------- B. Dose-response ----------
    dose = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ds, qs in sets:
        w, _ = per_set[(ds, qs)]
        flat = w[("real", "flat")].values
        for kind, levels, ax in (
            ("generic", [0.0, 0.25, 0.5, 0.75, 1.0], axes[0]),
            ("ocr", [0.0, 0.1, 0.2, 0.4], axes[1]),
        ):
            ys = []
            for lv in levels:
                v = "real" if lv == 0 else f"{kind}_{lv}"
                g = w[(v, SA)].values - flat
                lo, hi = bootstrap_ci(g)
                ys.append(g.mean())
                dose.append(
                    {
                        "set": LABEL[(ds, qs)],
                        "corruption": kind,
                        "level": lv,
                        "gain_vs_flat": g.mean(),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
            ax.plot(levels, ys, marker="o", label=LABEL[(ds, qs)])
    for ax, t in zip(axes, ("Share of headings made generic", "OCR noise rate on headings"), strict=True):
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(t)
    axes[0].set_ylabel("MRR@10 gain of SA over flat")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "dose_response.png", dpi=200)
    plt.close(fig)
    md += [
        "\n## B. Dose-response (SA gain over flat, MRR@10)\n",
        fmt(dose, ["set", "corruption", "level", "gain_vs_flat", "ci_lo", "ci_hi"]),
    ]
    out["B"] = dose

    # ---------- C. Does informativeness predict the gain? ----------
    corr, qcorr, variant_pts = [], [], []
    real_feats = docf[docf.variant == "real"]
    for ds, qs in sets:
        w, meta = per_set[(ds, qs)]
        gain = pd.Series(w[("real", SA)].values - w[("real", "flat")].values, index=w.index)
        plac = np.mean([w[(f"placebo_s{s}", SA)].values for s in range(3)], axis=0)
        gain_p = pd.Series(w[("real", SA)].values - plac, index=w.index)
        per_doc = (
            pd.DataFrame({"gain": gain, "gain_vs_placebo": gain_p, "document_id": meta.document_id})
            .groupby("document_id")
            .mean()
        )
        f = real_feats[real_feats.dataset == ds].set_index("document_id").join(per_doc, how="inner")
        for feat in FEATURES:
            for target in ("gain", "gain_vs_placebo"):
                rho, p, lo, hi = spearman_ci(f[feat].values, f[target].values)
                corr.append(
                    {
                        "set": LABEL[(ds, qs)],
                        "feature": feat,
                        "target": target,
                        "docs": len(f),
                        "spearman": rho,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "p": p,
                    }
                )
        # per question: does the gold section's own heading identify it?
        s = secf[secf.dataset == ds].set_index("uid")
        g_nrank = meta.gold_uid.map(s.path_nrank)
        rho, p, lo, hi = spearman_ci(g_nrank.values.astype(float), gain.values)
        qcorr.append(
            {
                "set": LABEL[(ds, qs)],
                "questions": int(g_nrank.notna().sum()),
                "spearman": rho,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
            }
        )
        # across heading variants: mean informativeness vs mean gain
        for v in sorted({c[0] for c in w.columns}):
            dfv = docf[(docf.dataset == ds) & (docf.variant == v)]
            if (v, SA) in w and len(dfv):
                variant_pts.append(
                    {
                        "set": LABEL[(ds, qs)],
                        "variant": v,
                        "self_nrank": float(dfv.self_nrank.mean()),
                        "gain_vs_flat": float((w[(v, SA)] - w[("real", "flat")]).mean()),
                    }
                )
    vp = pd.DataFrame(variant_pts)
    rho_v, p_v = stats.spearmanr(vp.self_nrank, vp.gain_vs_flat)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, g in vp.groupby("set"):
        ax.scatter(g.self_nrank, g.gain_vs_flat, label=name, s=18)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Heading informativeness (mean self-rank score)")
    ax.set_ylabel("MRR@10 gain of SA over flat")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"All heading variants (Spearman {rho_v:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "informativeness_vs_gain.png", dpi=200)
    plt.close(fig)
    md += [
        "\n## C. Does heading informativeness predict the gain?\n",
        "Per document (real headings): Spearman correlation between each feature and the document's mean gain.\n",
        fmt(
            [
                c
                for c in corr
                if c["feature"] in ("self_nrank", "title_mrr", "generic_rate", "title_idf", "sibling_sim")
            ],
            ["set", "feature", "target", "docs", "spearman", "ci_lo", "ci_hi", "p"],
        ),
        "\nPer question: does the gold section's own heading path rank it first among its document's sections?\n",
        fmt(qcorr, ["set", "questions", "spearman", "ci_lo", "ci_hi", "p"]),
        f"\nAcross all heading variants and datasets ({len(vp)} points): Spearman between mean informativeness "
        f"and mean gain = {rho_v:.3f} (p = {p_v:.2g}).\n",
    ]
    out["C"] = {"doc": corr, "question": qcorr, "variants": variant_pts, "variant_spearman": [rho_v, p_v]}

    # ---------- D/E. Oracle headroom and gating ----------
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    gate_rows, oracle_rows, gate_tests = [], [], []
    pooled = []
    for ds, qs in sets:
        w, meta = per_set[(ds, qs)]
        R = np.stack([w[("real", f"sa_a{a}")].values for a in alphas], axis=1)  # (n_q, n_alpha)
        docs = meta.document_id.values
        his = real_feats[real_feats.dataset == ds].set_index("document_id").self_nrank.reindex(docs).values
        qtext = {q.query_id: q.question for q in load_queries(DATA_DIR / QFILES[(ds, qs)])}
        emb = encoder.encode_queries([qtext[i] for i in w.index])
        pooled.append((LABEL[(ds, qs)], R, docs, his, emb, w[("real", "flat")].values))
        oracle_rows.append(oracle(LABEL[(ds, qs)], R, docs, alphas, a0))
        res, tests = gate_cv(LABEL[(ds, qs)], R, docs, his, emb, alphas, a0)
        gate_rows.append(res)
        gate_tests += tests
    # pooled across datasets (where informativeness varies most)
    names, Rs, ds_, hs, es, _ = zip(*pooled, strict=True)
    keep = [i for i, n in enumerate(names) if n != LABEL[("govreport", "llm")]]  # same documents as GovReport ICT
    R = np.concatenate([Rs[i] for i in keep])
    docs = np.concatenate([np.char.add(f"{i}:", ds_[i].astype(str)) for i in keep])
    his = np.concatenate([hs[i] for i in keep])
    emb = np.concatenate([es[i] for i in keep])
    oracle_rows.append(oracle("Pooled (3 datasets)", R, docs, alphas, a0))
    res, tests = gate_cv("Pooled (3 datasets)", R, docs, his, emb, alphas, a0)
    gate_rows.append(res)
    gate_tests += tests
    md += [
        "\n## D. Headroom from choosing the path weight per document or per question (MRR@10, in-sample)\n",
        fmt(oracle_rows, list(oracle_rows[0])),
        "\n## E. Setting the path weight (2-fold cross-validation over documents, 5 repeats; test MRR@10)\n",
        fmt(gate_rows, list(gate_rows[0])),
        "\n",
        fmt(gate_tests, ["set", "comparison", "mean_diff", "p_permutation", "p_wilcoxon"]),
    ]
    out["D"] = oracle_rows
    out["E"] = {"results": gate_rows, "tests": gate_tests}

    # ---------- F. Fine-tuning absorption ----------
    ft_path = DIR / "per_query_ft.csv.gz"
    if ft_path.exists():
        ft = pd.read_csv(ft_path)
        rows_f = []
        for ds, qs in sets:
            w, meta = per_set[(ds, qs)]
            wf, _ = wide(ft, ds, qs)
            his = (
                real_feats[real_feats.dataset == ds]
                .set_index("document_id")
                .self_nrank.reindex(meta.document_id.values)
                .values
            )
            terc = pd.qcut(pd.Series(his).rank(method="first"), 3, labels=["low", "mid", "high"]).values
            for name, ww in (("base", w), ("fine-tuned", wf.reindex(w.index))):
                gain = ww[("real", SA)].values - ww[("real", "flat")].values
                gp = (
                    ww[("real", SA)].values - ww[("placebo_s0", SA)].values
                    if ("placebo_s0", SA) in ww
                    else np.full(len(gain), np.nan)
                )
                for t in ("low", "mid", "high"):
                    m = terc == t
                    rows_f.append(
                        {
                            "set": LABEL[(ds, qs)],
                            "encoder": name,
                            "informativeness": t,
                            "n": int(m.sum()),
                            "gain_vs_flat": float(np.nanmean(gain[m])),
                            "gain_vs_placebo": float(np.nanmean(gp[m])),
                        }
                    )
        md += [
            "\n## F. Fine-tuning absorption (gain of SA by informativeness tercile)\n",
            fmt(rows_f, ["set", "encoder", "informativeness", "n", "gain_vs_flat", "gain_vs_placebo"]),
        ]
        out["F"] = rows_f

    # ---------- H. Gain by the target section's heading specificity ----------
    spec = specificity_by_uid()
    bins = [(0.0, 0.5, "generic (w < 0.5)"), (0.5, 0.9, "shared (0.5-0.9)"), (0.9, 1.01, "unique (w >= 0.9)")]
    rows_h = []
    for ds, qs in sets:
        w, meta = per_set[(ds, qs)]
        sp = meta.gold_uid.map(spec).values
        gain = w[("real", SA)].values - w[("real", "flat")].values
        for lo, hi, name in bins:
            m = (sp >= lo) & (sp < hi)
            if m.sum():
                ci = bootstrap_ci(gain[m])
                rows_h.append(
                    {
                        "set": LABEL[(ds, qs)],
                        "gold heading": name,
                        "n": int(m.sum()),
                        "gain_vs_flat": float(gain[m].mean()),
                        "ci_lo": ci[0],
                        "ci_hi": ci[1],
                    }
                )
        rho, p, lo_, hi_ = spearman_ci(sp, gain)
        rows_h.append(
            {
                "set": LABEL[(ds, qs)],
                "gold heading": f"Spearman {rho:.3f} [{lo_:.3f}, {hi_:.3f}]",
                "n": int(len(sp)),
                "gain_vs_flat": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": p,
            }
        )
    md += [
        "\n## H. Gain by the target section's heading specificity (w = log(N/df)/log N)\n",
        "The last row per set is the Spearman correlation between w and the per-question gain (p in last column).\n",
        fmt(rows_h, ["set", "gold heading", "n", "gain_vs_flat", "ci_lo", "ci_hi"]),
    ]
    out["H"] = rows_h

    # ---------- J. Robustness: does specificity survive controls? ----------
    ctrl = section_controls()
    cov = ["specificity", "log_words", "depth", "position", "log_sections"]
    rows_j, pooled_y, pooled_X, pooled_g, pooled_set = [], [], [], [], []
    for ds, qs in sets:
        w, meta = per_set[(ds, qs)]
        y = w[("real", SA)].values - w[("real", "flat")].values
        feats = pd.DataFrame([ctrl[u] for u in meta.gold_uid])
        feats.insert(0, "specificity", meta.gold_uid.map(spec).values)
        Z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
        X = np.column_stack([np.ones(len(y)), Z.values])
        beta, ci = clustered_ols(y, X, meta.document_id.values)
        for k, name in enumerate(cov, start=1):
            rows_j.append(
                {
                    "set": LABEL[(ds, qs)],
                    "covariate (standardized)": name,
                    "coef": float(beta[k]),
                    "ci_lo": float(ci[0, k]),
                    "ci_hi": float(ci[1, k]),
                }
            )
        if (ds, qs) != ("govreport", "llm"):
            pooled_y.append(y)
            pooled_X.append(Z.values)
            pooled_g.append(np.char.add(f"{ds}:", meta.document_id.values.astype(str)))
            pooled_set += [LABEL[(ds, qs)]] * len(y)
    y = np.concatenate(pooled_y)
    Z = np.concatenate(pooled_X)
    fe = pd.get_dummies(pd.Series(pooled_set), drop_first=True).values.astype(float)
    X = np.column_stack([np.ones(len(y)), Z, fe])
    beta, ci = clustered_ols(y, X, np.concatenate(pooled_g))
    for k, name in enumerate(cov, start=1):
        rows_j.append(
            {
                "set": "Pooled (3 datasets, dataset fixed effects)",
                "covariate (standardized)": name,
                "coef": float(beta[k]),
                "ci_lo": float(ci[0, k]),
                "ci_hi": float(ci[1, k]),
            }
        )
    md += [
        "\n## J. Robustness: per-question gain regressed on the target's heading specificity and controls\n",
        "OLS on standardized covariates (target section length, depth, relative position, document size); "
        "95% CIs from a bootstrap over documents.\n",
        fmt(rows_j, ["set", "covariate (standardized)", "coef", "ci_lo", "ci_hi"]),
    ]
    out["J"] = rows_j

    # ---------- K. Replication with a larger encoder ----------
    rep_path = DIR / "per_query_bgebase.csv.gz"
    if rep_path.exists():
        rep = pd.read_csv(rep_path)
        rows_k = []
        for ds, qs in sets:
            if not ((rep.dataset == ds) & (rep.query_set == qs)).any():
                continue
            wk, mk = wide(rep, ds, qs)
            flat_k, real_k = wk[("real", "flat")].values, wk[("real", SA)].values
            gain_k = real_k - flat_k
            sp = mk.gold_uid.map(spec).values
            rho, p, lo_, hi_ = spearman_ci(sp, gain_k)
            row = {
                "set": LABEL[(ds, qs)],
                "n": len(wk),
                "flat": flat_k.mean(),
                "SA real": real_k.mean(),
                "SA placebo": wk[("placebo_s0", SA)].mean(),
                "SA generic": wk[("generic_1.0", SA)].mean(),
                "gain, generic target": float(gain_k[sp < 0.5].mean()),
                "gain, unique target": float(gain_k[sp >= 0.9].mean()),
                "spearman(w, gain)": rho,
                "p": p,
                "p real vs placebo (cluster)": clustered_paired_test(
                    real_k, wk[("placebo_s0", SA)].values, mk.document_id.values
                )["p_cluster"],
            }
            rows_k.append(row)
        if rows_k:
            md += [
                "\n## K. Replication with bge-base-en-v1.5 (110M parameters; same settings, alpha = 0.2)\n",
                fmt(rows_k, list(rows_k[0])),
            ]
            out["K"] = rows_k

    # ---------- I. Specificity fallback ----------
    # safbm = the method (generic headings become neutral: document-mean path score);
    # safb  = ablation (generic headings fall back to the content score).
    dev = pq[(pq.dataset == "govreport_dev") & (pq.variant == "real")]
    rows_i, tests_i, split_i = [], [], []
    if len(dev) and dev.system.str.startswith("safbm_").any():
        dmean = dev.groupby("system").rr.mean()
        a_sa = max(alphas, key=lambda a: dmean[f"sa_a{a}"])
        a_m = max(alphas, key=lambda a: dmean[f"safbm_a{a}"])
        a_c = max(alphas, key=lambda a: dmean[f"safb_a{a}"])
        out["I_dev_alpha"] = {"structure-aware": a_sa, "fallback (mean)": a_m, "fallback (content)": a_c}
        fig, axes = plt.subplots(1, len(sets), figsize=(3.0 * len(sets), 2.9), sharex=True)
        for ax, (ds, qs) in zip(np.atleast_1d(axes), sets, strict=True):
            w, meta = per_set[(ds, qs)]
            flat = w[("real", "flat")].values
            sa = w[("real", f"sa_a{a_sa}")].values
            fm = w[("real", f"safbm_a{a_m}")].values
            fc = w[("real", f"safb_a{a_c}")].values
            rows_i.append(
                {
                    "set": LABEL[(ds, qs)],
                    "n": len(w),
                    "flat": flat.mean(),
                    f"SA (a={a_sa})": sa.mean(),
                    f"+ fallback, neutral (a={a_m})": fm.mean(),
                    f"+ fallback, content (a={a_c})": fc.mean(),
                }
            )
            lab = LABEL[(ds, qs)]
            fam = [
                {"set": lab, "comparison": "fallback (neutral) vs SA", **ptest(fm, sa, meta.document_id.values)},
                {"set": lab, "comparison": "fallback (content) vs SA", **ptest(fc, sa, meta.document_id.values)},
                {"set": lab, "comparison": "fallback (neutral) vs flat", **ptest(fm, flat, meta.document_id.values)},
                {"set": lab, "comparison": "SA vs flat", **ptest(sa, flat, meta.document_id.values)},
            ]
            tests_i += holm(fam)
            sp = meta.gold_uid.map(spec).values
            for lo, hi, name in bins:
                m = (sp >= lo) & (sp < hi)
                if m.sum():
                    split_i.append(
                        {
                            "set": lab,
                            "gold heading": name,
                            "n": int(m.sum()),
                            "SA - flat": float((sa - flat)[m].mean()),
                            "neutral fallback - flat": float((fm - flat)[m].mean()),
                            "content fallback - flat": float((fc - flat)[m].mean()),
                        }
                    )
            for key, name in (
                ("sa", "structure-aware"),
                ("safbm", "+ fallback (neutral)"),
                ("safb", "+ fallback (content)"),
            ):
                ax.plot(alphas, [w[("real", f"{key}_a{a}")].mean() for a in alphas], marker="o", ms=3, label=name)
            ax.axhline(flat.mean(), color="grey", lw=0.8, ls="--", label="flat")
            ax.set_title(lab, fontsize=8)
            ax.set_xlabel("path weight alpha", fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("MRR@10")
        np.atleast_1d(axes)[0].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "alpha_sweep_fallback.png", dpi=200)
        plt.close(fig)
        md += [
            f"\n## I. Specificity fallback (alpha tuned on GovReport dev: SA {a_sa}, neutral {a_m}, content {a_c}; "
            "MRR@10)\n",
            fmt(rows_i, list(rows_i[0])),
            "\n",
            fmt(tests_i, ["set", "comparison", "mean_diff", "p_permutation", "p_cluster", "p_holm"]),
            "\nBy the target section's heading specificity:\n",
            fmt(split_i, list(split_i[0])),
        ]
        out["I"] = {"table": rows_i, "tests": tests_i, "by_specificity": split_i}

    # ---------- G. LLM-rewritten headings ----------
    rows_g, tests_g = [], []
    for ds, qs in sets:
        w, meta = per_set[(ds, qs)]
        if ("llm", SA) not in w:
            continue
        path = DATA_DIR / f"{ds}_llm_headings.json"
        rewritten = set(json.loads(path.read_text())) if path.exists() else set()
        m = meta.document_id.isin(rewritten).values
        if m.sum() < 20:
            continue
        his = real_feats[real_feats.dataset == ds].set_index("document_id").self_nrank
        his_llm = docf[(docf.dataset == ds) & (docf.variant == "llm")].set_index("document_id").self_nrank
        docs_m = meta.document_id.values[m]
        r = {
            "set": LABEL[(ds, qs)],
            "n": int(m.sum()),
            "flat": w[("real", "flat")].values[m].mean(),
            "SA generic": w[("generic_1.0", SA)].values[m].mean(),
            "SA real": w[("real", SA)].values[m].mean(),
            "SA LLM-written": w[("llm", SA)].values[m].mean(),
            "informativeness real": float(his.reindex(docs_m).mean()),
            "informativeness LLM": float(his_llm.reindex(docs_m).mean()),
        }
        rows_g.append(r)
        tests_g.append(
            {
                "set": LABEL[(ds, qs)],
                "comparison": "LLM-written vs real",
                **ptest(w[("llm", SA)].values[m], w[("real", SA)].values[m], meta.document_id.values[m]),
            }
        )
        tests_g.append(
            {
                "set": LABEL[(ds, qs)],
                "comparison": "LLM-written vs generic",
                **ptest(w[("llm", SA)].values[m], w[("generic_1.0", SA)].values[m], meta.document_id.values[m]),
            }
        )
    if rows_g:
        md += [
            "\n## G. LLM-written headings (rewritten documents only, MRR@10)\n",
            fmt(rows_g, list(rows_g[0])),
            "\n",
            fmt(tests_g, ["set", "comparison", "mean_diff", "p_permutation", "p_cluster"]),
        ]
        out["G"] = {"table": rows_g, "tests": tests_g}

    (DIR / "results.json").write_text(json.dumps(out, indent=2, default=float))
    (DIR / "results.md").write_text("\n".join(md))
    print("\n".join(md))


def oracle(name: str, R: np.ndarray, docs: np.ndarray, alphas: list[float], a0: float) -> dict:
    means = R.mean(axis=0)
    per_doc = pd.DataFrame(R).groupby(docs).transform("mean").values
    doc_best = R[np.arange(len(R)), per_doc.argmax(axis=1)]
    return {
        "set": name,
        "fixed alpha=" + str(a0): float(means[alphas.index(a0)]),
        "best fixed alpha": f"{alphas[int(means.argmax())]} ({means.max():.3f})",
        "alpha=0": float(means[alphas.index(0.0)]),
        "per-document oracle": float(doc_best.mean()),
        "per-question oracle": float(R.max(axis=1).mean()),
    }


def gate_cv(name, R, docs, his, emb, alphas, a0, repeats: int = 5):
    """Test-fold MRR for each way of choosing alpha; folds split by document."""
    rng = np.random.default_rng(SEED)
    udocs = np.unique(docs)
    methods = ["fixed alpha=" + str(a0), "fixed, tuned", "informativeness gate", "query gate (ridge)"]
    preds = {m: np.zeros((repeats, len(R))) for m in methods}
    i0 = alphas.index(a0)
    for rep in range(repeats):
        perm = rng.permutation(udocs)
        fold_of = {d: k % 2 for k, d in enumerate(perm)}
        fold = np.array([fold_of[d] for d in docs])
        for k in (0, 1):
            tr, te = fold != k, fold == k
            preds[methods[0]][rep, te] = R[te, i0]
            best = R[tr].mean(axis=0).argmax()
            preds[methods[1]][rep, te] = R[te, best]
            # informativeness gate: best alpha per tercile of the train documents' scores
            h_tr = np.nan_to_num(his[tr], nan=np.nanmedian(his[tr]))
            h_te = np.nan_to_num(his[te], nan=np.nanmedian(his[tr]))
            edges = np.quantile(h_tr, [1 / 3, 2 / 3])
            b_tr, b_te = np.digitize(h_tr, edges), np.digitize(h_te, edges)
            choice = {b: R[tr][b_tr == b].mean(axis=0).argmax() if (b_tr == b).any() else best for b in range(3)}
            preds[methods[2]][rep, te] = R[te, [choice[b] for b in b_te]]
            # learned per-query gate (predict each alpha's reciprocal rank from the query embedding)
            model = Ridge(alpha=10.0).fit(emb[tr], R[tr])
            preds[methods[3]][rep, te] = R[te, model.predict(emb[te]).argmax(axis=1)]
    row = {"set": name, "n": len(R)}
    for m in methods:
        row[m] = float(preds[m].mean())
    per_q = {m: preds[m].mean(axis=0) for m in methods}
    tests = [
        {"set": name, "comparison": f"{m} vs {methods[0]}", **paired_test(per_q[m], per_q[methods[0]])}
        for m in methods[1:]
    ]
    return row, tests


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

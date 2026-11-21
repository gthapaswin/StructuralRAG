"""Heading study: can summary-role sections be found without a list of heading words?

The role-neutral method gives summary-role sections (a fixed English list: Introduction, Overview,
Conclusions, ...) a neutral heading score. This tests label-free replacements that never read the
heading text:

- position: the document's first top-level section (introductions and leads open a document)
- position+last: also the last top-level section (conclusions close it)
- centrality: the section whose text is on average most similar to the document's other sections
  (a summary restates the rest)

The rule is chosen on GovReport dev only, then reported on all five question sets with
document-clustered tests. Same settings as the main study (bge-small, alpha 0.2).

uv run python scripts/run_label_free_roles.py
Output: reports/heading_study/label_free_roles.{md,json}
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np

from structuralrag.analysis import experiment as ex
from structuralrag.analysis.informativeness import centrality_weights, position_weights, role_weights
from structuralrag.evaluation.statistical_tests import bootstrap_ci, clustered_paired_test
from structuralrag.retrieval.corpus import SectionCorpus, View
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries

log = logging.getLogger("run_label_free_roles")
OUT = REPORTS_DIR / "heading_study"
SETS = [
    ("GovReport dev (ICT, selection)", "dev_documents.jsonl", "dev_queries.jsonl"),
    ("GovReport (ICT)", "test_documents.jsonl", "test_queries.jsonl"),
    ("GovReport (LLM-written)", "test_documents.jsonl", "test_queries_llm.jsonl"),
    ("QASPER (human)", "qasper_documents.jsonl", "qasper_queries.jsonl"),
    ("NQ (human)", "nq_documents.jsonl", "nq_queries.jsonl"),
    ("ConditionalQA (human)", "conditionalqa_documents.jsonl", "conditionalqa_queries.jsonl"),
]
RULES = ["heading list", "position", "position+last", "centrality"]


def main() -> None:
    cfg = load_config()
    base = json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"]
    alpha, beta, sw = base["alpha"], base["beta"], base["sparse_weight"]
    encoder = Encoder(cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    res, corpora = {}, {}
    for name, dfile, qfile in SETS:
        if dfile not in corpora:
            c = SectionCorpus(load_documents(DATA_DIR / dfile))
            emb = encoder.encode_passages(c.views(View.TEXT))
            weights = {
                "heading list": role_weights(c),
                "position": position_weights(c),
                "position+last": position_weights(c, last=True),
                "centrality": centrality_weights(c, emb),
            }
            corpora[dfile] = (c, weights)
        corpus, weights = corpora[dfile]
        qs = load_queries(DATA_DIR / qfile)
        comp = ex.compute(corpus, encoder, qs)
        docs = np.array([q.document_id for q in qs])
        rr = {
            "flat": ex.reciprocal_ranks(ex.flat_scores(comp), comp),
            "SA": ex.reciprocal_ranks(ex.structure_scores(comp, alpha, beta, sw), comp),
        }
        regex = weights["heading list"] == 0
        for rule in RULES:
            w = weights[rule]
            cr = replace(comp, specificity=[w[corpus.doc_indices(q.document_id)] for q in qs])
            rr[rule] = ex.reciprocal_ranks(ex.structure_scores(cr, alpha, beta, sw, fallback="mean"), cr)
        agree = {
            r: {
                "neutralized_sections": int((weights[r] == 0).sum()),
                "share_on_list": float(regex[weights[r] == 0].mean()) if (weights[r] == 0).any() else 0.0,
                "list_sections_found": float((weights[r][regex] == 0).mean()) if regex.any() else 0.0,
            }
            for r in RULES[1:]
        }
        tests = {}
        for r in RULES:
            for ref in ("SA", "flat", "heading list"):
                if r == ref:
                    continue
                t = clustered_paired_test(rr[r], rr[ref], docs)
                tests[f"{r} vs {ref}"] = {
                    "diff": float(np.mean(rr[r] - rr[ref])),
                    "ci": bootstrap_ci(rr[r] - rr[ref]),
                    "p": t["p_cluster"],
                }
        res[name] = {"n": len(qs), "mrr": {k: float(v.mean()) for k, v in rr.items()}, "agree": agree, "tests": tests}
        log.info("%s: %s", name, {k: round(v.mean(), 3) for k, v in rr.items()})

    dev = res[SETS[0][0]]["mrr"]
    chosen = max(RULES[1:], key=lambda r: dev[r])
    md = [
        "# Heading study: label-free summary roles\n",
        "Which sections get a neutral heading score (path score = the document mean). 'Heading list' is "
        "the role-neutral method (fixed English list); the other rules never read heading text. "
        f"Rule chosen on GovReport dev: **{chosen}**. MRR@10, alpha = {alpha}, bge-small.\n",
        "| set | n | flat | SA | " + " | ".join(RULES) + " |",
        "|---|---|---|---|" + "---|" * len(RULES),
    ]
    for name, r in res.items():
        m = r["mrr"]
        md.append(
            f"| {name} | {r['n']} | {m['flat']:.3f} | {m['SA']:.3f} | "
            + " | ".join(f"{m[x]:.3f}" for x in RULES)
            + " |"
        )
    md += [
        "\nHow the label-free rules relate to the heading list (test documents):\n",
        "| set | rule | sections neutralized | of which on the list | list sections found |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for (name, dfile, _), r in zip(SETS, res.values(), strict=True):
        if dfile in seen:
            continue
        seen.add(dfile)
        for rule, a in r["agree"].items():
            md.append(
                f"| {name} | {rule} | {a['neutralized_sections']} | {a['share_on_list']:.0%} | "
                f"{a['list_sections_found']:.0%} |"
            )
    md += [
        f"\nDocument-clustered tests for the chosen rule ({chosen}) and the heading list:\n",
        "| set | comparison | difference | 95% CI | p |",
        "|---|---|---|---|---|",
    ]
    for name, r in res.items():
        for key in (f"{chosen} vs SA", f"{chosen} vs flat", f"{chosen} vs heading list", "heading list vs SA"):
            t = r["tests"][key]
            md.append(f"| {name} | {key} | {t['diff']:+.3f} | [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | {t['p']:.3f} |")
    (OUT / "label_free_roles.json").write_text(json.dumps({"chosen": chosen, "results": res}, indent=2, default=float))
    (OUT / "label_free_roles.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

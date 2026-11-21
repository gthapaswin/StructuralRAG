"""Heading study, part 1: controlled heading variants, per-query results, heading features.

For each dataset and heading variant (real, placebo, generic at several rates, OCR noise,
flattened paths) it scores flat retrieval and structure-aware retrieval over a grid of path
weights alpha, within the query's document. Section text and questions never change, so any
difference comes from the headings. Settings come from GovReport dev (no tuning here).

uv run python scripts/run_heading_study.py [--datasets govreport qasper]
Outputs: reports/heading_study/per_query.csv.gz, doc_features.csv, section_features.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from structuralrag.analysis import experiment as ex
from structuralrag.analysis import headings as hd
from structuralrag.analysis.informativeness import document_features, section_features
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.retrieval.dense import Encoder
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, REPORTS_DIR, load_config
from structuralrag.utils.io import load_documents, load_queries
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("run_heading_study")
OUT = REPORTS_DIR / "heading_study"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

DATASETS = {
    "govreport_dev": ("dev_documents.jsonl", {"ict": "dev_queries.jsonl"}),
    "govreport": ("test_documents.jsonl", {"ict": "test_queries.jsonl", "llm": "test_queries_llm.jsonl"}),
    "qasper": ("qasper_documents.jsonl", {"human": "qasper_queries.jsonl"}),
    "nq": ("nq_documents.jsonl", {"human": "nq_queries.jsonl"}),
}


def llm_rewritten(ds, docs):
    path = DATA_DIR / f"{ds}_llm_headings.json"
    if not path.exists():
        return None
    new = json.loads(path.read_text())
    return [hd.retitle(d, new[d.document_id]) if d.document_id in new else d for d in docs]


def variants(ds, docs, only=None):
    out = {"real": lambda: docs}
    for seed in (0, 1, 2):
        out[f"placebo_s{seed}"] = lambda seed=seed: hd.placebo(docs, seed)
    for frac in (0.25, 0.5, 0.75, 1.0):
        out[f"generic_{frac}"] = lambda frac=frac: hd.genericize(docs, frac, seed=0)
    for rate in (0.1, 0.2, 0.4):
        out[f"ocr_{rate}"] = lambda rate=rate: hd.ocr_noise(docs, rate, seed=0)
    out["flatten"] = lambda: hd.flatten(docs)
    out["llm"] = lambda: llm_rewritten(ds, docs)
    for name, make in out.items():
        if only and name not in only:
            continue
        vdocs = make()
        if vdocs is not None:
            yield name, vdocs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["govreport", "qasper", "nq"])
    ap.add_argument("--variants", nargs="+", default=None, help="subset of variants (default: all)")
    ap.add_argument("--encoder", default=None, help="embedding model path (default: config base model)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--merge", action="store_true", help="replace only the computed dataset/variant rows")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    base = json.loads((REPORTS_DIR / "results.json").read_text())["tuned_structure_params"]
    encoder = Encoder(args.encoder or cfg["retrieval"]["embedding_model"], cache_dir=CACHE_DIR)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, doc_frames, sec_frames = [], [], []
    for ds in args.datasets:
        doc_file, qsets = DATASETS[ds]
        docs = load_documents(DATA_DIR / doc_file)
        qsets = {k: load_queries(DATA_DIR / v) for k, v in qsets.items() if (DATA_DIR / v).exists()}
        real_corpus = SectionCorpus(docs)
        cache = {}
        for name, qs in qsets.items():
            cache[name] = (encoder.encode_queries([q.question for q in qs]), ex.text_bm25_scores(real_corpus, qs))
        for variant, vdocs in variants(ds, docs, args.variants):
            t0 = time.time()
            corpus = SectionCorpus(vdocs)
            df = document_features(corpus, encoder, pool=vdocs).assign(dataset=ds, variant=variant)
            doc_frames.append(df)
            if variant == "real":
                sec_frames.append(section_features(corpus, encoder).assign(dataset=ds))
            for qs_name, qs in qsets.items():
                q_emb, sp_text = cache[qs_name]
                comp = ex.compute(corpus, encoder, qs, q_emb=q_emb, sparse_text=sp_text)
                meta = pd.DataFrame(
                    {
                        "query_id": [q.query_id for q in qs],
                        "document_id": [q.document_id for q in qs],
                        "target_level": [q.metadata.get("target_level", np.nan) for q in qs],
                        "lead": [q.metadata.get("lead", False) for q in qs],
                        "gold_uid": [sorted(q.target_uids)[0] for q in qs],
                    }
                )
                rows = {"flat": ex.reciprocal_ranks(ex.flat_scores(comp), comp)}
                rr_alpha = ex.structure_rr_per_alpha(
                    comp, ALPHAS, beta=base["beta"], sparse_weight=base["sparse_weight"]
                )
                for a, rr in zip(ALPHAS, rr_alpha, strict=True):
                    rows[f"sa_a{a}"] = rr
                rr_fb = ex.structure_rr_per_alpha(
                    comp, ALPHAS, beta=base["beta"], sparse_weight=base["sparse_weight"], fallback=True
                )
                for a, rr in zip(ALPHAS, rr_fb, strict=True):
                    rows[f"safb_a{a}"] = rr
                rr_fbm = ex.structure_rr_per_alpha(
                    comp, ALPHAS, beta=base["beta"], sparse_weight=base["sparse_weight"], fallback="mean"
                )
                for a, rr in zip(ALPHAS, rr_fbm, strict=True):
                    rows[f"safbm_a{a}"] = rr
                for system, rr in rows.items():
                    frames.append(meta.assign(dataset=ds, query_set=qs_name, variant=variant, system=system, rr=rr))
                log.info(
                    "%s/%s/%s: flat %.3f  SA(a=%.1f) %.3f  (%.0fs)",
                    ds,
                    qs_name,
                    variant,
                    rows["flat"].mean(),
                    base["alpha"],
                    rows[f"sa_a{base['alpha']}"].mean(),
                    time.time() - t0,
                )
    t = f"_{args.tag}" if args.tag else ""
    outputs = {
        "per_query": (frames, ".csv.gz"),
        "doc_features": (doc_frames, ".csv"),
        "section_features": (sec_frames, ".csv"),
    }
    for name, (fr, ext) in outputs.items():
        if not fr:
            continue
        new = pd.concat(fr)
        path = OUT / f"{name}{t}{ext}"
        if args.merge and path.exists():
            old = pd.read_csv(path)
            keys = ["dataset", "variant"] if "variant" in new else ["dataset"]
            done = set(map(tuple, new[keys].drop_duplicates().values))
            old = old[[tuple(r) not in done for r in old[keys].values]]
            new = pd.concat([old, new])
        new.to_csv(path, index=False)
    settings = {"base_params": base, "alphas": ALPHAS, "encoder": encoder.model_name}
    (OUT / f"settings{t}.json").write_text(json.dumps(settings, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

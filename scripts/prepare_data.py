"""Download GovReport, build hierarchies, generate ICT queries, validate, save JSONL.

uv run python scripts/prepare_data.py [--config configs/experiment.yaml]
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from structuralrag.data_generation.ict_queries import generate_ict_queries
from structuralrag.ingestion.govreport import load_govreport
from structuralrag.preprocessing.hierarchy_builder import validate_hierarchy
from structuralrag.schema import Document, Query
from structuralrag.utils.config import DATA_DIR, load_config
from structuralrag.utils.io import write_json, write_jsonl
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("prepare_data")


def check_leakage(
    dev_docs: list[Document], test_docs: list[Document], queries: list[Query], docs: list[Document]
) -> dict:
    dev_ids, test_ids = {d.document_id for d in dev_docs}, {d.document_id for d in test_docs}
    overlap = dev_ids & test_ids
    if overlap:
        raise RuntimeError(f"Document leakage between dev and test: {sorted(overlap)[:5]}")
    by_uid = {s.uid: s for d in docs for s in d.sections}
    leaked, orphan = 0, 0
    for q in queries:
        for uid in q.target_uids:
            if uid not in by_uid:
                orphan += 1
            elif q.question in by_uid[uid].text:
                leaked += 1
    dup = sum(c - 1 for c in Counter((q.document_id, q.question) for q in queries).values() if c > 1)
    if leaked or orphan:
        raise RuntimeError(f"Validation failed: {leaked} queries leak into index, {orphan} orphan targets")
    return {"doc_overlap": 0, "query_text_in_index": leaked, "orphan_targets": orphan, "duplicate_queries": dup}


def stats(docs: list[Document], queries: list[Query]) -> dict:
    n_sec = [len(d.sections) for d in docs]
    depth = [max(s.level for s in d.sections) for d in docs]
    return {
        "documents": len(docs),
        "sections": sum(n_sec),
        "mean_sections_per_doc": sum(n_sec) / max(len(docs), 1),
        "max_depth": max(depth, default=0),
        "mean_depth": sum(depth) / max(len(depth), 1),
        "empty_sections": sum(1 for d in docs for s in d.sections if not s.text),
        "queries": len(queries),
        "query_target_levels": dict(Counter(q.metadata["target_level"] for q in queries)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d = cfg["data"]

    out = {}
    all_docs: dict[str, tuple[list[Document], list[Query]]] = {}
    for name, split, n in (("dev", d["dev_split"], d["dev_documents"]), ("test", d["test_split"], d["test_documents"])):
        raw = list(load_govreport(split, limit=n, seed=cfg["seed"]))
        for doc in raw:
            validate_hierarchy(doc)
        docs, queries = generate_ict_queries(
            raw, split=name, queries_per_doc=d["queries_per_document"], seed=cfg["seed"]
        )
        all_docs[name] = (docs, queries)
        write_jsonl(DATA_DIR / f"{name}_documents.jsonl", docs)
        write_jsonl(DATA_DIR / f"{name}_queries.jsonl", queries)
        out[name] = stats(docs, queries)
        log.info("%s: %s", name, out[name])

    (dev_docs, dev_q), (test_docs, test_q) = all_docs["dev"], all_docs["test"]
    out["validation"] = {
        "dev": check_leakage(dev_docs, test_docs, dev_q, dev_docs),
        "test": check_leakage(dev_docs, test_docs, test_q, test_docs),
    }
    out["source"] = "launch/gov_report (structure config)"
    write_json(DATA_DIR / "dataset_stats.json", out)
    print(f"Saved to {DATA_DIR}\n{out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

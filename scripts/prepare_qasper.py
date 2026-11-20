"""Second domain: QASPER research papers with human-written questions (evaluation only).

Nothing is tuned on QASPER: every method keeps the settings chosen on GovReport dev, so
the results measure transfer to a new domain.

uv run python scripts/prepare_qasper.py
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from structuralrag.ingestion.qasper import load_qasper
from structuralrag.preprocessing.hierarchy_builder import validate_hierarchy
from structuralrag.utils.config import DATA_DIR, load_config
from structuralrag.utils.io import write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    q_cfg = cfg["qasper"]
    docs, queries, totals = [], [], Counter()
    for doc, qs, stats in load_qasper(q_cfg["split"], limit=q_cfg["documents"], seed=cfg["seed"]):
        validate_hierarchy(doc)
        if not qs:
            continue
        docs.append(doc)
        queries += qs
        totals.update(stats)
    by_uid = {s.uid: s for d in docs for s in d.sections}
    assert all(u in by_uid for q in queries for u in q.target_uids), "gold section missing from index"
    stats = {
        "documents": len(docs),
        "sections": len(by_uid),
        "queries": len(queries),
        "max_depth": max(s.level for s in by_uid.values()),
        "queries_with_multiple_gold_sections": sum(len(q.target_section_ids) > 1 for q in queries),
        "query_target_levels": dict(Counter(q.metadata["target_level"] for q in queries)),
        "skipped": dict(totals),
    }
    write_jsonl(DATA_DIR / "qasper_documents.jsonl", docs)
    write_jsonl(DATA_DIR / "qasper_queries.jsonl", queries)
    write_json(DATA_DIR / "qasper_stats.json", stats)
    print(stats)
    for q in queries[:4]:
        print("-", q.question, q.target_section_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

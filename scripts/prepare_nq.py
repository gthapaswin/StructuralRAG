"""Build the Natural Questions domain: Wikipedia section trees + real user questions.

Download (two of six dev shards, ~530 MB) into data/raw/nq/ first:
  curl -L -o data/raw/nq/validation_0.parquet \
    https://huggingface.co/api/datasets/google-research-datasets/natural_questions/parquet/default/validation/0.parquet
uv run python scripts/prepare_nq.py
"""

from __future__ import annotations

import logging
from collections import Counter

from structuralrag.ingestion.natural_questions import load_nq
from structuralrag.utils.config import DATA_DIR
from structuralrag.utils.io import PROJECT_ROOT, write_json, write_jsonl

log = logging.getLogger("prepare_nq")


def main() -> None:
    files = sorted((PROJECT_ROOT / "data/raw/nq").glob("validation_*.parquet"))
    docs, queries, seen_titles, skipped = [], [], set(), 0
    for ex in load_nq(files):
        if ex is None:
            skipped += 1
            continue
        doc, q = ex
        if doc.title in seen_titles:  # one question per page keeps documents independent
            skipped += 1
            continue
        seen_titles.add(doc.title)
        docs.append(doc)
        queries.append(q)
    write_jsonl(DATA_DIR / "nq_documents.jsonl", docs)
    write_jsonl(DATA_DIR / "nq_queries.jsonl", queries)
    stats = {
        "documents": len(docs),
        "sections": sum(len(d.sections) for d in docs),
        "queries": len(queries),
        "skipped_examples": skipped,
        "max_depth": max(s.level for d in docs for s in d.sections),
        "gold_in_lead": sum(q.metadata["lead"] for q in queries),
        "query_target_levels": dict(Counter(str(q.metadata["target_level"]) for q in queries)),
    }
    write_json(DATA_DIR / "nq_stats.json", stats)
    log.info("%s", stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

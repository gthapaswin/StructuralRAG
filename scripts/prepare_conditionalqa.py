"""Build the ConditionalQA domain (held-out fourth domain for the heading study).

Download v1_0/{documents,train,dev}.json from github.com/haitian-sun/ConditionalQA into
data/raw/conditionalqa/ first. Train and dev questions are both used for evaluation only (nothing
is trained on them); the test split's answers are not public.

uv run python scripts/prepare_conditionalqa.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from structuralrag.ingestion.conditionalqa import page_to_document, question_to_query
from structuralrag.utils.config import DATA_DIR
from structuralrag.utils.io import PROJECT_ROOT, write_json, write_jsonl

log = logging.getLogger("prepare_conditionalqa")
RAW = PROJECT_ROOT / "data/raw/conditionalqa"


def main() -> None:
    pages = json.loads((RAW / "documents.json").read_text())
    docs, owners = {}, {}
    for k, p in enumerate(pages):
        doc, owner = page_to_document(p, f"cqa_{k:03d}")
        docs[p["url"]], owners[p["url"]] = doc, owner
    queries, skipped = [], Counter()
    for split in ("train", "dev"):
        for q in json.loads((RAW / f"{split}.json").read_text()):
            if q["url"] not in docs:
                skipped["no_document"] += 1
                continue
            query = question_to_query(q, docs[q["url"]], owners[q["url"]])
            if query is None:
                skipped["unanswerable_or_unmatched"] += 1
                continue
            queries.append(query)
    used = {q.document_id for q in queries}
    kept = [d for d in docs.values() if d.document_id in used]
    write_jsonl(DATA_DIR / "conditionalqa_documents.jsonl", kept)
    write_jsonl(DATA_DIR / "conditionalqa_queries.jsonl", queries)
    stats = {
        "documents": len(kept),
        "sections": sum(len(d.sections) for d in kept),
        "queries": len(queries),
        "skipped": dict(skipped),
        "max_depth": max(s.level for d in kept for s in d.sections),
        "queries_with_multiple_gold_sections": sum(len(q.target_section_ids) > 1 for q in queries),
        "query_target_levels": dict(Counter(str(q.metadata["target_level"]) for q in queries)),
    }
    write_json(DATA_DIR / "conditionalqa_stats.json", stats)
    log.info("%s", stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

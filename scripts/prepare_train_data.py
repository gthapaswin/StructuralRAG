"""Build training data from the GovReport *train* split (for fine-tuning).

Same pipeline as the evaluation data (section trees + ICT queries with the query sentence
removed), on reports that are disjoint from dev and test. Disjointness is checked by
report id and by a fingerprint of the report text, and the script fails if either overlaps.

uv run python scripts/prepare_train_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging

from structuralrag.data_generation.ict_queries import generate_ict_queries
from structuralrag.ingestion.govreport import load_govreport
from structuralrag.preprocessing.hierarchy_builder import validate_hierarchy
from structuralrag.schema import Document
from structuralrag.utils.config import DATA_DIR, load_config
from structuralrag.utils.io import load_documents, write_json, write_jsonl
from structuralrag.utils.reproducibility import set_seed

log = logging.getLogger("prepare_train_data")


def fingerprint(doc: Document) -> str:
    text = " ".join(" ".join(s.text.split()) for s in doc.sections)[:2000].lower()
    return hashlib.sha1(text.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ft = cfg["finetune"]

    held_out = load_documents(DATA_DIR / "dev_documents.jsonl") + load_documents(DATA_DIR / "test_documents.jsonl")
    held_ids = {d.document_id for d in held_out}
    held_fp = {fingerprint(d) for d in held_out}

    raw, dropped = [], 0
    for doc in load_govreport(ft["train_split"], limit=ft["train_documents"], seed=cfg["seed"]):
        validate_hierarchy(doc)
        if doc.document_id in held_ids or fingerprint(doc) in held_fp:
            dropped += 1  # duplicate of an evaluation report: never train on it
            continue
        raw.append(doc)
    docs, queries = generate_ict_queries(
        raw, split="train", queries_per_doc=ft["queries_per_document"], seed=cfg["seed"]
    )

    # Re-check after query generation: nothing we train on may match a held-out report.
    assert not {d.document_id for d in docs} & held_ids
    assert not {fingerprint(d) for d in raw} & held_fp

    write_jsonl(DATA_DIR / "train_documents.jsonl", docs)
    write_jsonl(DATA_DIR / "train_queries.jsonl", queries)
    stats = {
        "documents": len(docs),
        "sections": sum(len(d.sections) for d in docs),
        "queries": len(queries),
        "dropped_as_duplicates_of_eval_reports": dropped,
    }
    write_json(DATA_DIR / "train_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

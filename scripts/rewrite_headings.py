"""Heading study, part 2: an LLM writes new headings for every section from its text alone.

The model sees each section's first words (never the questions) and must write a short,
specific heading. Used two ways: as a repair for documents with uninformative headings, and
as a controlled change of heading informativeness while the text stays the same.

uv run python scripts/rewrite_headings.py [--datasets govreport qasper nq] [--max-docs 200 --nq-docs 250]
Output: data/processed/{dataset}_llm_headings.json  ({document_id: {section_id: heading}})
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.schema import Document
from structuralrag.utils.config import CACHE_DIR, DATA_DIR, load_config
from structuralrag.utils.io import load_documents

log = logging.getLogger("rewrite_headings")
FILES = {"govreport": "test_documents.jsonl", "qasper": "qasper_documents.jsonl", "nq": "nq_documents.jsonl"}
SNIPPET_WORDS = 60

SYSTEM = (
    "You write section headings for documents. For each numbered section, write one short, specific "
    "heading (3 to 8 words) that says what that section is about, based only on its text. Avoid generic "
    "headings such as Introduction, Background, Overview, Discussion or Conclusion unless nothing more "
    "specific fits. Sections marked (heading only) contain only the listed subsections."
)


def prompt(doc: Document) -> str:
    lines = [f"Document: {doc.title}", ""]
    for s in doc.sections:
        words = s.text.split()
        if words:
            body = " ".join(words[:SNIPPET_WORDS]) + (" ..." if len(words) > SNIPPET_WORDS else "")
        else:
            body = "(heading only; subsections " + ", ".join(s.children) + ")"
        lines.append(f"[{s.section_id}] {body}")
    return "\n".join(lines)


def schema(doc: Document) -> dict:
    ids = [s.section_id for s in doc.sections]
    return {
        "type": "object",
        "properties": {i: {"type": "string"} for i in ids},
        "required": ids,
        "additionalProperties": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["govreport", "qasper", "nq"])
    ap.add_argument("--max-docs", type=int, default=200, help="documents per dataset (fixed random sample)")
    ap.add_argument("--nq-docs", type=int, default=250)
    args = ap.parse_args()
    cfg = load_config()
    llm = cfg["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    client.check()
    for ds in args.datasets:
        docs = load_documents(DATA_DIR / FILES[ds])
        limit = args.nq_docs if ds == "nq" else args.max_docs
        if limit < len(docs):
            docs = random.Random(cfg["seed"]).sample(docs, limit)
        out_path = DATA_DIR / f"{ds}_llm_headings.json"
        out = json.loads(out_path.read_text()) if out_path.exists() else {}
        t0 = time.time()
        for n, d in enumerate(docs, 1):
            if d.document_id in out:
                continue
            res = client.chat_json(SYSTEM, prompt(d), schema(d), num_predict=24 * len(d.sections) + 64)
            out[d.document_id] = {s.section_id: str(res.get(s.section_id, "")).strip() or s.title for s in d.sections}
            if n % 10 == 0:
                out_path.write_text(json.dumps(out))
                log.info("%s: %d/%d documents (%.0fs)", ds, n, len(docs), time.time() - t0)
        out_path.write_text(json.dumps(out))
        log.info("%s: done, %d documents", ds, len(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

"""Pre-label frequent headings as functional / boilerplate with the local LLM, for human review.

Writes llm_label and llm_reason into data/annotations/heading_labels.csv and pre-fills `label`
with the model's choice (only where `label` is empty), so the reviewer only changes the rows they
disagree with. The original model label is kept, so the share of changed labels can be reported.

uv run python scripts/prelabel_headings.py
"""

from __future__ import annotations

import logging

import pandas as pd

from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.utils.config import CACHE_DIR, load_config
from structuralrag.utils.io import PROJECT_ROOT

log = logging.getLogger("prelabel_headings")
PATH = PROJECT_ROOT / "data/annotations/heading_labels.csv"
BATCH = 30
DOMAIN = {"govreport": "U.S. government (GAO/CRS) reports", "qasper": "NLP research papers", "nq": "Wikipedia articles"}

SYSTEM = (
    "You label section headings from long documents. For each heading answer:\n"
    "functional: the heading tells what kind of content the section holds, in a way a reader's question "
    "could target, even if many documents share it (e.g. Results, Datasets, Early life, Plot, "
    "Recommendations for Executive Action).\n"
    "boilerplate: a standard administrative or navigational part that questions rarely target, or a "
    "placeholder (e.g. GAO Contact, Staff Acknowledgments, References, See also, External links, Untitled 1).\n"
    "unsure: you cannot decide.\n"
    "Test: if you only saw this heading, would it help guess whether a typical question about this kind "
    "of document is answered in this section? Yes -> functional, no -> boilerplate."
)


def main() -> None:
    cfg = load_config()
    llm = cfg["llm"]
    client = OllamaClient(llm["model"], llm["host"], cache_path=CACHE_DIR / "ollama_cache.sqlite", seed=cfg["seed"])
    df = pd.read_csv(PATH, keep_default_na=False)
    labels, reasons = [""] * len(df), [""] * len(df)
    for ds, g in df.groupby("dataset", sort=False):
        idx = list(g.index)
        for start in range(0, len(idx), BATCH):
            chunk = idx[start : start + BATCH]
            keys = [f"h{i}" for i in range(len(chunk))]
            lines = [
                f'{k}: "{df.at[i, "heading"]}" (in {df.at[i, "documents_with_heading"]} documents; text starts: '
                f"{df.at[i, 'example_text_start'][:160]})"
                for k, i in zip(keys, chunk, strict=True)
            ]
            item = {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["functional", "boilerplate", "unsure"]},
                    "reason": {"type": "string"},
                },
                "required": ["label", "reason"],
            }
            schema = {"type": "object", "properties": dict.fromkeys(keys, item), "required": keys}
            user = f"Documents: {DOMAIN[ds]}.\nHeadings:\n" + "\n".join(lines)
            res = client.chat_json(SYSTEM, user, schema, num_predict=60 * len(chunk) + 64)
            for k, i in zip(keys, chunk, strict=True):
                labels[i] = res.get(k, {}).get("label", "unsure")
                reasons[i] = res.get(k, {}).get("reason", "")
            log.info("%s: labelled %d/%d", ds, min(start + BATCH, len(idx)), len(idx))
    df["llm_label"], df["llm_reason"] = labels, reasons
    df["label"] = [lab or llm for lab, llm in zip(df["label"], labels, strict=True)]
    cols = [
        "dataset",
        "heading",
        "documents_with_heading",
        "share_of_documents",
        "label",
        "llm_label",
        "llm_reason",
        "notes",
        "example_document",
        "example_text_start",
    ]
    df[cols].to_csv(PATH, index=False)
    log.info("%s", df.llm_label.value_counts().to_dict())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()

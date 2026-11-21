"""Build data/annotations/heading_labels.csv: the most frequent headings per dataset, to label.

uv run python scripts/make_heading_sheet.py [--top 70]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from structuralrag.preprocessing.text_cleaner import tokenize
from structuralrag.utils.config import DATA_DIR
from structuralrag.utils.io import PROJECT_ROOT, load_documents

FILES = {"govreport": "test_documents.jsonl", "qasper": "qasper_documents.jsonl", "nq": "nq_documents.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=70)
    args = ap.parse_args()
    rows = []
    for ds, f in FILES.items():
        docs = load_documents(DATA_DIR / f)
        seen, example = defaultdict(set), {}
        for d in docs:
            for s in d.sections:
                key = " ".join(tokenize(s.title))
                if not key:
                    continue
                seen[key].add(d.document_id)
                if key not in example and len(s.text.split()) >= 10:
                    example[key] = (s.title, d.title, " ".join(s.text.split()[:30]))
        common = [(k, v) for k, v in sorted(seen.items(), key=lambda kv: -len(kv[1])) if len(v) >= 2][: args.top]
        for key, ids in common:
            title, doc_title, text = example.get(key, (key, "", ""))
            rows.append(
                {
                    "dataset": ds,
                    "heading": title,
                    "documents_with_heading": len(ids),
                    "share_of_documents": round(len(ids) / len(docs), 3),
                    "label": "",
                    "notes": "",
                    "example_document": doc_title,
                    "example_text_start": text,
                }
            )
    out = PROJECT_ROOT / "data/annotations/heading_labels.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows), "headings ->", out)


if __name__ == "__main__":
    main()

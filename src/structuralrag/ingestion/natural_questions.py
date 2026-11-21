"""Adapter for Natural Questions (Kwiatkowski et al., TACL 2019): real Google queries over
Wikipedia pages. A third domain with human-written questions and very varied headings.

The page's section tree is rebuilt from its HTML heading tokens (<H2> = level 1,
<H3> = level 2, <H4> = level 3); text before the first <H2> is the lead section. The gold
section of a question is the section containing the long answer chosen by most of the
annotators who gave one (the dev split has five annotators; at least two must answer,
the usual NQ rule for counting a question as answerable).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.schema import Document, Query

log = logging.getLogger(__name__)

LEAD = "Introduction"
HEADING = re.compile(r"^<H([2-4])>$", re.IGNORECASE)
EDIT_LINK = re.compile(r"\s*[(\[]\s*edit\s*[)\]]\s*", re.IGNORECASE)  # Wikipedia "[edit]" link text
MIN_ANSWERS = 2


def parse_page(tokens: list[str], is_html: list[bool]) -> tuple[list[FlatHeading], list[int]]:
    """Return headings (lead first) and, per heading, the token index where its section starts."""
    heads = [FlatHeading(title=LEAD, depth=1, text="")]
    starts, texts = [0], [[]]
    i, n = 0, len(tokens)
    while i < n:
        tok, html = tokens[i], is_html[i]
        m = HEADING.match(tok) if html else None
        if m:
            close = f"</H{m.group(1)}>".upper()
            j, title = i + 1, []
            while j < n and not (is_html[j] and tokens[j].upper() == close):
                if not is_html[j]:
                    title.append(tokens[j])
                j += 1
            clean = EDIT_LINK.sub(" ", " ".join(title)).strip()
            heads.append(FlatHeading(title=clean or "Untitled", depth=int(m.group(1)) - 1, text=""))
            starts.append(i)
            texts.append([])
            i = j + 1
            continue
        if not html:
            texts[-1].append(tok)
        i += 1
    heads = [FlatHeading(title=h.title, depth=h.depth, text=" ".join(t)) for h, t in zip(heads, texts, strict=True)]
    # Drop the page's table-of-contents box (navigation, not content); tokens inside it map to
    # the preceding section.
    keep = [i for i, h in enumerate(heads) if not (i > 0 and h.title.lower() == "contents")]
    return [heads[i] for i in keep], [starts[i] for i in keep]


def _section_of(token: int, starts: list[int]) -> int:
    k = 0
    for j, s in enumerate(starts):
        if s <= token:
            k = j
    return k


def row_to_example(row: dict, min_sections: int = 4) -> tuple[Document, Query] | None:
    tok = row["document"]["tokens"]
    heads, starts = parse_page(tok["token"], tok["is_html"])
    if sum(1 for h in heads if len(h.text.split()) >= 5) < min_sections:
        return None
    answers = [a["start_token"] for a in row["annotations"]["long_answer"] if a["start_token"] >= 0]
    if len(answers) < MIN_ANSWERS:
        return None
    sec = Counter(_section_of(t, starts) for t in answers).most_common(1)[0][0]
    doc_id = f"nq_{row['id']}"
    doc = build_document(
        dataset="nq",
        document_id=doc_id,
        title=row["document"]["title"],
        headings=heads,
        metadata={"source": "google-research-datasets/natural_questions", "url": row["document"]["url"]},
    )
    if len(doc.sections) != len(heads):
        return None
    gold = doc.sections[sec]
    if len(gold.text.split()) < 5:
        return None
    query = Query(
        query_id=f"{doc_id}::q",
        dataset="nq",
        document_id=doc_id,
        question=row["question"]["text"].strip(),
        target_section_ids=[gold.section_id],
        split="validation",
        generator="human",
        metadata={"target_level": gold.level, "lead": sec == 0},
    )
    return doc, query


def load_nq(files: list[Path]) -> Iterator[tuple[Document, Query] | None]:
    import pyarrow.parquet as pq

    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=["id", "document", "question", "annotations"]).to_pylist():
                yield row_to_example(row)


def norm_answer(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def short_answers(files: list[Path], wanted: set[str]) -> dict[str, list[str]]:
    """Normalized short-answer strings per NQ example id (only for `wanted` ids)."""
    import pyarrow.parquet as pq

    out: dict[str, list[str]] = {}
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=["id", "document", "annotations"]).to_pylist():
                if row["id"] not in wanted:
                    continue
                toks, html = row["document"]["tokens"]["token"], row["document"]["tokens"]["is_html"]
                answers = set()
                for sa in row["annotations"]["short_answers"]:
                    for s, e in zip(sa["start_token"], sa["end_token"], strict=True):
                        text = " ".join(t for t, h in zip(toks[s:e], html[s:e], strict=True) if not h)
                        if norm_answer(text):
                            answers.add(norm_answer(text))
                out[row["id"]] = sorted(answers)
    return out

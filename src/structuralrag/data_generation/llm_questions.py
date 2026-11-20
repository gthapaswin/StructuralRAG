"""LLM-written questions (synthetic QA, plan Stage 5) with validation.

ICT queries are sentences lifted from the section, which favours lexical matching.
These questions are written by an LLM that is told to paraphrase, so they test
retrieval when the question does not reuse the section's wording.

Validation (all must pass):
1. format: a single question, ends with '?', 6-30 words, no section numbers;
2. no copying: shares no 5-word span with the section;
3. answerable: a second, independent LLM call confirms the section answers it.
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter

from structuralrag.llm.ollama_client import OllamaClient, OllamaError
from structuralrag.preprocessing.text_cleaner import split_sentences, tokenize
from structuralrag.schema import Document, Query

log = logging.getLogger(__name__)

MIN_SENTENCES = 3
MAX_SECTION_WORDS = 450
MAX_COPY_SPAN = 5

GEN_SYSTEM = (
    "You write realistic questions that a policy analyst might ask about a government report. "
    "Write ONE question that the given section answers. Paraphrase: do not copy phrases from the "
    "section, do not mention section numbers or 'this section', and keep it under 25 words."
)
CHECK_SYSTEM = (
    "You check question-answer pairs. Answer 'yes' only if the passage directly contains the "
    "information needed to answer the question; otherwise answer 'no'."
)
Q_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"answerable": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answerable"],
}


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def check_format(q: str) -> str | None:
    """Return a rejection reason, or None if the question is well-formed."""
    words = q.split()
    if not q.endswith("?") or q.count("?") != 1:
        return "not_single_question"
    if not 6 <= len(words) <= 30:
        return "length"
    if re.search(r"\bsection\b|\b\d+\.\d+\b", q, re.IGNORECASE):
        return "mentions_section"
    return None


def copies_text(question: str, section_text: str, span: int = MAX_COPY_SPAN) -> bool:
    q = tokenize(question, remove_stopwords=False)
    s = tokenize(section_text, remove_stopwords=False)
    return bool(_ngrams(q, span) & _ngrams(s, span))


def _passage(text: str) -> str:
    return " ".join(text.split()[:MAX_SECTION_WORDS])


def generate_llm_questions(
    docs: list[Document],
    client: OllamaClient,
    split: str,
    per_doc: int,
    seed: int,
) -> tuple[list[Query], dict[str, int]]:
    rng = random.Random(seed)
    queries: list[Query] = []
    stats: Counter[str] = Counter()
    for doc in docs:
        candidates = [s for s in doc.sections if len(split_sentences(s.text)) >= MIN_SENTENCES]
        rng.shuffle(candidates)
        made = 0
        for sec in candidates:
            if made >= per_doc:
                break
            passage = _passage(sec.text)
            user = f"Report section titled '{sec.title}':\n\n{passage}\n\nWrite the question."
            try:
                q = client.chat_json(GEN_SYSTEM, user, Q_SCHEMA, num_predict=80)["question"].strip()
            except (OllamaError, KeyError) as e:
                stats["error"] += 1
                log.warning("Generation failed for %s: %s", sec.uid, e)
                continue
            stats["generated"] += 1
            if reason := check_format(q):
                stats[f"rejected_{reason}"] += 1
                continue
            if copies_text(q, sec.text):
                stats["rejected_copies_text"] += 1
                continue
            verdict = client.chat_json(
                CHECK_SYSTEM, f"Passage:\n{passage}\n\nQuestion: {q}", CHECK_SCHEMA, num_predict=16
            )
            if verdict.get("answerable") != "yes":
                stats["rejected_unanswerable"] += 1
                continue
            stats["accepted"] += 1
            made += 1
            queries.append(
                Query(
                    query_id=f"{doc.document_id}::llm{len(queries)}",
                    dataset=doc.dataset,
                    document_id=doc.document_id,
                    question=q,
                    target_section_ids=[sec.section_id],
                    split=split,
                    generator=f"llm:{client.model}",
                    metadata={"target_level": sec.level},
                )
            )
    return queries, dict(stats)

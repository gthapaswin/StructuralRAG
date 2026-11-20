"""Text normalisation and sentence splitting."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_TOKEN = re.compile(r"[a-z0-9]+")
# Common abbreviations in government reports that should not end a sentence.
_ABBREV = ("U.S.", "U.N.", "e.g.", "i.e.", "No.", "Fig.", "fig.", "Dr.", "Mr.", "Ms.", "Inc.", "et al.")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "which",
        "will",
        "with",
        "within",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "our",
        "we",
        "they",
        "them",
        "than",
        "then",
        "also",
        "not",
        "no",
        "such",
    ]
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen from PDF extraction
    return _WS.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, f"\x00{i}\x00")
    parts = _SENT_SPLIT.split(protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(f"\x00{i}\x00", ab)
        p = p.strip()
        if p:
            out.append(p)
    return out


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks

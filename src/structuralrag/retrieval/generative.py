"""Generative retrieval conditioned on the table of contents (STAIR-style, zero-shot).

The LLM sees only the document's table of contents (section ids + titles, optionally a
short snippet) and the question, and generates a ranked list of section ids. Output is
constrained to the document's real ids, so every prediction is valid by construction.

Sections the model does not list get score -inf and are excluded from the ranking, so
the method is never credited for positions it did not predict.
"""

from __future__ import annotations

import numpy as np

from structuralrag.llm.ollama_client import OllamaClient
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.schema import Document

SYSTEM = (
    "You route questions to sections of a long government report. Using only the table of "
    "contents, return the ids of the sections most likely to contain the answer, best first. "
    "Prefer the most specific (deepest) section that fits."
)
SNIPPET_WORDS = 20


def render_toc(doc: Document, snippets: bool = False) -> str:
    lines = []
    for s in doc.sections:
        line = f"{'  ' * (s.level - 1)}{s.section_id} {s.title}"
        if snippets and s.text:
            line += f" — {' '.join(s.text.split()[:SNIPPET_WORDS])}..."
        lines.append(line)
    return "\n".join(lines)


def toc_schema(doc: Document, max_ids: int) -> dict:
    """JSON schema whose enum is the report's real section ids (constrained decoding)."""
    return {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {"type": "string", "enum": [s.section_id for s in doc.sections]},
                "minItems": 1,
                "maxItems": max_ids,
            }
        },
        "required": ["sections"],
    }


def toc_prompt(toc: str, query: str, max_ids: int) -> str:
    """User message. ToC first, question last: queries on one report share a cached prefix."""
    return f"Table of contents:\n{toc}\n\nQuestion: {query}\n\nReturn up to {max_ids} section ids, best first."


class GenerativeRetriever(Retriever):
    requires_document = True  # needs the document's ToC; corpus-wide search is not defined

    def __init__(self, corpus: SectionCorpus, client: OllamaClient, max_ids: int = 5, snippets: bool = False):
        super().__init__(corpus)
        self.client = client
        self.max_ids = max_ids
        self.snippets = snippets
        self.name = f"generative[{client.model}{', toc+snippets' if snippets else ''}]"
        self._toc = {d: render_toc(doc, snippets) for d, doc in corpus.docs.items()}

    def predict_ids(self, query: str, document_id: str) -> list[str]:
        doc = self.corpus.docs[document_id]
        valid = {s.section_id for s in doc.sections}
        user = toc_prompt(self._toc[document_id], query, self.max_ids)
        out = self.client.chat_json(SYSTEM, user, toc_schema(doc, self.max_ids), num_predict=24 * self.max_ids)
        seen: list[str] = []
        for sid in out.get("sections", []):
            # Ollama's grammar guarantees valid ids; a backend without one (MLX) may not, so check.
            if sid in valid and sid not in seen:
                seen.append(sid)
        return seen[: self.max_ids]

    def score_for_document(self, query: str, document_id: str) -> np.ndarray:
        scores = np.full(len(self.corpus), -np.inf)
        for rank, sid in enumerate(self.predict_ids(query, document_id), start=1):
            scores[self.corpus.uid_to_idx[f"{document_id}::{sid}"]] = 1.0 / rank
        return scores

    def score(self, query: str) -> np.ndarray:
        raise NotImplementedError("Generative retrieval needs a document_id; use score_for_document().")

    def search(self, query: str, top_k: int = 5, document_id: str | None = None):
        if document_id is None:
            raise ValueError("Generative retrieval requires a document_id")
        return self._to_results(self.rank(self.score_for_document(query, document_id), top_k, document_id))


class GenerativeFusionRetriever(Retriever):
    """Structure-aware ranking re-ordered by the LLM's picks (RRF-style fusion).

    score = 1/(k + rank_structure) + w * 1/(k + rank_llm), the LLM term only for ids it listed.
    Keeps a full ranking (unlike pure generative) while letting the LLM promote sections.
    """

    requires_document = True

    def __init__(self, generative: GenerativeRetriever, base: Retriever, k: int = 60, weight: float = 1.0):
        super().__init__(base.corpus)
        self.generative, self.base, self.k, self.weight = generative, base, k, weight
        self.name = f"generative+{base.name}"

    def score_for_document(self, query: str, document_id: str) -> np.ndarray:
        idx = self.corpus.doc_indices(document_id)
        base = self.base.score_batch([query])[0]
        fused = np.full(len(self.corpus), -np.inf)
        order = idx[np.argsort(-base[idx], kind="stable")]
        fused[order] = 1.0 / (self.k + np.arange(1, len(order) + 1))
        gen = self.generative.score_for_document(query, document_id)
        listed = np.isfinite(gen)
        gen_rank = np.round(1.0 / gen[listed])
        fused[listed] += self.weight / (self.k + gen_rank)
        return fused

    def score(self, query: str) -> np.ndarray:
        raise NotImplementedError("Needs a document_id; use score_for_document().")

    def search(self, query: str, top_k: int = 5, document_id: str | None = None):
        if document_id is None:
            raise ValueError("Generative fusion requires a document_id")
        return self._to_results(self.rank(self.score_for_document(query, document_id), top_k, document_id))

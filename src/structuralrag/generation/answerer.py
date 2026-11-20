"""Grounded answer generation: cited sentences, claim checking and abstention.

1. Generate: the LLM answers from the evidence only, as a list of sentences, each
   citing section ids. Citations are constrained (JSON-schema enum) to the ids that
   are actually in the evidence, so the model cannot cite a section it did not see.
2. Check: a *different* LLM judges each sentence against the text of the sections
   it cites; unsupported sentences are removed (and reported, not hidden).
3. Abstain: if the generator says the evidence does not answer the question, or no
   sentence survives the check, return "not covered" instead of guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from structuralrag.generation.context import Context
from structuralrag.llm.ollama_client import OllamaClient

MAX_SENTENCES = 4

GEN_SYSTEM = (
    "You answer questions about a government report using ONLY the evidence provided. "
    "Each evidence block starts with a section id in brackets. Write a short answer of 1-4 "
    "sentences; every sentence must cite the id(s) of the block(s) that support it. Do not use "
    "outside knowledge. If the evidence does not contain the answer, set answerable to false "
    "and return no sentences."
)
CHECK_SYSTEM = (
    "You verify claims against evidence. Answer 'supported' only if every fact in the claim is "
    "stated in or directly implied by the evidence. Otherwise answer 'unsupported'."
)
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["supported", "unsupported"]}},
    "required": ["verdict"],
}
NOT_COVERED = "The retrieved sections of this report do not answer this question."


@dataclass
class Sentence:
    text: str
    citations: list[str]
    supported: bool | None = None  # None = not checked


@dataclass
class Answer:
    question: str
    sentences: list[Sentence] = field(default_factory=list)  # kept sentences
    removed: list[Sentence] = field(default_factory=list)  # failed the claim check
    abstained: bool = False
    abstain_reason: str | None = None
    context_mode: str = ""

    @property
    def text(self) -> str:
        if self.abstained:
            return NOT_COVERED
        return " ".join(f"{s.text} [{', '.join(s.citations)}]" for s in self.sentences)

    @property
    def citations(self) -> list[str]:
        return sorted({c for s in self.sentences for c in s.citations})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text
        return d


def answer_schema(citable_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "sentences": {
                "type": "array",
                "maxItems": MAX_SENTENCES,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {"type": "string", "enum": citable_ids},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                    },
                    "required": ["text", "citations"],
                },
            },
        },
        "required": ["answerable", "sentences"],
    }


class Answerer:
    def __init__(self, generator: OllamaClient, checker: OllamaClient | None = None):
        self.generator = generator
        self.checker = checker

    def draft(self, question: str, context: Context) -> Answer:
        ans = Answer(question=question, context_mode=context.mode)
        if not context.blocks:
            ans.abstained, ans.abstain_reason = True, "no_evidence"
            return ans
        user = f"Evidence:\n{context.text}\n\nQuestion: {question}"
        out = self.generator.chat_json(GEN_SYSTEM, user, answer_schema(context.citable_ids), num_predict=400)
        for s in out.get("sentences", []):
            text = s.get("text", "").strip()
            cites = list(dict.fromkeys(c for c in s.get("citations", []) if c in context.citable_ids))
            if text and cites:
                ans.sentences.append(Sentence(text, cites))
        if not out.get("answerable", False) or not ans.sentences:
            ans.sentences = []
            ans.abstained, ans.abstain_reason = True, "generator"
        return ans

    def check(self, ans: Answer, context: Context) -> Answer:
        """Remove sentences the checker model judges unsupported by their cited evidence."""
        if self.checker is None or ans.abstained:
            return ans
        kept: list[Sentence] = []
        for s in ans.sentences:
            evidence = "\n\n".join(f"[{c}] {context.body(c)}" for c in s.citations)
            verdict = self.checker.chat_json(
                CHECK_SYSTEM, f"Evidence:\n{evidence}\n\nClaim: {s.text}", CHECK_SCHEMA, num_predict=16
            )
            s.supported = verdict.get("verdict") == "supported"
            (kept if s.supported else ans.removed).append(s)
        ans.sentences = kept
        if not kept:
            ans.abstained, ans.abstain_reason = True, "all_claims_unsupported"
        return ans

    def answer(self, question: str, context: Context, check: bool = True) -> Answer:
        ans = self.draft(question, context)
        return self.check(ans, context) if check else ans

"""Answer-quality metrics, judged independently of the models that produced the answer.

Faithfulness uses an NLI cross-encoder (not an LLM), so neither the generator
(qwen3) nor the in-pipeline claim checker (llama3.1) grades its own work. Long
evidence is split into overlapping sentence windows; a claim counts as supported
if any window entails it (SummaC-style zero-shot aggregation).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from structuralrag.generation.answerer import Answer
from structuralrag.generation.context import Context
from structuralrag.preprocessing.text_cleaner import split_sentences
from structuralrag.utils.reproducibility import best_device

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
WINDOW, STRIDE = 3, 2


@lru_cache(maxsize=1)
def _load(model_name: str):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=best_device())
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return model, labels["entailment"]


def windows(text: str) -> list[str]:
    sents = split_sentences(text) or [text]
    if len(sents) <= WINDOW:
        return [" ".join(sents)]
    return [" ".join(sents[i : i + WINDOW]) for i in range(0, len(sents) - WINDOW + STRIDE, STRIDE)]


class NLIJudge:
    def __init__(self, model_name: str = NLI_MODEL):
        self.model_name = model_name

    def entailment(self, premise: str, hypothesis: str) -> float:
        """Max entailment probability of the hypothesis over windows of the premise."""
        model, ent = _load(self.model_name)
        wins = windows(premise)
        logits = np.asarray(model.predict([(w, hypothesis) for w in wins], show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return float(probs[:, ent].max())

    def supported(self, premise: str, hypothesis: str, threshold: float = 0.5) -> bool:
        return self.entailment(premise, hypothesis) >= threshold


def _related(a: str, b: str) -> bool:
    """Same section, or one is an ancestor of the other."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def answer_metrics(ans: Answer, context: Context, gold_section: str | None, judge: NLIJudge) -> dict:
    row: dict = {
        "abstained": float(ans.abstained),
        "n_sentences": len(ans.sentences),
        "n_removed": len(ans.removed),
        "context_words": context.n_words,
        "gold_in_context": float(gold_section in context.citable_ids) if gold_section else float("nan"),
    }
    if ans.abstained:
        row.update(faithfulness=np.nan, citation_hit=np.nan, citation_hit_branch=np.nan, citation_precision=np.nan)
        return row
    support = [judge.supported("\n".join(context.body(c) for c in s.citations), s.text) for s in ans.sentences]
    row["faithfulness"] = float(np.mean(support))
    if gold_section:
        cites = ans.citations
        row["citation_hit"] = float(gold_section in cites)
        row["citation_hit_branch"] = float(any(_related(c, gold_section) for c in cites))
        row["citation_precision"] = float(np.mean([_related(c, gold_section) for c in cites]))
    return row

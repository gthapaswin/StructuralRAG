import numpy as np
import pytest

from structuralrag.data_generation.llm_questions import check_format, copies_text, generate_llm_questions
from structuralrag.evaluation.evaluator import evaluate
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.generative import GenerativeFusionRetriever, GenerativeRetriever, render_toc
from structuralrag.schema import Query


class FakeClient:
    """Returns canned JSON; records the prompts and schemas it was given."""

    model = "fake"
    host = "http://fake"

    def __init__(self, sections=None, question="Which metric captures false positives in model evaluation?"):
        self.sections = sections or ["3.2", "3.2", "3"]
        self.question = question
        self.calls = []

    def chat_json(self, system, user, schema, num_predict=128):
        self.calls.append((system, user, schema))
        props = schema["properties"]
        if "sections" in props:
            return {"sections": self.sections}
        if "question" in props:
            return {"question": self.question}
        return {"answerable": "yes"}


def test_toc_rendering(ml_doc):
    toc = render_toc(ml_doc)
    assert "  2.1 Supervised Learning" in toc and "Classification predicts" not in toc
    assert "Classification predicts" in render_toc(ml_doc, snippets=True)


def test_generative_constrained_and_deduplicated(corpus):
    client = FakeClient()
    gen = GenerativeRetriever(corpus, client)
    ids = gen.predict_ids("precision?", "ml_handbook")
    assert ids == ["3.2", "3"]
    schema = client.calls[0][2]
    assert set(schema["properties"]["sections"]["items"]["enum"]) == {s.section_id for s in corpus.sections}
    system, user, _ = client.calls[0]
    assert user.index("Table of contents") < user.index("Question"), "ToC must precede the question"


def test_unpredicted_sections_are_not_ranked(corpus):
    gen = GenerativeRetriever(corpus, FakeClient())
    res = gen.search("precision?", top_k=10, document_id="ml_handbook")
    assert [r.section_id for r in res] == ["3.2", "3"]
    with pytest.raises(ValueError):
        gen.search("x")


def test_fusion_keeps_full_ranking_and_promotes_llm_pick(corpus):
    base = BM25Retriever(corpus)
    fusion = GenerativeFusionRetriever(GenerativeRetriever(corpus, FakeClient(["2.2"])), base)
    res = fusion.search("fraction of correct predictions", top_k=10, document_id="ml_handbook")
    assert len(res) == len(corpus)
    base_rank = [r.section_id for r in base.search("fraction of correct predictions", 10)].index("2.2")
    assert [r.section_id for r in res].index("2.2") < base_rank


def test_evaluator_supports_generative(corpus):
    q = Query(
        query_id="q",
        dataset="toy",
        document_id="ml_handbook",
        question="precision?",
        target_section_ids=["3.2"],
        split="test",
        generator="test",
    )
    s = evaluate(GenerativeRetriever(corpus, FakeClient()), [q], latency_sample=0).summary()
    assert s["recall@1"] == 1.0 and s["mrr@10"] == 1.0
    with pytest.raises(ValueError):
        evaluate(GenerativeRetriever(corpus, FakeClient()), [q], scope="corpus")


def test_question_validation():
    assert check_format("What does the agency plan to change next year?") is None
    assert check_format("Tell me about it.") == "not_single_question"
    assert check_format("What is in section 2.1 of the report today?") == "mentions_section"
    text = "Precision measures false positives while recall measures false negatives."
    assert copies_text("Why is it that precision measures false positives while recall differs?", text)
    assert not copies_text("Which metric reflects mistaken positive predictions?", text)


def test_llm_question_pipeline(ml_doc):
    ml_doc.sections[-1].text = "One sentence here. Another one follows. A third closes it."
    qs, stats = generate_llm_questions([ml_doc], FakeClient(), "test", per_doc=1, seed=0)
    assert len(qs) == 1 and stats["accepted"] == 1
    assert qs[0].target_section_ids == ["3.2"] and qs[0].generator == "llm:fake"
    assert np.isfinite(len(qs[0].question))

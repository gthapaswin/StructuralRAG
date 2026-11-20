import numpy as np

from structuralrag.llm.mlx_client import ids_in_text, parse_json
from structuralrag.retrieval.base import Retriever
from structuralrag.retrieval.dense import query_prefix
from structuralrag.retrieval.generative import GenerativeRetriever
from structuralrag.retrieval.reranker import RerankRetriever, passage
from structuralrag.schema import Query
from structuralrag.training.hard_negatives import mine_hard_negatives, related


def q(sid, text="q"):
    return Query(
        query_id="x",
        dataset="toy",
        document_id="ml_handbook",
        question=text,
        target_section_ids=[sid],
        split="train",
        generator="ict",
    )


def test_related_uses_dotted_ids():
    assert related("2.1", "2.1.1") and related("2.1.1", "2") and not related("2.1", "2.10")


def test_hard_negatives_skip_gold_family_and_empty_sections(corpus):
    gold = "2.1.1"
    scores = np.arange(len(corpus), dtype=float)[None, :]  # later sections score higher
    negs = mine_hard_negatives(corpus, [q(gold)], scores, n=10)[0]
    ids = [corpus.sections[i].section_id for i in negs]
    assert all(not related(i, gold) for i in ids)
    assert all(corpus.sections[i].text for i in negs)  # heading-only sections 2 and 3 excluded
    assert ids == sorted(ids, key=lambda s: -corpus.uid_to_idx[f"ml_handbook::{s}"])  # hardest first


def test_query_prefix_applies_to_finetuned_bge_paths():
    assert query_prefix("BAAI/bge-small-en-v1.5") == query_prefix("artifacts/models/bge-small-govreport") != ""
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_parse_json_tolerates_surrounding_text():
    assert parse_json('Sure: {"sections": ["1", "2"]} done') == {"sections": ["1", "2"]}
    assert parse_json("no json here") == {} and parse_json('{"sections": [') == {}


class UnconstrainedClient:
    """Like MLX output: may contain ids that do not exist."""

    model, host = "fake", "mlx"

    def chat_json(self, system, user, schema, num_predict=128):
        return {"sections": ["9.9", "3.2", "3.2", "1"]}


def test_generative_drops_invalid_ids(corpus):
    gen = GenerativeRetriever(corpus, UnconstrainedClient())
    assert gen.predict_ids("q", "ml_handbook") == ["3.2", "1"]


class FixedRetriever(Retriever):
    name = "fixed"

    def score(self, query):
        return np.arange(len(self.corpus), dtype=float)


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return [1.0 if "Accuracy" in p else 0.0 for _, p in pairs]


def test_reranker_rescores_only_top_n(corpus, monkeypatch):
    r = RerankRetriever(FixedRetriever(corpus), "fake", top_n=3)
    monkeypatch.setattr(RerankRetriever, "model", property(lambda self: FakeCrossEncoder()))
    s = r.score_for_document("what is accuracy", "ml_handbook")
    assert np.isfinite(s).sum() == 3
    top = r.search("what is accuracy", top_k=1, document_id="ml_handbook")[0]
    assert top.title == "Accuracy"
    assert passage(corpus.sections[corpus.uid_to_idx["ml_handbook::3.1"]]) == (
        "Model Evaluation > Accuracy\nAccuracy is the fraction of predictions that are correct."
    )


def test_ids_in_text_skips_list_markers_and_numbers():
    text = "1. **2.1 New Standards**\n2. **2.2 Restatements** in 2017\n### Best first:\n4.1, 4.2 and 2.1."
    assert ids_in_text(text) == ["2.1", "2.2", "2017", "4.1", "4.2"]


class EmptyClient(UnconstrainedClient):
    def chat_json(self, system, user, schema, num_predict=128):
        return {"sections": ["9.9"]}


def test_evaluator_scores_empty_output_as_miss(corpus):
    from structuralrag.evaluation.evaluator import evaluate

    r = evaluate(GenerativeRetriever(corpus, EmptyClient()), [q("3.1")], latency_sample=0)
    row = r.per_query[0]
    assert row["top1"] is None and row["mrr@10"] == 0.0 and row["n_returned"] == 0

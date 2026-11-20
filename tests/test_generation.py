import pytest

from structuralrag.evaluation.answer_metrics import _related, answer_metrics, windows
from structuralrag.generation.answerer import NOT_COVERED, Answerer, answer_schema
from structuralrag.generation.context import build_context


def idx(corpus, *sids):
    return [corpus.uid_to_idx[f"ml_handbook::{s}"] for s in sids]


class FakeGen:
    model = "fake-gen"

    def __init__(self, out):
        self.out, self.calls = out, []

    def chat_json(self, system, user, schema, num_predict=128):
        self.calls.append((user, schema))
        return self.out


class FakeChecker:
    model = "fake-check"

    def __init__(self, unsupported_words=("invented",)):
        self.bad = unsupported_words

    def chat_json(self, system, user, schema, num_predict=16):
        claim = user.split("Claim:")[-1]
        return {"verdict": "unsupported" if any(w in claim for w in self.bad) else "supported"}


class FakeJudge:
    def supported(self, premise, hypothesis):
        return hypothesis.split()[0].lower() in premise.lower()


def test_plain_context_is_body_text_only(corpus):
    ctx = build_context(corpus, idx(corpus, "3.2", "2"), "plain")
    assert ctx.citable_ids == ["3.2"]  # heading-only section 2 has no body text
    assert "Model Evaluation" not in ctx.text


def test_hierarchical_context_adds_path_parent_and_children(corpus):
    ctx = build_context(corpus, idx(corpus, "2.1.1", "2"), "hierarchical")
    assert ctx.citable_ids[0] == "2.1.1"
    assert "Machine Learning › Supervised Learning › Classification" in ctx.text
    assert "Context from 2.1 Supervised Learning" in ctx.text
    # Heading-only section 2 expands into its children; 2.1.1 is not repeated.
    assert set(ctx.citable_ids) == {"2.1.1", "2.1", "2.2"}
    assert len(ctx.citable_ids) == len(set(ctx.citable_ids))


def test_word_budget_is_respected(corpus):
    all_idx = list(range(len(corpus)))
    for mode in ("plain", "hierarchical"):
        assert build_context(corpus, all_idx, mode, budget=40).n_words <= 40


def test_context_rejects_unknown_mode(corpus):
    with pytest.raises(ValueError):
        build_context(corpus, [0], "fancy")


def test_citations_constrained_to_context():
    schema = answer_schema(["2.1", "3"])
    assert schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"] == ["2.1", "3"]


def test_answer_with_check_removes_unsupported(corpus):
    ctx = build_context(corpus, idx(corpus, "3.1", "3.2"), "hierarchical")
    gen = FakeGen(
        {
            "answerable": True,
            "sentences": [
                {"text": "Accuracy is the fraction of correct predictions.", "citations": ["3.1"]},
                {"text": "It was invented in 1950.", "citations": ["3.1", "9.9"]},
            ],
        }
    )
    ans = Answerer(gen, FakeChecker()).answer("What is accuracy?", ctx)
    assert [s.text for s in ans.sentences] == ["Accuracy is the fraction of correct predictions."]
    assert ans.removed[0].citations == ["3.1"], "ids outside the context are dropped"
    assert ans.text.endswith("[3.1]") and not ans.abstained


def test_abstains_when_generator_says_unanswerable(corpus):
    ctx = build_context(corpus, idx(corpus, "3.1"), "plain")
    ans = Answerer(FakeGen({"answerable": False, "sentences": []})).answer("Who won the match?", ctx)
    assert ans.abstained and ans.abstain_reason == "generator" and ans.text == NOT_COVERED


def test_abstains_when_every_claim_fails_check(corpus):
    ctx = build_context(corpus, idx(corpus, "3.1"), "plain")
    gen = FakeGen({"answerable": True, "sentences": [{"text": "An invented fact.", "citations": ["3.1"]}]})
    ans = Answerer(gen, FakeChecker()).answer("q", ctx)
    assert ans.abstained and ans.abstain_reason == "all_claims_unsupported" and len(ans.removed) == 1


def test_answer_metrics(corpus):
    ctx = build_context(corpus, idx(corpus, "3.1", "3.2"), "plain")
    gen = FakeGen(
        {
            "answerable": True,
            "sentences": [
                {"text": "Accuracy measures correct predictions.", "citations": ["3.1"]},
                {"text": "Bananas are yellow.", "citations": ["3.2"]},
            ],
        }
    )
    ans = Answerer(gen).answer("q", ctx, check=False)
    m = answer_metrics(ans, ctx, "3.1", FakeJudge())
    assert m["faithfulness"] == 0.5 and m["citation_hit"] == 1.0 and m["citation_precision"] == 0.5
    assert m["gold_in_context"] == 1.0 and m["abstained"] == 0.0


def test_related_and_windows():
    assert _related("2.1", "2.1.3") and _related("2.1.3", "2.1") and not _related("2.1", "2.10")
    text = "A one. B two. C three. D four. E five. F six."
    assert all(len(w.split(".")) <= 4 for w in windows(text)) and len(windows(text)) >= 2

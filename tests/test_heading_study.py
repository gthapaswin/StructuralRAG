import numpy as np
from conftest import FakeEncoder

from structuralrag.analysis import experiment as ex
from structuralrag.analysis import headings as hd
from structuralrag.analysis.informativeness import document_features, section_features
from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.retrieval.corpus import SectionCorpus
from structuralrag.schema import Query


def other_doc():
    return build_document(
        "toy", "other", "Other", [FlatHeading("Budget", 1, "Costs rose."), FlatHeading("Staffing", 2, "Staff fell.")]
    )


def test_retitle_rebuilds_paths_and_keeps_text(ml_doc):
    new = hd.retitle(ml_doc, {"2": "ML"})
    child = new.section("2.1")
    assert child.path == ["ML", "Supervised Learning"]
    assert [s.text for s in new.sections] == [s.text for s in ml_doc.sections]


def test_placebo_takes_titles_from_other_documents_at_same_depth(ml_doc):
    docs = hd.placebo([ml_doc, other_doc()], seed=1)
    own = {s.title for s in ml_doc.sections}
    for s in docs[0].sections:
        if s.level <= 2:
            assert s.title not in own


def test_genericize_and_flatten(ml_doc):
    g = hd.genericize([ml_doc], 1.0)[0]
    assert all(s.title == f"Section {s.section_id}" for s in g.sections)
    assert hd.genericize([ml_doc], 0.0)[0].sections == ml_doc.sections
    f = hd.flatten([ml_doc])[0]
    assert all(s.path == [s.title] for s in f.sections)


def test_ocr_noise_changes_titles_only(ml_doc):
    n = hd.ocr_noise([ml_doc], 0.5, seed=0)[0]
    assert any(a.title != b.title for a, b in zip(n.sections, ml_doc.sections, strict=True))
    assert [s.text for s in n.sections] == [s.text for s in ml_doc.sections]


def test_heading_features_rank_informative_headings_higher(ml_doc):
    enc = FakeEncoder()
    real = document_features(SectionCorpus([ml_doc]), enc, pool=[ml_doc]).iloc[0]
    generic = hd.genericize([ml_doc], 1.0)[0]
    gen = document_features(SectionCorpus([generic]), enc, pool=[generic]).iloc[0]
    assert real.self_mrr > gen.self_mrr
    sec = section_features(SectionCorpus([ml_doc]), enc)
    assert sec.path_nrank.dropna().between(0, 1).all()


def test_reciprocal_ranks_within_document(ml_doc):
    corpus = SectionCorpus([ml_doc, other_doc()])
    q = Query(
        query_id="q",
        dataset="toy",
        document_id="ml_handbook",
        question="precision recall",
        target_section_ids=["3.2"],
        split="test",
        generator="ict",
    )
    comp = ex.compute(corpus, FakeEncoder(), [q])
    rr = ex.reciprocal_ranks(ex.structure_scores(comp, alpha=0.2), comp)
    assert rr[0] == 1.0
    assert ex.structure_rr_per_alpha(comp, [0.0, 1.0]).shape == (2, 1)
    assert np.all(ex.reciprocal_ranks(ex.flat_scores(comp), comp) <= 1)


def test_nq_page_parsing_builds_sections_from_heading_tokens():
    from structuralrag.ingestion.natural_questions import _section_of, parse_page

    toks = ["<H1>", "Page", "</H1>", "Lead", "text", "<H2>", "History", "(", "edit", ")", "</H2>", "Old"]
    toks += ["<H3>", "Early", "</H3>", "x", "<H2>", "Contents", "</H2>", "1", "History"]
    html = [t.startswith("<") for t in toks]
    heads, starts = parse_page(toks, html)
    assert [(h.title, h.depth) for h in heads] == [("Introduction", 1), ("History", 1), ("Early", 2)]
    assert heads[0].text == "Page Lead text" and heads[2].text == "x"
    assert _section_of(12, starts) == 2 and _section_of(3, starts) == 0


def test_heading_specificity_and_fallback(ml_doc):
    from structuralrag.analysis.informativeness import heading_specificity

    docs = [ml_doc, other_doc(), hd.retitle(other_doc(), {}).model_copy(update={"document_id": "third"})]
    corpus = SectionCorpus(docs)
    spec = heading_specificity(corpus)
    by_uid = dict(zip([s.uid for s in corpus.sections], spec, strict=True))
    assert by_uid["ml_handbook::1"] == 1.0  # "Introduction" appears only in this document
    assert by_uid["other::1"] < 1.0  # "Budget" appears in two of three documents
    q = Query(
        query_id="q",
        dataset="toy",
        document_id="other",
        question="budget costs",
        target_section_ids=["1"],
        split="test",
        generator="ict",
    )
    comp = ex.compute(corpus, FakeEncoder(), [q])
    plain = ex.structure_scores(comp, alpha=1.0, sparse_weight=0)[0]
    fb = ex.structure_scores(comp, alpha=1.0, sparse_weight=0, fallback=True)[0]
    w = comp.specificity[0]
    np.testing.assert_allclose(fb, comp.content[0] + w * comp.path[0] + (1 - w) * comp.content[0])
    assert not np.allclose(plain, fb)


def test_neutral_fallback_uses_document_mean_path_score(ml_doc):
    corpus = SectionCorpus([ml_doc, other_doc()])
    q = Query(
        query_id="q",
        dataset="toy",
        document_id="ml_handbook",
        question="accuracy",
        target_section_ids=["3.1"],
        split="test",
        generator="ict",
    )
    comp = ex.compute(corpus, FakeEncoder(), [q])
    fb = ex.structure_scores(comp, alpha=1.0, sparse_weight=0, fallback="mean")[0]
    w, p = comp.specificity[0], comp.path[0]
    np.testing.assert_allclose(fb, comp.content[0] + w * p + (1 - w) * p.mean())


def test_clustered_test_is_more_conservative_with_correlated_questions():
    from structuralrag.evaluation.statistical_tests import clustered_paired_test, paired_test

    rng = np.random.default_rng(0)
    doc_effect = np.repeat(rng.normal(0, 1, 20), 10)  # 20 documents x 10 questions, shared effects
    a, b = doc_effect + rng.normal(0, 0.1, 200), np.zeros(200)
    assert clustered_paired_test(a, b, np.repeat(np.arange(20), 10))["p_cluster"] > paired_test(a, b)["p_permutation"]
    assert clustered_paired_test(np.ones(30), np.zeros(30), np.arange(30))["p_cluster"] < 0.001

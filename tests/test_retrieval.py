import numpy as np
import pytest

from structuralrag.retrieval.base import rrf
from structuralrag.retrieval.bm25 import BM25Retriever
from structuralrag.retrieval.corpus import View
from structuralrag.retrieval.dense import DenseRetriever
from structuralrag.retrieval.hybrid import HybridRetriever
from structuralrag.retrieval.structure_aware import StructureAwareRetriever, StructureParams


def test_views(corpus):
    i = corpus.uid_to_idx["ml_handbook::2.1.1"]
    assert corpus.view(i, View.TEXT).startswith("Classification predicts")
    assert "Machine Learning > Supervised Learning > Classification" in corpus.view(i, View.PATH)
    assert "Supervised Learning: Supervised learning trains" in corpus.view(i, View.PARENT)
    assert corpus.view(i, View.FULL).startswith("Machine Learning Handbook.")


def test_bm25_finds_lexical_match(corpus):
    top = BM25Retriever(corpus).search("fraction of correct predictions", top_k=1)[0]
    assert top.section_id == "3.1"


def test_document_filter_and_validity(corpus):
    r = BM25Retriever(corpus)
    res = r.search("learning", top_k=50, document_id="ml_handbook")
    assert len(res) == len(corpus)
    assert all(x.uid in corpus.uid_to_idx for x in res)
    with pytest.raises(KeyError):
        r.search("x", document_id="missing")


def test_rrf_prefers_consensus():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    fused = rrf([a, b])
    assert fused.argmax() == 1 or fused[1] >= fused[2]
    assert fused.shape == (3,)


def test_dense_and_hybrid(corpus, fake_encoder):
    dense = DenseRetriever(corpus, fake_encoder)
    assert dense.search("labeled examples predict targets", top_k=1)[0].section_id == "2.1"
    hybrid = HybridRetriever(BM25Retriever(corpus), dense)
    assert hybrid.search("clusters unlabeled data", top_k=1)[0].section_id == "2.2"


def test_structure_aware_uses_hierarchy(corpus, fake_encoder):
    sa = StructureAwareRetriever(corpus, fake_encoder, StructureParams(alpha=1.0, beta=0.0, sparse_weight=0.0))
    # Query only mentions the *path* ("model evaluation"), not the body text of 3.2.
    scores = sa.score("model evaluation precision")
    best = corpus.sections[int(np.argmax(scores))]
    assert best.section_id in {"3", "3.2"}
    comp = sa.components(["model evaluation"])
    assert set(comp) == {"content", "path", "parent", "sparse"}
    # Parent term for 3.2 equals content score of its parent (3).
    i, p = corpus.uid_to_idx["ml_handbook::3.2"], corpus.uid_to_idx["ml_handbook::3"]
    assert comp["parent"][0, i] == pytest.approx(comp["content"][0, p])

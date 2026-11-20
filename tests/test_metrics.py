import math

import numpy as np
import pytest

from structuralrag.evaluation.retrieval_metrics import ndcg_at_k, per_query_metrics, recall_at_k, reciprocal_rank
from structuralrag.evaluation.statistical_tests import bootstrap_ci, paired_test
from structuralrag.evaluation.structural_metrics import structural_metrics


def test_recall_and_rr():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0.0
    assert recall_at_k(ranked, {"b"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert reciprocal_rank(ranked, {"c"}, k=2) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a"], {"a"}, 10) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x"], {"a"}, 10) == 0.0


def test_per_query_keys():
    m = per_query_metrics(["a", "b"], {"b"})
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["mrr@10"] == 0.5


def test_structural(corpus):
    s = structural_metrics("ml_handbook::2.2", {"ml_handbook::2.1.1"}, corpus)
    assert s["valid_id"] == 1 and s["correct_document"] == 1 and s["same_branch"] == 1
    assert s["tree_distance"] == 3  # 2.2 -> 2 -> 2.1 -> 2.1.1
    assert structural_metrics("ml_handbook::3.1", {"ml_handbook::2.1"}, corpus)["same_branch"] == 0


def test_stats():
    a = np.ones(50)
    lo, hi = bootstrap_ci(a)
    assert lo == hi == 1.0
    t = paired_test(np.ones(40), np.zeros(40))
    assert t["mean_diff"] == 1 and t["p_permutation"] < 0.01
    assert paired_test(np.ones(10), np.ones(10))["p_permutation"] == 1.0

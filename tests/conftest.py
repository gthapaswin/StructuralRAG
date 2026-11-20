from __future__ import annotations

import numpy as np
import pytest

from structuralrag.preprocessing.hierarchy_builder import FlatHeading, build_document
from structuralrag.retrieval.corpus import SectionCorpus


@pytest.fixture
def ml_doc():
    headings = [
        FlatHeading("Introduction", 1, "This handbook introduces core ideas of machine learning for practitioners."),
        FlatHeading("Machine Learning", 1, ""),
        FlatHeading(
            "Supervised Learning", 2, "Supervised learning trains models on labeled examples to predict targets."
        ),
        FlatHeading("Classification", 3, "Classification predicts discrete class labels such as spam or not spam."),
        FlatHeading(
            "Unsupervised Learning", 2, "Unsupervised learning discovers clusters and structure in unlabeled data."
        ),
        FlatHeading("Model Evaluation", 1, ""),
        FlatHeading("Accuracy", 2, "Accuracy is the fraction of predictions that are correct."),
        FlatHeading(
            "Precision and Recall", 2, "Precision measures false positives while recall measures false negatives."
        ),
    ]
    return build_document("toy", "ml_handbook", "Machine Learning Handbook", headings)


@pytest.fixture
def corpus(ml_doc):
    return SectionCorpus([ml_doc])


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder so tests need no model download."""

    model_name = "fake"

    def _enc(self, texts):
        from structuralrag.preprocessing.text_cleaner import tokenize

        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, hash(tok) % 256] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)

    def encode_passages(self, texts):
        return self._enc(texts)

    def encode_queries(self, queries):
        return self._enc(queries)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()

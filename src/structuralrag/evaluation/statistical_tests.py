"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 0) -> dict[str, float]:
    """Paired randomization (sign-flip) test on per-query differences, plus Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    rng = np.random.default_rng(seed)
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm = np.abs((signs * diff).mean(axis=1))
    p_perm = float((np.sum(perm >= observed) + 1) / (n_perm + 1))
    try:
        p_wil = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(diff) else 1.0
    except ValueError:
        p_wil = float("nan")
    return {"mean_diff": float(diff.mean()), "p_permutation": p_perm, "p_wilcoxon": p_wil}


def clustered_paired_test(
    a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Paired sign-flip test that treats documents (not questions) as the independent units.

    Questions from the same document share its sections, so their differences are not
    independent. The per-document sums of differences are sign-flipped together; the statistic
    is the mean per-question difference.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    rng = np.random.default_rng(seed)
    observed = abs(sums.sum())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(sums)))
    perm = np.abs(signs @ sums)
    return {
        "mean_diff": float(diff.mean()),
        "p_cluster": float((np.sum(perm >= observed - 1e-12) + 1) / (n_perm + 1)),
        "n_documents": int(len(sums)),
    }

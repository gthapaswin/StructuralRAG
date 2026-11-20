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

"""Dice, Spearman rank agreement, permutation test, bootstrap CI.

Manuscript Batch 2, section 5.1 and 5.2:

    Dice similarity coefficient against ground truth, computed per image and
    averaged. Where a method produces an empty mask and the reference is also
    empty, Dice is defined as 1.

    Significance is assessed by permutation test with 20,000 permutations of one
    ranking, reporting the two-sided proportion of permuted correlations at
    least as extreme as observed. Bootstrap 95% confidence intervals on mean
    Dice use 5,000 resamples.
"""

import numpy as np
from scipy.stats import spearmanr

N_PERMUTATIONS = 20_000
N_BOOTSTRAP = 5_000


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice coefficient. Empty prediction AND empty reference scores 1."""
    p = (np.asarray(pred) > 0)
    r = (np.asarray(ref) > 0)
    ps, rs = p.sum(), r.sum()
    if ps == 0 and rs == 0:
        return 1.0
    if ps == 0 or rs == 0:
        return 0.0
    return float(2.0 * np.logical_and(p, r).sum() / (ps + rs))


def binarise(mask: np.ndarray) -> np.ndarray:
    """Binarise on ANY non-zero value.

    Medetec masks are 0/1 and FUSeg masks 0/255 (manuscript 4.3). A fixed
    mid-range threshold silently empties the Medetec masks and yields Dice of
    exactly zero for every method. That bug was hit once already; this function
    exists so it cannot happen again.
    """
    return (np.asarray(mask) > 0).astype(np.uint8)


def spearman_permutation(a, b, n_perm: int = N_PERMUTATIONS, seed: int = 0):
    """Spearman rho with a two-sided permutation p-value.

    Returns (rho, p). p is the proportion of permuted |rho| >= observed |rho|,
    computed with the standard (count + 1) / (n + 1) correction so p is never 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("length mismatch")
    if len(a) < 3:
        return float("nan"), float("nan")

    rho = float(spearmanr(a, b).statistic)
    if not np.isfinite(rho):
        return rho, float("nan")

    rng = np.random.default_rng(seed)
    b_perm = np.tile(b, (n_perm, 1))
    idx = np.argsort(rng.random(b_perm.shape), axis=1)
    b_perm = np.take_along_axis(b_perm, idx, axis=1)

    from scipy.stats import rankdata
    ra = rankdata(a)
    rb = rankdata(b_perm, axis=1)
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean(axis=1, keepdims=True)
    num = (rb_c * ra_c).sum(axis=1)
    den = np.sqrt((ra_c ** 2).sum() * (rb_c ** 2).sum(axis=1))
    null = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

    p = (np.sum(np.abs(null) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return rho, float(p)


def bootstrap_ci(values, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
                 seed: int = 0):
    """Percentile bootstrap CI for the mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def min_significant_rho(n: int, n_perm: int = 4000, seed: int = 0) -> float:
    """Smallest |rho| reaching p < 0.05 at this n, by simulation.

    Manuscript 3.8 claims |rho| >= 0.83 at n = 6 and |rho| >= 0.42 at n = 22.
    This function lets that claim be checked rather than asserted.
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import rankdata
    a = rankdata(rng.random(n))
    null = []
    for _ in range(n_perm):
        b = rankdata(rng.random(n))
        ac, bc = a - a.mean(), b - b.mean()
        den = np.sqrt((ac ** 2).sum() * (bc ** 2).sum())
        null.append(abs((ac * bc).sum() / den) if den > 0 else 0.0)
    return float(np.percentile(null, 95))

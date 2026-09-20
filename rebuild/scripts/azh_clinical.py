#!/usr/bin/env python3
"""Add AZH as a third clinical reference and build the ceiling matrix.

Until now the clinical-to-clinical ceiling has been a single pairwise number,
Medetec against FUSeg, rho = 0.567. One measurement cannot say how much that
quantity varies, which is the weakest point in the whole argument: every claim
of the form "the benchmark reaches N% of ceiling" rests on it.

The AZH Wound and Vascular Center dataset (Wang et al., Sci Rep 2020) is the
source the Foot Ulcer Segmentation Challenge was later built from. It is
published as 224 px cropped and zero-padded patches; FUSeg is published as
512 px frames. So AZH is NOT an independent third clinic. It is the same
clinical source under a different curation and framing convention, and that is
what makes it worth having: it turns the framing result from something we
produced by cropping into something two published datasets already disagree
about.

Scores all 22 methods on AZH, then computes every pairwise clinical rank
correlation with permutation p and bootstrap CI.

Writes results/clinical_ceiling_matrix.json and extends
results/clinical_ranking_cache.json with an "azh" entry.
"""

import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.methods import METHOD_NAMES                   # noqa: E402
from woundbench.metrics import spearman_permutation, bootstrap_ci  # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark   # noqa: E402

AZH = os.path.join(DATA, "wound_dataset")
N_AZH = 150          # comparable to Medetec's 152 and FUSeg's 120
SLICES = [("azh", "test", 0, 150), ("azh_train_slice", "train", 0, 150)]


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    cache_path = os.path.join(OUT, "clinical_ranking_cache.json")
    cache = json.load(open(cache_path))
    print(f"cache already holds: {sorted(cache)}", flush=True)

    for key, split, a, b in SLICES:
        if key in cache:
            print(f"{key} already scored, skipping", flush=True)
            continue
        X, Y = load_pairs(os.path.join(AZH, split, "images"),
                          os.path.join(AZH, split, "labels"), a, b)
        print(f"\n{key}: {len(X)} image/label pairs from AZH/{split}", flush=True)
        frac = float(np.mean([m.mean() for m in Y]))
        print(f"  mean lesion fraction {frac:.4f}", flush=True)
        summ, _ = score_benchmark(key, X, Y, reg, names)
        cache[key] = [summ[n]["mean_dice"] for n in names]
        cache[key + "__lesion_fraction"] = frac
        json.dump(cache, open(cache_path, "w"), indent=2)
        print(f"  cached {key}", flush=True)

    # ---- the ceiling matrix -------------------------------------------------
    refs = [k for k in ("medetec", "fuseg", "azh", "azh_train_slice")
            if k in cache and isinstance(cache[k], list)]
    print("\n" + "=" * 72)
    print("CLINICAL-TO-CLINICAL RANK AGREEMENT, ALL PAIRS")
    print("=" * 72)
    print(f"{'pair':<34}{'rho':>9}{'p':>9}{'95% CI':>20}")
    print("-" * 72)
    out = {}
    for a, b in itertools.combinations(refs, 2):
        r, p = spearman_permutation(cache[a], cache[b], n_perm=20000)
        lo, hi = _rho_ci(cache[a], cache[b])
        out[f"{a}__{b}"] = {"rho": r, "p": p, "ci_lo": lo, "ci_hi": hi}
        print(f"{a + ' vs ' + b:<34}{r:>+9.3f}{p:>9.4f}"
              f"{'[' + format(lo, '+.3f') + ', ' + format(hi, '+.3f') + ']':>20}")

    indep = [v["rho"] for k, v in out.items()
             if "azh_train_slice" not in k]
    print("\nceiling as a single number was 0.567 (Medetec vs FUSeg alone)")
    if len(indep) > 1:
        print(f"across {len(indep)} clinical pairs: mean {np.mean(indep):.3f}, "
              f"range {min(indep):.3f} to {max(indep):.3f}, "
              f"spread {max(indep) - min(indep):.3f}")
        print("Any '% of ceiling' statement inherits that spread.")

    # lesion fractions, which is what the framing argument is about
    print("\n" + "=" * 72)
    print("FRAMING CONVENTION OF EACH CLINICAL REFERENCE")
    print("=" * 72)
    for k in refs:
        f = cache.get(k + "__lesion_fraction")
        print(f"  {k:<20}{'lesion fraction ' + format(f, '.4f') if f else ''}")

    json.dump({"matrix": out, "refs": refs},
              open(os.path.join(OUT, "clinical_ceiling_matrix.json"), "w"),
              indent=2)
    print(f"\nsaved {os.path.join(OUT, 'clinical_ceiling_matrix.json')}")


def _rho_ci(a, b, n_boot=5000, seed=0):
    """Bootstrap CI on a rank correlation by resampling methods."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(i.tolist())) < 4:
            continue
        v = spearmanr(a[i], b[i]).statistic
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    main()

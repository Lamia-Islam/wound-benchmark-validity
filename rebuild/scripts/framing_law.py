#!/usr/bin/env python3
"""Is the clinical reference's verdict a continuous function of its framing?

The framing test showed a single reversal: crop FUSeg to Medetec's lesion
fraction and its contrast preference flips from +0.964 to -0.964. A reversal
between two points is a demonstration. If the verdict is instead a monotonic
function of lesion fraction, and independently curated clinical datasets land on
that function at their own fractions, it becomes a quantitative statement with
predictive content.

Construction. FUSeg is cropped around its own lesions to a series of target
lesion fractions spanning the clinical range. Each crop is a clinical reference
in its own right, built from identical images and identical annotations, so
lesion fraction is the only thing that varies. For each one the cached contrast
sweep is re-correlated and the trend of agreement against contrast is recorded.

The test. Three real clinical datasets have measured lesion fractions:

    FUSeg    0.0109    trend +0.964
    AZH      0.0285    trend ?
    Medetec  0.1024    trend -0.893

AZH was never used to build anything here and its trend is predicted by the
curve before it is measured. If it lands on the curve, framing is the governing
variable rather than a property of any one dataset.

Resumable via results/framing_law_cells.jsonl.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.methods import METHOD_NAMES                       # noqa: E402
from woundbench.metrics import spearman_permutation               # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark  # noqa: E402
from framing_test import crop_to_fraction                         # noqa: E402

TARGETS = [0.005, 0.010, 0.020, 0.035, 0.055, 0.080, 0.110, 0.160]
N_FUSEG = 120
CELLS = os.path.join(OUT, "framing_law_cells.jsonl")

MEASURED = {                       # real datasets, for the overlay
    "fuseg": 0.0109,
    "azh": 0.0285,
    "medetec": 0.1024,
}


def done():
    d = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            if line.strip():
                r = json.loads(line)
                d[r["target"]] = r
    return d


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    dr = json.load(open(os.path.join(OUT, "dose_response.json")))
    contrast = dr["contrast_sweep"]
    x_contrast = [r["measured_sat_delta"] for r in contrast]

    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, N_FUSEG)
    print(f"{len(fu_i)} FUSeg frames loaded", flush=True)

    D = done()
    todo = [t for t in TARGETS if t not in D]
    print(f"{len(D)} crop levels done, {len(todo)} to go\n", flush=True)

    for t in todo:
        ci, cm = [], []
        for im, ms in zip(fu_i, fu_m):
            a, b = crop_to_fraction(im, ms, target_frac=t)
            if a is not None:
                ci.append(a)
                cm.append(b)
        if len(ci) < 40:
            print(f"  target {t}: only {len(ci)} usable crops, skipped",
                  flush=True)
            continue
        achieved = float(np.median([m.mean() for m in cm]))
        summ, _ = score_benchmark(f"crop{t}", ci, cm, reg, names, log=None)
        ref = [summ[n]["mean_dice"] for n in names]
        agree = [spearman_permutation([r["per_method_dice"][n] for n in names],
                                      ref, n_perm=4000)[0] for r in contrast]
        trend, p = spearman_permutation(x_contrast, agree, n_perm=20000)
        row = {"target": t, "achieved": achieved, "n": len(ci),
               "trend": trend, "p": p, "agreement": agree,
               "per_method_dice": {n: summ[n]["mean_dice"] for n in names}}
        open(CELLS, "a").write(json.dumps(row) + "\n")
        print(f"  target {t:.3f} -> achieved {achieved:.4f}  n {len(ci):>3d}  "
              f"contrast trend {trend:+.3f} (p {p:.4f})", flush=True)

    D = done()
    rows = [D[t] for t in TARGETS if t in D]
    if len(rows) < 3:
        print("not enough levels to fit a curve yet")
        return

    print("\n" + "=" * 74)
    print("VERDICT AS A FUNCTION OF THE REFERENCE'S LESION FRACTION")
    print("=" * 74)
    print(f"{'target':>9}{'achieved':>11}{'n':>6}{'contrast trend':>17}{'p':>9}")
    for r in rows:
        print(f"{r['target']:>9.3f}{r['achieved']:>11.4f}{r['n']:>6d}"
              f"{r['trend']:>+17.3f}{r['p']:>9.4f}")

    f = [r["achieved"] for r in rows]
    tr = [r["trend"] for r in rows]
    mono, pmono = spearman_permutation(f, tr, n_perm=20000)
    print(f"\nmonotonicity of trend against lesion fraction: "
          f"Spearman {mono:+.3f}, p {pmono:.4f}")
    print("negative means: the larger the lesion fills the frame, the more the")
    print("reference prefers LOW contrast. That is the framing effect, as a curve.")

    sign_change = any(tr[i] * tr[i + 1] < 0 for i in range(len(tr) - 1))
    if sign_change:
        for i in range(len(tr) - 1):
            if tr[i] * tr[i + 1] < 0:
                lo, hi = f[i], f[i + 1]
                print(f"\nthe verdict changes sign between lesion fraction "
                      f"{lo:.4f} and {hi:.4f}")
                break

    # ---- the out-of-sample check -------------------------------------------
    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache.json")))
    print("\n" + "=" * 74)
    print("REAL CLINICAL DATASETS PLACED ON THE CURVE")
    print("=" * 74)
    print(f"{'dataset':<12}{'lesion frac':>13}{'predicted':>12}{'measured':>11}"
          f"{'residual':>11}")
    overlay = {}
    for k, frac in sorted(MEASURED.items(), key=lambda kv: kv[1]):
        if k not in cache or not isinstance(cache[k], list):
            print(f"{k:<12}{frac:>13.4f}{'':>12}{'not scored yet':>22}")
            continue
        ref = cache[k]
        agree = [spearman_permutation([r["per_method_dice"][n] for n in names],
                                      ref, n_perm=4000)[0] for r in contrast]
        meas = spearman_permutation(x_contrast, agree, n_perm=20000)[0]
        pred = float(np.interp(frac, f, tr))
        overlay[k] = {"lesion_fraction": frac, "predicted": pred,
                      "measured": meas, "residual": meas - pred,
                      "agreement": agree}
        print(f"{k:<12}{frac:>13.4f}{pred:>+12.3f}{meas:>+11.3f}"
              f"{meas - pred:>+11.3f}")

    if "azh" in overlay:
        r = overlay["azh"]
        print(f"\nAZH is the held-out point. It was never used to build the "
              f"curve.")
        print(f"  curve predicts {r['predicted']:+.3f}, AZH measures "
              f"{r['measured']:+.3f}, residual {r['residual']:+.3f}")
        others = [abs(v["trend"]) for v in rows]
        print(f"  for scale, the trend ranges over "
              f"{min(tr):+.3f} to {max(tr):+.3f} across the curve")

    json.dump({"curve": [{k: v for k, v in r.items()
                          if k != "per_method_dice"} for r in rows],
               "monotonicity": {"rho": mono, "p": pmono},
               "overlay": overlay},
              open(os.path.join(OUT, "framing_law.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'framing_law.json')}")


if __name__ == "__main__":
    main()

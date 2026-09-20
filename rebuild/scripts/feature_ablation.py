#!/usr/bin/env python3
"""Leave-one-feature-out, to identify the original 14-feature set.

The manuscript says the pixel classifiers used FOURTEEN features but enumerates
FIFTEEN (9 colour + 4 texture + 2 self-referenced). This rebuild uses 15 and its
logistic regression scores ~0.93 Dice on generator A where the manuscript
records 0.166.

Test: drop each of the 15 features in turn, refit, and score on generator A and
both clinical datasets. If exactly one drop sends generator A performance toward
0.166 WHILE HOLDING the clinical scores near their reported values (0.451
Medetec, 0.437 FUSeg), that identifies the feature set the original used.

If no single drop does that, the 14-vs-15 discrepancy is not the explanation and
the search moves elsewhere.

Writes results/feature_ablation.json.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import features as feat                       # noqa: E402
from woundbench import learned as L                           # noqa: E402
from woundbench.generator_a import build as build_gen_a       # noqa: E402
from woundbench.metrics import dice                           # noqa: E402
from woundbench.postproc import postprocess_prob              # noqa: E402
from rank_validity import DATA, OUT, load_pairs               # noqa: E402

TARGET_GEN_A = 0.166      # manuscript, logistic regression on generator A
TARGET_MEDETEC = 0.451
TARGET_FUSEG = 0.437
THRESHOLD = 0.9


def score_with(clf, keep_idx, images, masks):
    ds = []
    for im, gt in zip(images, masks):
        f = feat.extract(im)
        h, w, _ = f.shape
        flat = f.reshape(-1, f.shape[-1])[:, keep_idx]
        p = clf.predict_proba(flat)[:, 1].reshape(h, w).astype(np.float32)
        ds.append(dice(postprocess_prob(p, THRESHOLD), gt))
    return float(np.mean(ds))


def main():
    img_dir = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train",
                           "images")
    lab_dir = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train",
                           "labels")
    print("loading training images", flush=True)
    tr_i, tr_m = load_pairs(img_dir, lab_dir, 0, L.N_TRAIN_IMAGES_PIXEL)

    print("sampling pixels once (reused for every variant)", flush=True)
    X, y = L.sample_pixels(tr_i, tr_m, seed=0)
    print(f"  {X.shape[0]:,} px x {X.shape[1]} features", flush=True)

    # evaluation sets, subsampled to keep 16 refits tractable
    ga_i, ga_m, _ = build_gen_a(n_per_class=5)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"), 0, 40)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, 40)
    print(f"  eval: genA {len(ga_i)}, medetec {len(me_i)}, fuseg {len(fu_i)}\n",
          flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(keep_idx):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=0))
        clf.fit(X[:, keep_idx], y)
        return clf

    rows = []
    all_idx = list(range(len(feat.FEATURE_NAMES)))

    variants = [("none (all 15)", all_idx)]
    for i, nm in enumerate(feat.FEATURE_NAMES):
        variants.append((f"drop {nm}", [j for j in all_idx if j != i]))

    print(f"{'variant':<26}{'genA':>8}{'Medetec':>10}{'FUSeg':>9}"
          f"{'|genA-0.166|':>14}")
    print("-" * 67)
    for label, keep in variants:
        clf = fit(keep)
        g = score_with(clf, keep, ga_i, ga_m)
        m = score_with(clf, keep, me_i, me_m)
        f_ = score_with(clf, keep, fu_i, fu_m)
        rows.append({"variant": label, "n_features": len(keep),
                     "gen_a": g, "medetec": m, "fuseg": f_,
                     "gen_a_gap": abs(g - TARGET_GEN_A)})
        print(f"{label:<26}{g:>8.3f}{m:>10.3f}{f_:>9.3f}"
              f"{abs(g - TARGET_GEN_A):>14.3f}", flush=True)

    print(f"\nmanuscript targets: genA {TARGET_GEN_A}, "
          f"Medetec {TARGET_MEDETEC}, FUSeg {TARGET_FUSEG}")

    best = min(rows[1:], key=lambda r: r["gen_a_gap"])
    print(f"\nclosest single drop to the manuscript's generator A value:")
    print(f"  {best['variant']}  genA {best['gen_a']:.3f} "
          f"(target {TARGET_GEN_A}), Medetec {best['medetec']:.3f} "
          f"(target {TARGET_MEDETEC})")
    spread = max(r["gen_a"] for r in rows) - min(r["gen_a"] for r in rows)
    print(f"  spread of generator A scores across all 16 variants: {spread:.3f}")
    if best["gen_a_gap"] > 0.3:
        print("\n  VERDICT: no single feature drop reproduces the manuscript's")
        print("  generator A behaviour. The 14-vs-15 discrepancy is NOT the")
        print("  explanation; the cause lies elsewhere.")
    else:
        print("\n  VERDICT: a single feature drop reproduces it. That feature")
        print("  set is the likely original.")

    with open(os.path.join(OUT, "feature_ablation.json"), "w") as fh:
        json.dump({"rows": rows,
                   "targets": {"gen_a": TARGET_GEN_A,
                               "medetec": TARGET_MEDETEC,
                               "fuseg": TARGET_FUSEG}}, fh, indent=2)
    print(f"\nsaved {os.path.join(OUT, 'feature_ablation.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refit the learned methods on the identified 14-feature set, then re-run
the four-benchmark rank agreement.

The leave-one-out ablation identified `a_localsd` as the feature whose removal
reproduces the manuscript's generator A behaviour. This refits all three pixel
classifiers without it and re-tests the headline:

    generator A vs generator B   manuscript -0.526,  15-feature rebuild +0.456
    generator A vs FUSeg         manuscript -0.466,  15-feature rebuild -0.099
    generator A vs Medetec       manuscript -0.121,  15-feature rebuild +0.342

The U-Net is unchanged (it takes raw pixels, not these features), so it is
reloaded from the existing model file rather than retrained.

Writes results/models_14.pkl and results/rank_validity_14.json.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import features as feat                       # noqa: E402
from woundbench import learned as L                           # noqa: E402
from woundbench.generator_a import build as build_gen_a       # noqa: E402
from woundbench.generator_b import build as build_gen_b       # noqa: E402
from woundbench.methods import (METHOD_NAMES, METHOD_FAMILY,  # noqa: E402
                                MethodRegistry)
from woundbench.metrics import spearman_permutation           # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark, TARGETS  # noqa: E402


class Registry14(MethodRegistry):
    """MethodRegistry whose pixel classifiers use the 14-feature set."""

    KEEP = feat.MANUSCRIPT_14_IDX

    def fit_pixel_classifiers(self, images, masks, seed=0, log=print):
        X, y = L.sample_pixels(images, masks, seed=seed)
        X14 = X[:, self.KEEP]
        if log:
            log(f"  {X14.shape[0]:,} px x {X14.shape[1]} features "
                f"(dropped a_localsd)")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.logreg_all = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000,
                                                 random_state=seed)).fit(X14, y)
        n_col = len(feat.COLOUR_FEATURES)
        self.logreg_colour = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000,
                                                 random_state=seed)
        ).fit(X[:, :n_col], y)
        self.gbm_all = HistGradientBoostingClassifier(
            random_state=seed).fit(X14, y)
        return self

    def probability(self, name, bgr):
        if name == "unet":
            return L.unet_prob(self.unet, bgr)
        f = feat.extract(bgr)
        h, w, c = f.shape
        flat = f.reshape(-1, c)
        if name == "logreg_colour":
            flat = flat[:, :len(feat.COLOUR_FEATURES)]
            clf = self.logreg_colour
        else:
            flat = flat[:, self.KEEP]
            clf = self.logreg_all if name == "logreg_all" else self.gbm_all
        return clf.predict_proba(flat)[:, 1].reshape(h, w).astype(np.float32)


def main():
    img_dir = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train",
                           "images")
    lab_dir = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train",
                           "labels")
    print("refitting pixel classifiers on 14 features", flush=True)
    tr_i, tr_m = load_pairs(img_dir, lab_dir, 0, L.N_TRAIN_IMAGES_PIXEL)

    reg = Registry14()
    reg.fit_pixel_classifiers(tr_i, tr_m, seed=0)
    with open(os.path.join(OUT, "models.pkl"), "rb") as f:
        reg.unet = pickle.load(f)["registry"].unet     # unchanged by features
    print("  U-Net reloaded unchanged", flush=True)

    names = METHOD_NAMES
    benchmarks = {}
    print("scoring generator A", flush=True)
    ga_i, ga_m, _ = build_gen_a(n_per_class=15)
    benchmarks["generator_a"] = score_benchmark("gA", ga_i, ga_m, reg, names,
                                                log=None)[0]
    print("scoring generator B", flush=True)
    gb_i, gb_m, _ = build_gen_b(n_per_class=15)
    benchmarks["generator_b"] = score_benchmark("gB", gb_i, gb_m, reg, names,
                                                log=None)[0]
    print("scoring medetec", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    benchmarks["medetec"] = score_benchmark("me", me_i, me_m, reg, names,
                                            log=None)[0]
    print("scoring fuseg", flush=True)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, 120)
    benchmarks["fuseg"] = score_benchmark("fu", fu_i, fu_m, reg, names,
                                          log=None)[0]

    print("\nlearned-method scores on generator A (the diagnostic):")
    for n in ["logreg_all", "gbm_all", "logreg_colour", "unet"]:
        print(f"  {n:<16}{benchmarks['generator_a'][n]['mean_dice']:>8.3f}")
    print("  manuscript: logreg 0.166, gbm 0.011, unet 0.080")

    keys = list(benchmarks)
    print("\n" + "=" * 78)
    print("rank agreement, 14-feature classifiers")
    print("=" * 78)
    print(f"{'comparison':<34}{'rho':>9}{'p':>9}{'15-feat':>10}"
          f"{'target':>10}")
    prev = json.load(open(os.path.join(OUT, "rank_validity.json")))["rank_agreement"]
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = [benchmarks[a][n]["mean_dice"] for n in names]
            vb = [benchmarks[b][n]["mean_dice"] for n in names]
            rho, p = spearman_permutation(va, vb)
            out[f"{a}__{b}"] = {"rho": rho, "p": p}
            old = prev.get(f"{a}__{b}", {}).get("rho", float("nan"))
            t = TARGETS.get((a, b))
            ts = f"{t[0]:>10.3f}" if t else " " * 10
            print(f"{a + ' vs ' + b:<34}{rho:>9.3f}{p:>9.3f}{old:>10.3f}{ts}")

    with open(os.path.join(OUT, "models_14.pkl"), "wb") as f:
        pickle.dump({"registry": reg}, f)
    with open(os.path.join(OUT, "rank_validity_14.json"), "w") as f:
        json.dump({"methods": names,
                   "benchmarks": benchmarks,
                   "rank_agreement": out}, f, indent=2)
    print(f"\nsaved {os.path.join(OUT, 'rank_validity_14.json')}")


if __name__ == "__main__":
    main()

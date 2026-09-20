#!/usr/bin/env python3
"""Regenerate Table 3: rank agreement across benchmarks.

This is the GO/NO-GO step for the whole rebuild. The manuscript reports, over
22 methods scored by Dice on four benchmarks:

    generator A vs Medetec      rho = -0.121   p = 0.588
    generator A vs FUSeg        rho = -0.466   p = 0.033
    generator B vs Medetec      rho =  0.439   p = 0.040
    generator B vs FUSeg        rho =  0.560   p = 0.007
    generator A vs generator B  rho = -0.526   p = 0.014
    Medetec vs FUSeg            rho =  0.560   p = 0.008

Generator B does not exist yet in this rebuild, so this script covers the three
benchmarks that do: generator A, Medetec, FUSeg. The load-bearing check
available now is Medetec vs FUSeg (target +0.560) and generator A vs each
clinical set (targets -0.121 and -0.466).

Writes results/rank_validity.json.
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_a import build as build_gen_a       # noqa: E402
from woundbench.generator_b import build as build_gen_b       # noqa: E402
from woundbench.methods import METHOD_NAMES, METHOD_FAMILY    # noqa: E402
from woundbench.metrics import (dice, binarise,               # noqa: E402
                                spearman_permutation, bootstrap_ci)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "wound-segmentation", "data")
OUT = os.path.join(ROOT, "results")

TARGETS = {
    ("generator_a", "medetec"): (-0.121, 0.588),
    ("generator_a", "fuseg"): (-0.466, 0.033),
    ("generator_a", "generator_b"): (-0.526, 0.014),   # the headline result
    ("generator_b", "medetec"): (0.439, 0.040),
    ("generator_b", "fuseg"): (0.560, 0.007),
    ("medetec", "fuseg"): (0.560, 0.008),
}


def load_pairs(img_dir, lab_dir, start=0, stop=None):
    ext = (".png", ".jpg", ".jpeg")
    imgs = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                  if p.lower().endswith(ext))
    labs = sorted(p for p in glob.glob(os.path.join(lab_dir, "*"))
                  if p.lower().endswith(ext))
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    lab_by = {stem(p): p for p in labs}
    pairs = [(i, lab_by[stem(i)]) for i in imgs if stem(i) in lab_by][start:stop]
    X, Y = [], []
    for ip, lp in pairs:
        im = cv2.imread(ip)
        ms = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if im is not None and ms is not None:
            X.append(im)
            Y.append(binarise(ms))
    return X, Y


def score_benchmark(name, images, masks, registry, names, log=print):
    """Mean Dice per method over one benchmark."""
    per_method = {n: [] for n in names}
    t0 = time.time()
    for k, (im, gt) in enumerate(zip(images, masks)):
        out = registry.run_all(im, names=names)
        for n, pred in out.items():
            per_method[n].append(dice(pred, gt))
        if log and (k + 1) % 25 == 0:
            log(f"    {name}: {k + 1}/{len(images)} "
                f"({time.time() - t0:.0f}s)")
    summary = {}
    for n in names:
        m, lo, hi = bootstrap_ci(per_method[n])
        summary[n] = {"mean_dice": m, "ci_lo": lo, "ci_hi": hi,
                      "n_images": len(per_method[n])}
    return summary, per_method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuseg-n", type=int, default=120,
                    help="FUSeg validation images used as the eval slice "
                         "(manuscript uses 120)")
    ap.add_argument("--gen-a-per-class", type=int, default=15)
    ap.add_argument("--classical-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    import pickle
    from woundbench.methods import MethodRegistry
    model_path = os.path.join(OUT, "models.pkl")
    if args.classical_only or not os.path.exists(model_path):
        print("running CLASSICAL METHODS ONLY (18 of 22)", flush=True)
        registry = MethodRegistry()
        names = [n for n in METHOD_NAMES if METHOD_FAMILY[n]
                 not in ("learned", "CNN")]
    else:
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        registry = blob["registry"]
        names = METHOD_NAMES
        print(f"loaded fitted models (threshold "
              f"{registry.learned_threshold:.2f})", flush=True)
    print(f"{len(names)} methods\n", flush=True)

    benchmarks = {}

    print("generator A", flush=True)
    ga_imgs, ga_masks, _ = build_gen_a(n_per_class=args.gen_a_per_class)
    benchmarks["generator_a"] = score_benchmark(
        "generator_a", ga_imgs, ga_masks, registry, names)

    print("generator B", flush=True)
    gb_imgs, gb_masks, _ = build_gen_b(n_per_class=args.gen_a_per_class)
    benchmarks["generator_b"] = score_benchmark(
        "generator_b", gb_imgs, gb_masks, registry, names)

    print("medetec (152 clinical images)", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    benchmarks["medetec"] = score_benchmark(
        "medetec", me_i, me_m, registry, names)

    print(f"fuseg (validation images 0-{args.fuseg_n})", flush=True)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, args.fuseg_n)
    benchmarks["fuseg"] = score_benchmark("fuseg", fu_i, fu_m, registry, names)

    # ---- per-method table ------------------------------------------------
    keys = list(benchmarks)
    print("\n" + "=" * 78)
    print(f"{'method':<26}{'family':<22}" + "".join(f"{k:>10}" for k in keys))
    print("=" * 78)
    order = sorted(names, key=lambda n: -benchmarks["medetec"][0][n]["mean_dice"])
    for n in order:
        row = "".join(f"{benchmarks[k][0][n]['mean_dice']:>10.3f}" for k in keys)
        print(f"{n:<26}{METHOD_FAMILY[n]:<22}{row}")

    # ---- rank agreement --------------------------------------------------
    print("\n" + "=" * 78)
    print("rank agreement (Spearman, permutation test, n = "
          f"{len(names)} methods)")
    print("=" * 78)
    print(f"{'comparison':<34}{'rho':>9}{'p':>9}   {'target rho':>11}{'target p':>10}")
    rank_results = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va = [benchmarks[a][0][n]["mean_dice"] for n in names]
            vb = [benchmarks[b][0][n]["mean_dice"] for n in names]
            rho, p = spearman_permutation(va, vb)
            rank_results[f"{a}__{b}"] = {"rho": rho, "p": p}
            t = TARGETS.get((a, b))
            tgt = f"{t[0]:>11.3f}{t[1]:>10.3f}" if t else " " * 21
            print(f"{a + ' vs ' + b:<34}{rho:>9.3f}{p:>9.3f}   {tgt}")

    blob = {
        "methods": names,
        "families": {n: METHOD_FAMILY[n] for n in names},
        "benchmarks": {k: v[0] for k, v in benchmarks.items()},
        "per_image_dice": {k: v[1] for k, v in benchmarks.items()},
        "rank_agreement": rank_results,
        "targets": {f"{a}__{b}": {"rho": r, "p": p}
                    for (a, b), (r, p) in TARGETS.items()},
    }
    path = os.path.join(OUT, "rank_validity.json")
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

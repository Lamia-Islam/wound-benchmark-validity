#!/usr/bin/env python3
"""Regenerate Table 5: which generator B design property governs rank transfer.

Manuscript 5.3:

    Generator B is ablated one design feature at a time, each reverting to
    generator A's corresponding assumption, with all other features held at
    generator B settings and the random seed fixed. For each variant, all 22
    methods are scored and the resulting ranking is correlated against both
    clinical datasets.

Manuscript Table 5 (targets):

    | ablated feature            | rho Medetec | rho FUSeg | mean rho | change |
    | none (full generator B)    |  0.356      |  0.510    |  0.433   |   -    |
    | grade-independent size     |  0.291      |  0.430    |  0.360   | -0.073 |
    | off-centre placement       |  0.260      |  0.448    |  0.354   | -0.079 |
    | clutter                    |  0.303      |  0.436    |  0.369   | -0.064 |
    | irregular lesion shape     |  0.372      |  0.534    |  0.453   | +0.020 |
    | textured background        |  0.070      | -0.268    | -0.099   | -0.532 |

The claim under test: removing background texture costs ~0.532 in mean rank
correlation and REVERSES the sign, while no other single feature costs more
than ~0.08.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_b import build as build_gen_b, ABLATIONS  # noqa: E402
from woundbench.methods import METHOD_NAMES, MethodRegistry         # noqa: E402
from woundbench.metrics import spearman_permutation                 # noqa: E402

from rank_validity import DATA, OUT, load_pairs, score_benchmark    # noqa: E402

TARGETS = {
    "none": (0.356, 0.510, 0.433, None),
    "grade_independent_size": (0.291, 0.430, 0.360, -0.073),
    "off_centre_placement": (0.260, 0.448, 0.354, -0.079),
    "clutter": (0.303, 0.436, 0.369, -0.064),
    "irregular_shape": (0.372, 0.534, 0.453, +0.020),
    "textured_background": (0.070, -0.268, -0.099, -0.532),
}

ORDER = ["none", "grade_independent_size", "off_centre_placement", "clutter",
         "irregular_shape", "textured_background"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--fuseg-n", type=int, default=120)
    args = ap.parse_args()

    with open(os.path.join(OUT, "models.pkl"), "rb") as f:
        registry = pickle.load(f)["registry"]
    names = METHOD_NAMES
    print(f"{len(names)} methods, threshold {registry.learned_threshold:.2f}\n",
          flush=True)

    # Clinical reference rankings, scored once.
    print("scoring medetec", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    med, _ = score_benchmark("medetec", me_i, me_m, registry, names, log=None)

    print(f"scoring fuseg (0-{args.fuseg_n})", flush=True)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, args.fuseg_n)
    fus, _ = score_benchmark("fuseg", fu_i, fu_m, registry, names, log=None)

    med_v = [med[n]["mean_dice"] for n in names]
    fus_v = [fus[n]["mean_dice"] for n in names]

    rows = {}
    for ab in ORDER:
        print(f"ablation: {ab}", flush=True)
        imgs, masks, _ = build_gen_b(n_per_class=args.per_class, ablation=ab)
        summ, _ = score_benchmark(ab, imgs, masks, registry, names, log=None)
        v = [summ[n]["mean_dice"] for n in names]
        r_med, p_med = spearman_permutation(v, med_v)
        r_fus, p_fus = spearman_permutation(v, fus_v)
        rows[ab] = {"rho_medetec": r_med, "p_medetec": p_med,
                    "rho_fuseg": r_fus, "p_fuseg": p_fus,
                    "mean_rho": (r_med + r_fus) / 2,
                    "per_method_dice": {n: summ[n]["mean_dice"] for n in names}}
        print(f"    rho_med {r_med:+.3f}  rho_fuseg {r_fus:+.3f}  "
              f"mean {(r_med + r_fus) / 2:+.3f}", flush=True)

    base = rows["none"]["mean_rho"]
    for ab in ORDER:
        rows[ab]["change"] = None if ab == "none" else rows[ab]["mean_rho"] - base

    print("\n" + "=" * 92)
    print("Table 5: ablation of generator B   (rebuild vs manuscript target)")
    print("=" * 92)
    print(f"{'ablated feature':<26}{'rho Med':>16}{'rho FUSeg':>16}"
          f"{'mean rho':>16}{'change':>16}")
    print(f"{'':26}{'ours / ms':>16}{'ours / ms':>16}{'ours / ms':>16}"
          f"{'ours / ms':>16}")
    print("-" * 92)
    for ab in ORDER:
        r = rows[ab]
        t = TARGETS[ab]
        ch = "   -  /   -  " if r["change"] is None else \
             f"{r['change']:+.3f}/{t[3]:+.3f}"
        print(f"{ab:<26}"
              f"{r['rho_medetec']:+7.3f}/{t[0]:+7.3f}"
              f"{r['rho_fuseg']:+9.3f}/{t[1]:+6.3f}"
              f"{r['mean_rho']:+9.3f}/{t[2]:+6.3f}"
              f"{ch:>16}")

    path = os.path.join(OUT, "ablation.json")
    with open(path, "w") as f:
        json.dump({"rows": rows, "targets": TARGETS, "methods": names}, f,
                  indent=2)
    print(f"\nsaved {path}")

    tex = rows["textured_background"]
    print("\n--- the claim under test ---")
    print(f"texture removal changes mean rho by {tex['change']:+.3f} "
          f"(manuscript: -0.532)")
    print(f"sign of FUSeg correlation after removal: {tex['rho_fuseg']:+.3f} "
          f"(manuscript: -0.268, i.e. reversed)")
    others = [abs(rows[a]["change"]) for a in ORDER
              if a not in ("none", "textured_background")]
    print(f"largest other single-feature effect: {max(others):.3f} "
          f"(manuscript: 0.079)")


if __name__ == "__main__":
    main()

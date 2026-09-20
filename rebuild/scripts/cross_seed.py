#!/usr/bin/env python3
"""Cross-seed variance on the texture dose-response.

Addresses the single-seed caveat. Five seeds at each of five texture levels,
22 methods scored at every cell. Reports mean and spread per level, and tests
the monotone trend with seed treated as a repeated factor (a per-seed Spearman,
then the distribution of those).

Writes results/cross_seed.json.
"""

import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_b import GeneratorB, GenBConfig      # noqa: E402
from woundbench.methods import METHOD_NAMES                     # noqa: E402
from woundbench.metrics import spearman_permutation             # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark  # noqa: E402

LEVELS = [0.0, 11.0, 18.0, 28.0, 40.0]
SEEDS = [2024, 7, 101, 555, 31337]
N_PER_CLASS = 15


def main():
    with open(os.path.join(OUT, "models.pkl"), "rb") as f:
        reg = pickle.load(f)["registry"]
    names = METHOD_NAMES

    print("scoring clinical reference rankings once", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    med, _ = score_benchmark("medetec", me_i, me_m, reg, names, log=None)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, 120)
    fus, _ = score_benchmark("fuseg", fu_i, fu_m, reg, names, log=None)
    med_v = [med[n]["mean_dice"] for n in names]
    fus_v = [fus[n]["mean_dice"] for n in names]
    ceiling = spearman_permutation(med_v, fus_v)[0]
    print(f"  clinical ceiling rho = {ceiling:+.3f}\n", flush=True)

    cells = {}
    for amp in LEVELS:
        for seed in SEEDS:
            cfg = replace(GenBConfig(), mottle_amplitude=amp)
            recs = GeneratorB(seed=seed, config=cfg).generate(N_PER_CLASS)
            imgs = [r[2] for r in recs]
            masks = [r[3] for r in recs]
            summ, _ = score_benchmark("x", imgs, masks, reg, names, log=None)
            v = [summ[n]["mean_dice"] for n in names]
            rm = spearman_permutation(v, med_v, n_perm=4000)[0]
            rf = spearman_permutation(v, fus_v, n_perm=4000)[0]
            cells[f"{amp}|{seed}"] = {"amplitude": amp, "seed": seed,
                                      "rho_medetec": rm, "rho_fuseg": rf,
                                      "mean_rho": (rm + rf) / 2}
            print(f"  amp {amp:>5.1f}  seed {seed:>6d}  "
                  f"rho_med {rm:+.3f}  rho_fuseg {rf:+.3f}  "
                  f"mean {(rm + rf) / 2:+.3f}", flush=True)

    # --- per level summary ------------------------------------------------
    print("\n" + "=" * 78)
    print("CROSS-SEED SUMMARY  (5 seeds per level)")
    print("=" * 78)
    print(f"{'amplitude':>10}{'mean rho':>12}{'SD':>8}{'min':>8}{'max':>8}"
          f"{'% ceiling':>11}")
    summary = {}
    for amp in LEVELS:
        vals = [c["mean_rho"] for c in cells.values() if c["amplitude"] == amp]
        vm = [c["rho_medetec"] for c in cells.values() if c["amplitude"] == amp]
        vf = [c["rho_fuseg"] for c in cells.values() if c["amplitude"] == amp]
        summary[amp] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)),
                        "mean_medetec": float(np.mean(vm)),
                        "sd_medetec": float(np.std(vm, ddof=1)),
                        "mean_fuseg": float(np.mean(vf)),
                        "sd_fuseg": float(np.std(vf, ddof=1))}
        s = summary[amp]
        print(f"{amp:>10.1f}{s['mean']:>12.3f}{s['sd']:>8.3f}"
              f"{s['min']:>8.3f}{s['max']:>8.3f}{s['mean'] / ceiling:>10.0%}")

    print(f"\n{'':10}{'vs Medetec':>22}{'vs FUSeg':>22}")
    print(f"{'amplitude':>10}{'mean':>12}{'SD':>10}{'mean':>12}{'SD':>10}")
    for amp in LEVELS:
        s = summary[amp]
        print(f"{amp:>10.1f}{s['mean_medetec']:>12.3f}{s['sd_medetec']:>10.3f}"
              f"{s['mean_fuseg']:>12.3f}{s['sd_fuseg']:>10.3f}")

    # --- trend, per seed --------------------------------------------------
    print("\nmonotone trend tested WITHIN each seed:")
    per_seed = []
    for seed in SEEDS:
        xs = [amp for amp in LEVELS]
        ys = [cells[f"{amp}|{seed}"]["mean_rho"] for amp in LEVELS]
        r, p = spearman_permutation(xs, ys, n_perm=20000)
        per_seed.append(r)
        print(f"  seed {seed:>6d}:  Spearman {r:+.3f}  (p {p:.4f})")
    print(f"  all five positive: {all(r > 0 for r in per_seed)}")
    print(f"  median across seeds: {np.median(per_seed):+.3f}")

    # trend on the seed-averaged curve
    r_avg, p_avg = spearman_permutation(
        LEVELS, [summary[a]["mean"] for a in LEVELS], n_perm=20000)
    print(f"\ntrend on the seed-averaged curve: Spearman {r_avg:+.3f}, "
          f"p {p_avg:.4f}")

    blob = {"cells": cells, "summary": {str(k): v for k, v in summary.items()},
            "levels": LEVELS, "seeds": SEEDS, "ceiling": ceiling,
            "per_seed_trend": per_seed,
            "averaged_trend": {"rho": r_avg, "p": p_avg}}
    path = os.path.join(OUT, "cross_seed.json")
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

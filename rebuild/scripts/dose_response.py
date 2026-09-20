#!/usr/bin/env python3
"""Dose-response: is predictive validity a monotone function of a MEASURABLE
image property?

This closes a limitation the manuscript explicitly flagged:

    Ablation granularity. Background texture was ablated as a binary presence
    or absence. A dose-response analysis over texture amplitude would establish
    the relationship more precisely than the present on/off comparison.

Two one-factor-at-a-time sweeps over generator B:

  A. lesion-to-surround colour contrast (`lesion_sat_delta`), swept across the
     measured clinical value (~33), generator A's value (~56), and beyond.
  B. background texture amplitude (`mottle_amplitude`), swept from zero (which
     is generator A's uniform assumption) upward.

At each level: generate the benchmark, score all 22 methods by Dice, and
correlate the resulting ranking against both clinical datasets. The measured
contrast of the generated images is reported alongside the nominal setting, so
the x-axis is an image property rather than a knob.

Why this is stronger than the binary ablation: a monotone relationship between
a property you can MEASURE on any benchmark and that benchmark's predictive
validity is a specification a benchmark designer can check their own generator
against. The binary version only says which of two designs was better.

Writes results/dose_response.json.
"""

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_b import GeneratorB, GenBConfig       # noqa: E402
from woundbench.methods import METHOD_NAMES                      # noqa: E402
from woundbench.metrics import spearman_permutation              # noqa: E402

from rank_validity import DATA, OUT, load_pairs, score_benchmark  # noqa: E402

# Measured on Medetec (n=152) and FUSeg validation (n=114); see STATUS.md.
CLINICAL_SAT_DELTA = 33.0
GEN_A_SAT_DELTA = 55.8

CONTRAST_LEVELS = [12.0, 22.0, 33.0, 45.0, 56.0, 70.0, 85.0]
TEXTURE_LEVELS = [0.0, 2.0, 5.0, 11.0, 18.0, 28.0]


def measured_contrast(images, masks):
    """Actual lesion-vs-perilesional saturation and a* delta of a benchmark."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    sat, a_st = [], []
    for im, m in zip(images, masks):
        if m.sum() < 50 or (m == 0).sum() < 50:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(float)
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(float)
        ring = (cv2.dilate(m, k) > 0) & (m == 0)
        if ring.sum() < 50:
            ring = (m == 0)
        sat.append(hsv[..., 1][m > 0].mean() - hsv[..., 1][ring].mean())
        a_st.append(lab[..., 1][m > 0].mean() - lab[..., 1][ring].mean())
    return float(np.mean(sat)), float(np.mean(a_st))


def build_variant(cfg, seed, n_per_class):
    recs = GeneratorB(seed=seed, config=cfg).generate(n_per_class)
    return [r[2] for r in recs], [r[3] for r in recs]


def run_sweep(label, field, levels, registry, names, med_v, fus_v,
              seed, n_per_class):
    print(f"\n### sweep {label}: {field}", flush=True)
    rows = []
    for lv in levels:
        cfg = replace(GenBConfig(), **{field: lv})
        imgs, masks = build_variant(cfg, seed, n_per_class)
        sat_d, a_d = measured_contrast(imgs, masks)
        summ, _ = score_benchmark(label, imgs, masks, registry, names, log=None)
        v = [summ[n]["mean_dice"] for n in names]
        r_med, p_med = spearman_permutation(v, med_v)
        r_fus, p_fus = spearman_permutation(v, fus_v)
        mean_rho = (r_med + r_fus) / 2
        rows.append({field: lv, "measured_sat_delta": sat_d,
                     "measured_a_delta": a_d,
                     "rho_medetec": r_med, "p_medetec": p_med,
                     "rho_fuseg": r_fus, "p_fuseg": p_fus,
                     "mean_rho": mean_rho,
                     "per_method_dice": {n: summ[n]["mean_dice"] for n in names}})
        print(f"  {field}={lv:>6.1f}  measured sat delta {sat_d:+6.1f}  "
              f"rho_med {r_med:+.3f}  rho_fuseg {r_fus:+.3f}  "
              f"mean {mean_rho:+.3f}", flush=True)
    return rows


def spearman_trend(x, y):
    """Monotonicity of the dose-response curve itself."""
    return spearman_permutation(x, y, n_perm=20000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--fuseg-n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    with open(os.path.join(OUT, "models.pkl"), "rb") as f:
        registry = pickle.load(f)["registry"]
    names = METHOD_NAMES
    print(f"{len(names)} methods, threshold {registry.learned_threshold:.2f}",
          flush=True)

    print("scoring clinical reference rankings", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    med, _ = score_benchmark("medetec", me_i, me_m, registry, names, log=None)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, args.fuseg_n)
    fus, _ = score_benchmark("fuseg", fu_i, fu_m, registry, names, log=None)
    med_v = [med[n]["mean_dice"] for n in names]
    fus_v = [fus[n]["mean_dice"] for n in names]
    print(f"  clinical anchor: medetec vs fuseg rho "
          f"{spearman_permutation(med_v, fus_v)[0]:+.3f}", flush=True)

    contrast = run_sweep("contrast", "lesion_sat_delta", CONTRAST_LEVELS,
                         registry, names, med_v, fus_v, args.seed,
                         args.per_class)
    texture = run_sweep("texture", "mottle_amplitude", TEXTURE_LEVELS,
                        registry, names, med_v, fus_v, args.seed,
                        args.per_class)

    # --- trend tests -----------------------------------------------------
    print("\n" + "=" * 74)
    print("DOSE-RESPONSE TRENDS")
    print("=" * 74)

    cx = [r["measured_sat_delta"] for r in contrast]
    cy = [r["mean_rho"] for r in contrast]
    r_c, p_c = spearman_trend(cx, cy)
    print(f"\ncontrast sweep: measured saturation delta vs mean rank agreement")
    print(f"  {'sat delta':>11}{'mean rho':>11}{'rho_med':>10}{'rho_fuseg':>11}")
    for r in contrast:
        print(f"  {r['measured_sat_delta']:>11.1f}{r['mean_rho']:>11.3f}"
              f"{r['rho_medetec']:>10.3f}{r['rho_fuseg']:>11.3f}")
    print(f"  monotone trend: Spearman {r_c:+.3f}, p {p_c:.4f}")
    print(f"  (clinical measures ~{CLINICAL_SAT_DELTA:.0f}; "
          f"generator A sits at ~{GEN_A_SAT_DELTA:.0f})")

    tx = [r["mottle_amplitude"] for r in texture]
    ty = [r["mean_rho"] for r in texture]
    r_t, p_t = spearman_trend(tx, ty)
    print(f"\ntexture sweep: mottling amplitude vs mean rank agreement")
    print(f"  {'amplitude':>11}{'mean rho':>11}{'rho_med':>10}{'rho_fuseg':>11}")
    for r in texture:
        print(f"  {r['mottle_amplitude']:>11.1f}{r['mean_rho']:>11.3f}"
              f"{r['rho_medetec']:>10.3f}{r['rho_fuseg']:>11.3f}")
    print(f"  monotone trend: Spearman {r_t:+.3f}, p {p_t:.4f}")
    print(f"  (amplitude 0 is generator A's uniform-background assumption)")

    blob = {"methods": names,
            "clinical": {"medetec": med_v, "fuseg": fus_v,
                         "sat_delta_target": CLINICAL_SAT_DELTA},
            "contrast_sweep": contrast, "texture_sweep": texture,
            "trends": {"contrast": {"rho": r_c, "p": p_c},
                       "texture": {"rho": r_t, "p": p_t}},
            "seed": args.seed, "n_per_class": args.per_class}
    path = os.path.join(OUT, "dose_response.json")
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does profile distance to clinical predict validity ACROSS generator families?

This is the test the single-statistic version failed. Every benchmark whose rank
agreement has already been measured is re-described by its 12-dimensional
background profile, the distance from that profile to the clinical profile is
computed, and the distance is correlated against observed rank agreement.

The point is the ACROSS-FAMILY correlation. Generator A, B and C were built on
different principles; if profile distance orders all of them correctly, the
descriptor is a property of images. If it only works within families, it is not.

No method scoring happens here. Every rank agreement value is read from results
already on disk, so this is a re-description of existing measurements rather
than a new experiment that could be tuned to succeed.

Writes results/profile_test.json.
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import profile as P                            # noqa: E402
from woundbench.generator_a import build as build_a            # noqa: E402
from woundbench.generator_b import GeneratorB, GenBConfig      # noqa: E402
from woundbench.generator_c import build as build_c            # noqa: E402
from woundbench.metrics import spearman_permutation            # noqa: E402
from rank_validity import DATA, OUT, load_pairs                # noqa: E402

N_PER_CLASS = 15
SEED = 2024


def clinical_profiles():
    me = load_pairs(os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
                    os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    fu = load_pairs(os.path.join(DATA, "Foot Ulcer Segmentation Challenge",
                                 "validation", "images"),
                    os.path.join(DATA, "Foot Ulcer Segmentation Challenge",
                                 "validation", "labels"), 0, 120)
    pm = P.profile_of_set(*me)
    pf = P.profile_of_set(*fu)
    ref = {k: float(np.sqrt(max(pm[k], 1e-6) * max(pf[k], 1e-6)))
           for k in P.PROFILE_KEYS}   # geometric mean of the two references
    return pm, pf, ref


def observed_validity():
    """Every (benchmark, mean rho) pair already measured, from disk."""
    out = []

    rv = json.load(open(os.path.join(OUT, "rank_validity.json")))["rank_agreement"]
    out.append(("A", None,
                (rv["generator_a__medetec"]["rho"] + rv["generator_a__fuseg"]["rho"]) / 2))

    # generator B: cross-seed means where available, else the single-seed sweep
    cs = {}
    p = os.path.join(OUT, "cross_seed_cells.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        for a in sorted({r["amplitude"] for r in rows}):
            cs[a] = float(np.mean([r["mean_rho"] for r in rows
                                   if r["amplitude"] == a]))
    dr = json.load(open(os.path.join(OUT, "dose_response.json")))
    for r in dr["texture_sweep_extended"]:
        a = r["mottle_amplitude"]
        if a > 55:
            continue      # off-distribution, excluded everywhere else too
        out.append(("B", a, cs.get(a, r["mean_rho"])))

    p = os.path.join(OUT, "genc_cells.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        for a in sorted({r["amp"] for r in rows}):
            out.append(("C", a, float(np.mean([r["mean_rho"] for r in rows
                                               if r["amp"] == a]))))
    return out


def build_benchmark(fam, amp):
    if fam == "A":
        i, m, _ = build_a(n_per_class=N_PER_CLASS)
        return i, m
    if fam == "B":
        recs = GeneratorB(seed=SEED,
                          config=replace(GenBConfig(), mottle_amplitude=amp)
                          ).generate(N_PER_CLASS)
        return [r[2] for r in recs], [r[3] for r in recs]
    i, m, _ = build_c(seed=SEED, n_per_class=N_PER_CLASS, noise_amplitude=amp)
    return i, m


def main():
    print("clinical profiles", flush=True)
    pm, pf, ref = clinical_profiles()
    print(f"{'entry':>8}{'Medetec':>10}{'FUSeg':>10}{'reference':>12}")
    for k in P.PROFILE_KEYS:
        print(f"{k:>8}{pm[k]:>10.2f}{pf[k]:>10.2f}{ref[k]:>12.2f}")

    obs = observed_validity()
    print(f"\n{len(obs)} benchmarks with measured rank agreement\n", flush=True)

    rows = []
    print(f"{'benchmark':<16}{'distance':>10}{'mean rho':>10}")
    print("-" * 36)
    for fam, amp, rho in obs:
        imgs, masks = build_benchmark(fam, amp)
        prof = P.profile_of_set(imgs, masks)
        d = P.distance(prof, ref)
        name = f"{fam}" + (f" amp{amp:g}" if amp is not None else "")
        rows.append({"family": fam, "amp": amp, "name": name,
                     "distance": d, "mean_rho": rho, "profile": prof})
        print(f"{name:<16}{d:>10.3f}{rho:>10.3f}", flush=True)

    d = [r["distance"] for r in rows]
    v = [r["mean_rho"] for r in rows]
    r_all, p_all = spearman_permutation(d, v)
    print("\n" + "=" * 62)
    print("ACROSS ALL GENERATOR FAMILIES")
    print("=" * 62)
    print(f"  profile distance vs rank agreement: Spearman {r_all:+.3f}, "
          f"p {p_all:.4f}   (n = {len(rows)})")
    print("  negative is the predicted direction: closer profile, higher agreement")

    # the single statistic, on the same points, for comparison
    from woundbench.imagestats import texture_of_set
    lum = []
    for r, (fam, amp, _) in zip(rows, obs):
        imgs, masks = build_benchmark(fam, amp)
        lum.append(texture_of_set(imgs, masks)[0])
    r_lum, p_lum = spearman_permutation(lum, v)
    print(f"\n  for comparison, single luminance statistic vs agreement: "
          f"Spearman {r_lum:+.3f}, p {p_lum:.4f}")

    for fam in ("B", "C"):
        sub = [(x["distance"], x["mean_rho"]) for x in rows if x["family"] == fam]
        if len(sub) >= 3:
            rr, pp = spearman_permutation([s[0] for s in sub],
                                          [s[1] for s in sub])
            print(f"  within generator {fam} only: Spearman {rr:+.3f}, p {pp:.4f}")

    # which profile entries separate the families
    print("\n" + "=" * 62)
    print("WHERE EACH FAMILY DEVIATES FROM CLINICAL (log ratio)")
    print("=" * 62)
    print(f"{'entry':>8}" + "".join(f"{f:>10}" for f in ("A", "B", "C")))
    reps = {}
    for fam in ("A", "B", "C"):
        cand = [x for x in rows if x["family"] == fam]
        reps[fam] = min(cand, key=lambda x: x["distance"])
    for k in P.PROFILE_KEYS:
        line = f"{k:>8}"
        for fam in ("A", "B", "C"):
            lr = np.log(max(reps[fam]["profile"][k], 1e-3) / max(ref[k], 1e-3))
            line += f"{lr:>+10.2f}"
        print(line)
    print("\n(0 means matched to clinical; +1 is e times too much, -1 too little)")

    json.dump({"reference": ref, "medetec": pm, "fuseg": pf, "rows": rows,
               "across_families": {"rho": r_all, "p": p_all},
               "single_statistic": {"rho": r_lum, "p": p_lum}},
              open(os.path.join(OUT, "profile_test.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'profile_test.json')}")


if __name__ == "__main__":
    main()

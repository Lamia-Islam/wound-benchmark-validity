#!/usr/bin/env python3
"""Cross-seed variance, resumable.

Each completed cell is appended to results/cross_seed_cells.jsonl immediately,
and cells already present are skipped, so a container restart costs at most one
cell rather than the whole run.

Clinical reference rankings are cached too, since scoring 272 clinical images
with 22 methods is the single most expensive step and does not depend on the
sweep.
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

LEVELS = [0.0, 11.0, 28.0, 40.0]
SEEDS = [2024, 7, 101, 555, 31337]
N_PER_CLASS = 15
CELLS = os.path.join(OUT, "cross_seed_cells.jsonl")
CLIN = os.path.join(OUT, "clinical_ranking_cache.json")


def load_done():
    done = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[(r["amplitude"], r["seed"])] = r
    return done


def clinical_rankings(reg, names):
    if os.path.exists(CLIN):
        c = json.load(open(CLIN))
        print("  clinical rankings loaded from cache", flush=True)
        return c["medetec"], c["fuseg"]
    print("  scoring clinical rankings (cached after this)", flush=True)
    me_i, me_m = load_pairs(
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
        os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"))
    med, _ = score_benchmark("me", me_i, me_m, reg, names, log=None)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, 120)
    fus, _ = score_benchmark("fu", fu_i, fu_m, reg, names, log=None)
    m = [med[n]["mean_dice"] for n in names]
    f = [fus[n]["mean_dice"] for n in names]
    json.dump({"medetec": m, "fuseg": f}, open(CLIN, "w"))
    return m, f


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    med_v, fus_v = clinical_rankings(reg, names)
    ceiling = spearman_permutation(med_v, fus_v)[0]
    print(f"  clinical ceiling rho = {ceiling:+.3f}", flush=True)

    done = load_done()
    todo = [(a, s) for a in LEVELS for s in SEEDS if (a, s) not in done]
    print(f"  {len(done)} cells already done, {len(todo)} to go\n", flush=True)

    for amp, seed in todo:
        cfg = replace(GenBConfig(), mottle_amplitude=amp)
        recs = GeneratorB(seed=seed, config=cfg).generate(N_PER_CLASS)
        summ, _ = score_benchmark("x", [r[2] for r in recs], [r[3] for r in recs],
                                  reg, names, log=None)
        v = [summ[n]["mean_dice"] for n in names]
        rm = spearman_permutation(v, med_v, n_perm=4000)[0]
        rf = spearman_permutation(v, fus_v, n_perm=4000)[0]
        row = {"amplitude": amp, "seed": seed, "rho_medetec": rm,
               "rho_fuseg": rf, "mean_rho": (rm + rf) / 2}
        with open(CELLS, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  amp {amp:>5.1f}  seed {seed:>6d}  rho_med {rm:+.3f}  "
              f"rho_fuseg {rf:+.3f}  mean {(rm + rf) / 2:+.3f}", flush=True)

    # ---- summary ---------------------------------------------------------
    done = load_done()
    print("\n" + "=" * 74)
    print("CROSS-SEED SUMMARY")
    print("=" * 74)
    print(f"{'amplitude':>10}{'n':>4}{'mean':>9}{'SD':>8}{'min':>8}{'max':>8}"
          f"{'% ceiling':>11}")
    summary = {}
    for amp in LEVELS:
        vals = [r["mean_rho"] for (a, _), r in done.items() if a == amp]
        if not vals:
            continue
        v = np.array(vals)
        summary[amp] = {"n": len(v), "mean": float(v.mean()),
                        "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "min": float(v.min()), "max": float(v.max())}
        s = summary[amp]
        sd = f"{s['sd']:>8.3f}" if s["sd"] is not None else f"{'-':>8}"
        print(f"{amp:>10.1f}{s['n']:>4d}{s['mean']:>9.3f}{sd}"
              f"{s['min']:>8.3f}{s['max']:>8.3f}{s['mean'] / ceiling:>10.0%}")

    complete = [a for a in LEVELS if summary.get(a, {}).get("n", 0) == len(SEEDS)]
    if len(complete) >= 3:
        r, p = spearman_permutation(complete,
                                    [summary[a]["mean"] for a in complete],
                                    n_perm=20000)
        print(f"\ntrend on seed-averaged curve ({len(complete)} complete "
              f"levels): Spearman {r:+.3f}, p {p:.4f}")

        # separation test: does the lowest level's seed range overlap the top?
        lo, hi = summary[complete[0]], summary[complete[-1]]
        print(f"\nseparation check:")
        print(f"  amp {complete[0]:g}: {lo['mean']:.3f} "
              f"(range {lo['min']:.3f}-{lo['max']:.3f})")
        print(f"  amp {complete[-1]:g}: {hi['mean']:.3f} "
              f"(range {hi['min']:.3f}-{hi['max']:.3f})")
        print(f"  ranges overlap: {not (hi['min'] > lo['max'])}")

        # per-seed monotonicity
        per_seed = []
        for s in SEEDS:
            ys = [done[(a, s)]["mean_rho"] for a in complete if (a, s) in done]
            if len(ys) == len(complete):
                per_seed.append(spearman_permutation(complete, ys,
                                                     n_perm=4000)[0])
        if per_seed:
            print(f"\nwithin-seed trend, {len(per_seed)} seeds: "
                  f"median {np.median(per_seed):+.3f}, "
                  f"all positive: {all(x > 0 for x in per_seed)}")

    json.dump({"summary": {str(k): v for k, v in summary.items()},
               "ceiling": ceiling, "levels": LEVELS, "seeds": SEEDS},
              open(os.path.join(OUT, "cross_seed.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'cross_seed.json')}")


if __name__ == "__main__":
    main()

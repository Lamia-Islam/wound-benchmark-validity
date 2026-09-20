#!/usr/bin/env python3
"""Repeat the framing curve across generator seeds.

Every point on the framing curve is a trend computed over the seven contrast
benchmarks, and those seven benchmarks were built with a single seed (2024).
Seed count has already overturned two conclusions in this project, so the curve
does not get to stand until it survives the same treatment.

This rebuilds the contrast sweep at three further seeds and recomputes, for each
seed independently:

  - the verdict of each of the eight cropped references,
  - the monotonicity of verdict against lesion fraction,
  - where the verdict changes sign,
  - the predicted and measured verdicts of the three real clinical datasets.

Nothing about the references is rebuilt. The eight crop vectors come from
framing_law_cells.jsonl and the clinical vectors from the ranking cache, so the
only thing varying is the synthetic side.

Resumable via results/framing_seed_cells.jsonl.
"""

import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_b import GeneratorB, GenBConfig        # noqa: E402
from woundbench.methods import METHOD_NAMES                      # noqa: E402
from woundbench.metrics import spearman_permutation              # noqa: E402
from rank_validity import OUT, score_benchmark                   # noqa: E402
from dose_response import CONTRAST_LEVELS, measured_contrast     # noqa: E402

SEEDS = [7, 101, 555]
N_PER_CLASS = 15
CELLS = os.path.join(OUT, "framing_seed_cells.jsonl")
MEASURED_FRACTION = {"fuseg": 0.0109, "azh": 0.0285, "medetec": 0.1024}


def done():
    d = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            if line.strip():
                r = json.loads(line)
                d[(r["seed"], r["level"])] = r
    return d


def verdict(sweep, ref, names):
    """Trend of rank agreement against measured contrast, for one reference."""
    x = [s["sat"] for s in sweep]
    agree = [spearman_permutation([s["dice"][n] for n in names], ref,
                                  n_perm=4000)[0] for s in sweep]
    r, p = spearman_permutation(x, agree, n_perm=20000)
    return r, p, agree


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES

    crops = [json.loads(l) for l in open(
        os.path.join(OUT, "framing_law_cells.jsonl")) if l.strip()]
    crops.sort(key=lambda r: r["achieved"])
    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache.json")))

    D = done()
    todo = [(s, lv) for s in SEEDS for lv in CONTRAST_LEVELS
            if (s, lv) not in D]
    print(f"{len(D)} contrast cells done, {len(todo)} to go", flush=True)

    for seed, lv in todo:
        cfg = replace(GenBConfig(), lesion_sat_delta=lv)
        recs = GeneratorB(seed=seed, config=cfg).generate(N_PER_CLASS)
        imgs = [r[2] for r in recs]
        masks = [r[3] for r in recs]
        sat, _ = measured_contrast(imgs, masks)
        summ, _ = score_benchmark("c", imgs, masks, reg, names, log=None)
        row = {"seed": seed, "level": lv, "sat": sat,
               "dice": {n: summ[n]["mean_dice"] for n in names}}
        open(CELLS, "a").write(json.dumps(row) + "\n")
        print(f"  seed {seed:>6d} level {lv:>5.1f}  measured sat "
              f"{sat:+6.1f}", flush=True)

    D = done()
    seeds_ready = [s for s in SEEDS
                   if all((s, lv) in D for lv in CONTRAST_LEVELS)]
    # seed 2024 comes from the original sweep, already on disk
    dr = json.load(open(os.path.join(OUT, "dose_response.json")))
    sweeps = {2024: [{"sat": r["measured_sat_delta"],
                      "dice": r["per_method_dice"]}
                     for r in dr["contrast_sweep"]]}
    for s in seeds_ready:
        sweeps[s] = [{"sat": D[(s, lv)]["sat"], "dice": D[(s, lv)]["dice"]}
                     for lv in CONTRAST_LEVELS]
    print(f"\n{len(sweeps)} seeds available: {sorted(sweeps)}", flush=True)
    if len(sweeps) < 2:
        return

    out = {}
    print("\n" + "=" * 78)
    print("THE CURVE, PER SEED")
    print("=" * 78)
    hdr = f"{'lesion frac':>12}" + "".join(f"{('seed ' + str(s)):>12}"
                                           for s in sorted(sweeps))
    print(hdr)
    print("-" * len(hdr))
    per_seed_curve = {s: [] for s in sweeps}
    for c in crops:
        line = f"{c['achieved']:>12.4f}"
        for s in sorted(sweeps):
            v = verdict(sweeps[s], [c["per_method_dice"][n] for n in names],
                        names)[0]
            per_seed_curve[s].append(v)
            line += f"{v:>+12.3f}"
        print(line, flush=True)

    frac = [c["achieved"] for c in crops]
    print("\n" + "=" * 78)
    print("MONOTONICITY AND SIGN CHANGE, PER SEED")
    print("=" * 78)
    print(f"{'seed':>8}{'Spearman':>12}{'p':>10}{'sign change between':>28}")
    monos, crossings = [], []
    for s in sorted(sweeps):
        y = per_seed_curve[s]
        r, p = spearman_permutation(frac, y, n_perm=20000)
        monos.append(r)
        cr = None
        for i in range(len(y) - 1):
            if y[i] * y[i + 1] < 0:
                cr = (frac[i], frac[i + 1])
                break
        if cr:
            crossings.append(cr)
        txt = f"{cr[0]:.4f} and {cr[1]:.4f}" if cr else "no sign change"
        print(f"{s:>8}{r:>+12.3f}{p:>10.4f}{txt:>28}")
        out[str(s)] = {"curve": y, "mono_rho": r, "mono_p": p,
                       "crossing": cr}

    print(f"\nmonotonicity across {len(monos)} seeds: mean {np.mean(monos):+.3f}, "
          f"range {min(monos):+.3f} to {max(monos):+.3f}")
    if crossings:
        los = [c[0] for c in crossings]
        his = [c[1] for c in crossings]
        print(f"sign change lies between {min(los):.4f} and {max(his):.4f} "
              f"in every seed ({len(crossings)}/{len(sweeps)} seeds cross)")

    print("\n" + "=" * 78)
    print("HELD-OUT CLINICAL DATASETS, PER SEED")
    print("=" * 78)
    print(f"{'dataset':<10}{'frac':>9}" +
          "".join(f"{('s' + str(s)):>10}" for s in sorted(sweeps)) +
          f"{'mean':>9}{'sd':>8}{'predicted':>11}{'residual':>10}")
    overlay = {}
    for k, f0 in sorted(MEASURED_FRACTION.items(), key=lambda kv: kv[1]):
        if k not in cache:
            continue
        meas, preds = [], []
        for s in sorted(sweeps):
            m = verdict(sweeps[s], cache[k], names)[0]
            meas.append(m)
            preds.append(float(np.interp(f0, frac, per_seed_curve[s])))
        mm, pp = float(np.mean(meas)), float(np.mean(preds))
        sd = float(np.std(meas, ddof=1)) if len(meas) > 1 else 0.0
        overlay[k] = {"fraction": f0, "measured": meas, "predicted": preds,
                      "mean_measured": mm, "mean_predicted": pp,
                      "sd_measured": sd, "residual": mm - pp}
        print(f"{k:<10}{f0:>9.4f}" +
              "".join(f"{v:>+10.3f}" for v in meas) +
              f"{mm:>+9.3f}{sd:>8.3f}{pp:>+11.3f}{mm - pp:>+10.3f}")

    print("\nFUSeg is not a held-out check; the curve is built from its crops.")
    for k in ("medetec", "azh"):
        if k in overlay:
            o = overlay[k]
            print(f"{k}: measured {o['mean_measured']:+.3f} "
                  f"(sd {o['sd_measured']:.3f} over {len(sweeps)} seeds), "
                  f"curve predicts {o['mean_predicted']:+.3f}, "
                  f"residual {o['residual']:+.3f}")

    json.dump({"per_seed": out, "overlay": overlay,
               "fractions": frac, "seeds": sorted(sweeps)},
              open(os.path.join(OUT, "framing_law_seeds.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'framing_law_seeds.json')}")


if __name__ == "__main__":
    main()

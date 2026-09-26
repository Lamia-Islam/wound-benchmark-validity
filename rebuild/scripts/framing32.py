#!/usr/bin/env python3
"""The framing result, rebuilt at n=32, across seeds, under two metrics.

Three objections are answered in one experiment.

  The pool. The original curve was measured over 21 thresholding algorithms and
  one small U-Net. Everything here uses the 32-method pool, which includes ten
  modern architectures.

  The seeds. Every point on the original curve rested on contrast benchmarks
  built with a single seed. Here the contrast sweep is rebuilt at four seeds and
  the curve is computed independently for each.

  The metric. Dice is biased by region size and the framing experiment varies
  region size, so the two are confounded by construction. Every quantity is
  computed under Dice and under normalised surface Dice at 2 px, which is
  boundary-sensitive and does not carry the same bias. If the transition
  survives under NSD, the confound is excluded.

Resumable. Crop references are cached in framing32_refs.jsonl and contrast
benchmarks in framing32_sweep.jsonl.
"""

import json
import os
import pickle
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import deep as D                                   # noqa: E402
from woundbench.combined import CombinedRegistry                   # noqa: E402
from woundbench.generator_b import GeneratorB, GenBConfig          # noqa: E402
from woundbench.metrics import dice, nsd, spearman_permutation     # noqa: E402
from rank_validity import DATA, OUT, load_pairs                    # noqa: E402
from framing_test import crop_to_fraction                          # noqa: E402
from dose_response import CONTRAST_LEVELS, measured_contrast       # noqa: E402

TARGETS = [0.005, 0.010, 0.020, 0.035, 0.055, 0.080, 0.110, 0.160]
SEEDS = [2024, 7, 101, 555]
N_FUSEG = 120
N_PER_CLASS = 15
METRICS = ("dice", "nsd")
REFS = os.path.join(OUT, "framing32_refs.jsonl")
SWEEP = os.path.join(OUT, "framing32_sweep.jsonl")


def registry():
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(OUT, "deep")).load()
    thr = json.load(open(os.path.join(OUT, "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    return reg, reg.available()


def score_two(images, masks, reg, names, label="", log=True):
    """Mean Dice and mean NSD per method, in one pass over the images."""
    acc = {m: {n: [] for n in names} for m in METRICS}
    t0 = time.time()
    for k, (im, gt) in enumerate(zip(images, masks)):
        out = reg.run_all(im, names=names)
        for n, pred in out.items():
            acc["dice"][n].append(dice(pred, gt))
            acc["nsd"][n].append(nsd(pred, gt))
        if log and (k + 1) % 40 == 0:
            print(f"      {label} {k + 1}/{len(images)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {m: {n: float(np.mean(acc[m][n])) for n in names} for m in METRICS}


def _load(path, key):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                d[key(r)] = r
    return d


def build_refs(reg, names, fu_i, fu_m, todo=None):
    have = _load(REFS, lambda r: r["target"])
    for t in (todo or TARGETS):
        if t in have:
            continue
        ci, cm = [], []
        for im, ms in zip(fu_i, fu_m):
            a, b = crop_to_fraction(im, ms, target_frac=t)
            if a is not None:
                ci.append(a)
                cm.append(b)
        if len(ci) < 40:
            continue
        print(f"  crop {t:.3f}: {len(ci)} images", flush=True)
        sc = score_two(ci, cm, reg, names, label=f"crop{t:.3f}")
        row = {"target": t,
               "achieved": float(np.median([m.mean() for m in cm])),
               "n": len(ci), "scores": sc}
        open(REFS, "a").write(json.dumps(row) + "\n")
        have[t] = row
    return have


def build_sweep(reg, names, todo=None):
    have = _load(SWEEP, lambda r: (r["seed"], r["level"]))
    for seed in (todo or SEEDS):
        for lv in CONTRAST_LEVELS:
            if (seed, lv) in have:
                continue
            cfg = replace(GenBConfig(), lesion_sat_delta=lv)
            recs = GeneratorB(seed=seed, config=cfg).generate(N_PER_CLASS)
            imgs = [r[2] for r in recs]
            masks = [r[3] for r in recs]
            sat, _ = measured_contrast(imgs, masks)
            sc = score_two(imgs, masks, reg, names, log=False)
            row = {"seed": seed, "level": lv, "sat": sat, "scores": sc}
            open(SWEEP, "a").write(json.dumps(row) + "\n")
            have[(seed, lv)] = row
            print(f"  sweep seed {seed:>5d} level {lv:>5.1f}  sat {sat:+6.1f}",
                  flush=True)
    return have


def verdict(sweep_rows, ref_vec, names, metric):
    from scipy.stats import spearmanr
    x = [r["sat"] for r in sweep_rows]
    # only the rho is needed for the agreement values; the permutation test is
    # reserved for the trend itself, which is the quantity being reported
    agree = [float(spearmanr([r["scores"][metric][n] for n in names],
                             ref_vec).statistic)
             for r in sweep_rows]
    r, p = spearman_permutation(x, agree, n_perm=20000)
    return r, p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None, help="comma list of crop targets")
    ap.add_argument("--seeds", default=None, help="comma list of sweep seeds")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    global TARGETS, SEEDS
    todo_t = ([float(x) for x in args.targets.split(",")]
              if args.targets else list(TARGETS))
    todo_s = ([int(x) for x in args.seeds.split(",")]
              if args.seeds else list(SEEDS))

    reg, names = registry()
    print(f"pool: {len(names)} methods", flush=True)

    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, N_FUSEG)

    refs = build_refs(reg, names, fu_i, fu_m, todo_t)
    sweeps = build_sweep(reg, names, todo_s)

    ready = [s for s in SEEDS
             if all((s, lv) in sweeps for lv in CONTRAST_LEVELS)]
    got = sorted(refs)
    print(f"\n{len(got)}/{len(TARGETS)} crop levels, "
          f"{len(ready)}/{len(SEEDS)} seeds complete", flush=True)
    if len(got) < 4 or not ready:
        print("not enough yet")
        return

    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    frac = [refs[t]["achieved"] for t in got]
    out = {"fractions": frac, "seeds": ready, "curves": {}, "overlay": {}}

    for metric in METRICS:
        print("\n" + "=" * 78)
        print(f"VERDICT CURVE, metric = {metric}, n = {len(names)}")
        print("=" * 78)
        hdr = f"{'lesion frac':>12}" + "".join(f"{('seed ' + str(s)):>12}"
                                               for s in ready) + f"{'mean':>10}"
        print(hdr)
        print("-" * len(hdr))
        per_seed = {s: [] for s in ready}
        for t in got:
            line = f"{refs[t]['achieved']:>12.4f}"
            vals = []
            for s in ready:
                rows = [sweeps[(s, lv)] for lv in CONTRAST_LEVELS]
                v = verdict(rows, [refs[t]["scores"][metric][n] for n in names],
                            names, metric)[0]
                per_seed[s].append(v)
                vals.append(v)
                line += f"{v:>+12.3f}"
            line += f"{np.mean(vals):>+10.3f}"
            print(line, flush=True)

        monos, cross = [], []
        for s in ready:
            r, p = spearman_permutation(frac, per_seed[s], n_perm=20000)
            monos.append(r)
            y = per_seed[s]
            c = None
            for i in range(len(y) - 1):
                if y[i] * y[i + 1] < 0:
                    c = (frac[i], frac[i + 1])
                    break
            cross.append(c)
        print(f"\nmonotonicity per seed: "
              f"{', '.join(format(m, '+.3f') for m in monos)}")
        print(f"  mean {np.mean(monos):+.3f}, range {min(monos):+.3f} "
              f"to {max(monos):+.3f}")
        ok = [c for c in cross if c]
        if ok:
            print(f"sign change in {len(ok)}/{len(ready)} seeds, between "
                  f"{min(c[0] for c in ok):.4f} and {max(c[1] for c in ok):.4f}")
        else:
            print("no sign change in any seed")

        # held-out clinical datasets, averaged over seeds
        print(f"\n{'dataset':<12}{'frac':>9}{'predicted':>12}{'measured':>11}"
              f"{'sd':>8}{'residual':>11}")
        mean_curve = [float(np.mean([per_seed[s][i] for s in ready]))
                      for i in range(len(got))]
        ov = {}
        for k in ("fuseg", "azh", "medetec"):
            if k not in cache:
                continue
            f0 = cache.get(k + "__lesion_fraction")
            meas = [verdict([sweeps[(s, lv)] for lv in CONTRAST_LEVELS],
                            cache[k], names, metric)[0] for s in ready]
            mm = float(np.mean(meas))
            sd = float(np.std(meas, ddof=1)) if len(meas) > 1 else 0.0
            pred = float(np.interp(f0, frac, mean_curve))
            ov[k] = {"fraction": f0, "predicted": pred, "measured": mm,
                     "sd": sd, "residual": mm - pred}
            print(f"{k:<12}{f0:>9.4f}{pred:>+12.3f}{mm:>+11.3f}{sd:>8.3f}"
                  f"{mm - pred:>+11.3f}")
        out["curves"][metric] = {"per_seed": per_seed, "mono": monos,
                                 "crossings": cross, "mean": mean_curve}
        out["overlay"][metric] = ov

    json.dump(out, open(os.path.join(OUT, "framing32.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'framing32.json')}")


if __name__ == "__main__":
    main()

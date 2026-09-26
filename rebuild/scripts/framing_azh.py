#!/usr/bin/env python3
"""The framing curve rebuilt from AZH crops instead of FUSeg crops.

Eleven of the 32 methods were trained on FUSeg training images. The framing
curve in framing32.py is built entirely from crops of FUSeg validation frames,
so although those images are disjoint from training, the distribution is not,
and that makes it the weakest of the paper's three main results.

AZH is the reference no method was trained on. Rebuilding the curve from AZH
crops removes the contamination. The cost is coverage: AZH ships as 224 px
patches with a median lesion fraction of 0.029, and cropping can only raise a
lesion fraction, never lower it, so this curve starts at 0.035 rather than
0.008. The crossover region found on FUSeg, between 0.110 and 0.160, sits
comfortably inside the range AZH can reach, which is what matters.

The contrast sweep is reused unchanged from framing32_sweep.jsonl, four seeds,
so nothing on the synthetic side differs between the two curves. Only the
reference changes.

Held-out placement uses Medetec at lesion fraction 0.102. FUSeg at 0.011 is
below the range this curve covers and is not placed.

Resumable via results/framing_azh_refs.jsonl.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import deep as D                                  # noqa: E402
from woundbench.combined import CombinedRegistry                  # noqa: E402
from woundbench.metrics import spearman_permutation               # noqa: E402
from rank_validity import DATA, OUT, load_pairs                   # noqa: E402
from framing_test import crop_to_fraction                         # noqa: E402
from framing32 import METRICS, SWEEP, _load, score_two            # noqa: E402
from dose_response import CONTRAST_LEVELS                         # noqa: E402

TARGETS = [0.035, 0.055, 0.080, 0.110, 0.160, 0.220, 0.300]
SEEDS = [2024, 7, 101, 555]
N_AZH = 150
REFS = os.path.join(OUT, "framing_azh_refs.jsonl")


def registry():
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(OUT, "deep")).load()
    thr = json.load(open(os.path.join(OUT, "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    return reg, reg.available()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None)
    args = ap.parse_args()
    todo = ([float(x) for x in args.targets.split(",")]
            if args.targets else list(TARGETS))

    reg, names = registry()
    az_i, az_m = load_pairs(
        os.path.join(DATA, "wound_dataset", "test", "images"),
        os.path.join(DATA, "wound_dataset", "test", "labels"), 0, N_AZH)
    print(f"{len(az_i)} AZH frames, pool {len(names)} methods", flush=True)

    have = _load(REFS, lambda r: r["target"])
    for t in todo:
        if t in have:
            continue
        ci, cm = [], []
        for im, ms in zip(az_i, az_m):
            a, b = crop_to_fraction(im, ms, target_frac=t, out=224)
            if a is not None:
                ci.append(a)
                cm.append(b)
        if len(ci) < 40:
            print(f"  target {t}: only {len(ci)} usable crops, skipped",
                  flush=True)
            continue
        ach = float(np.median([m.mean() for m in cm]))
        print(f"  crop {t:.3f}: {len(ci)} images, achieved {ach:.4f}",
              flush=True)
        sc = score_two(ci, cm, reg, names, label=f"azh{t:.3f}")
        row = {"target": t, "achieved": ach, "n": len(ci), "scores": sc}
        open(REFS, "a").write(json.dumps(row) + "\n")
        have[t] = row

    have = _load(REFS, lambda r: r["target"])
    got = sorted(have)
    sweeps = _load(SWEEP, lambda r: (r["seed"], r["level"]))
    ready = [s for s in SEEDS
             if all((s, lv) in sweeps for lv in CONTRAST_LEVELS)]
    print(f"\n{len(got)}/{len(TARGETS)} crop levels, {len(ready)} seeds",
          flush=True)
    if len(got) < 4 or not ready:
        return

    from scipy.stats import spearmanr

    def slope(seed, vec, metric):
        rows = [sweeps[(seed, lv)] for lv in CONTRAST_LEVELS]
        x = np.array([r["sat"] for r in rows])
        y = np.array([float(spearmanr([r["scores"][metric][n] for n in names],
                                      vec).statistic) for r in rows])
        return float(np.polyfit(x, y, 1)[0] * 10.0)

    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    frac = [have[t]["achieved"] for t in got]
    out = {}

    for metric in METRICS:
        print("\n" + "=" * 76)
        print(f"AZH-BUILT FRAMING CURVE   metric = {metric}   n = {len(names)}")
        print("=" * 76)
        print(f"{'lesion frac':>12}" + "".join(f"{('s' + str(s)):>10}"
                                               for s in ready) +
              f"{'mean':>10}{'sd':>8}")
        per = {s: [] for s in ready}
        for t in got:
            vals = []
            for s in ready:
                v = slope(s, [have[t]["scores"][metric][n] for n in names],
                          metric)
                per[s].append(v)
                vals.append(v)
            print(f"{have[t]['achieved']:>12.4f}" +
                  "".join(f"{v:>+10.3f}" for v in vals) +
                  f"{np.mean(vals):>+10.3f}{np.std(vals, ddof=1):>8.3f}",
                  flush=True)

        monos = [spearman_permutation(frac, per[s], n_perm=20000)
                 for s in ready]
        print("\nmonotonicity per seed: " +
              ", ".join(f"{m[0]:+.3f} (p {m[1]:.4f})" for m in monos))
        mean_curve = [float(np.mean([per[s][i] for s in ready]))
                      for i in range(len(got))]
        zc = None
        for i in range(len(mean_curve) - 1):
            if mean_curve[i] * mean_curve[i + 1] < 0:
                zc = (frac[i], frac[i + 1])
                break
        print(f"sign change between {zc[0]:.4f} and {zc[1]:.4f}" if zc
              else "no sign change")

        span = max(mean_curve) - min(mean_curve)
        f0 = cache["medetec__lesion_fraction"]
        ms = [slope(s, cache["medetec"], metric) for s in ready]
        mm, sd = float(np.mean(ms)), float(np.std(ms, ddof=1))
        pred = float(np.interp(f0, frac, mean_curve))
        print(f"\nheld-out Medetec at lesion fraction {f0:.4f}: "
              f"predicted {pred:+.3f}, measured {mm:+.3f} (sd {sd:.3f}), "
              f"residual {mm - pred:+.3f} = {abs(mm - pred) / span:.1%} of range")
        out[metric] = {"fractions": frac, "curve": mean_curve,
                       "per_seed": {str(k): v for k, v in per.items()},
                       "mono": [m[0] for m in monos],
                       "mono_p": [m[1] for m in monos], "zero_cross": zc,
                       "overlay": {"medetec": {"fraction": f0,
                                               "predicted": pred,
                                               "measured": mm, "sd": sd,
                                               "residual": mm - pred}}}

    json.dump(out, open(os.path.join(OUT, "framing_azh.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'framing_azh.json')}")


if __name__ == "__main__":
    main()

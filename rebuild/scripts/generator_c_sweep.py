#!/usr/bin/env python3
"""External validation: does an independently built generator land on the same
texture-to-validity curve?

Generator C shares no code path with generator B (see woundbench/generator_c.py
for the mechanism-by-mechanism differences). If the texture specification is a
property of images rather than of generator B's implementation, C's points
should fall near B's curve when both are plotted against MEASURED texture.

This is the strongest test available without a third-party benchmark: the two
generators are matched on a measured image statistic, not on a shared knob.

Resumable: each cell is appended to results/genc_cells.jsonl.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_c import build as build_c            # noqa: E402
from woundbench.imagestats import texture_of_set               # noqa: E402
from woundbench.methods import METHOD_NAMES                     # noqa: E402
from woundbench.metrics import spearman_permutation             # noqa: E402
from rank_validity import OUT, score_benchmark                  # noqa: E402

AMPS = [0.0, 2.0, 3.5, 5.0, 8.0, 12.0]
SEEDS = [2024, 7, 101]
CELLS = os.path.join(OUT, "genc_cells.jsonl")
CLIN = os.path.join(OUT, "clinical_ranking_cache.json")


def load_done():
    d = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            if line.strip():
                r = json.loads(line)
                d[(r["amp"], r["seed"])] = r
    return d


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    cache = json.load(open(CLIN))
    med_v, fus_v = cache["medetec"], cache["fuseg"]
    ceiling = spearman_permutation(med_v, fus_v)[0]
    print(f"clinical ceiling rho = {ceiling:+.3f}", flush=True)

    done = load_done()
    todo = [(a, s) for a in AMPS for s in SEEDS if (a, s) not in done]
    print(f"{len(done)} done, {len(todo)} to go\n", flush=True)

    for amp, seed in todo:
        imgs, masks, _ = build_c(seed=seed, n_per_class=15, noise_amplitude=amp)
        tex = texture_of_set(imgs, masks)[0]
        summ, _ = score_benchmark("C", imgs, masks, reg, names, log=None)
        v = [summ[n]["mean_dice"] for n in names]
        rm = spearman_permutation(v, med_v, n_perm=4000)[0]
        rf = spearman_permutation(v, fus_v, n_perm=4000)[0]
        row = {"amp": amp, "seed": seed, "texture": tex,
               "rho_medetec": rm, "rho_fuseg": rf, "mean_rho": (rm + rf) / 2,
               "per_method_dice": {n: summ[n]["mean_dice"] for n in names}}
        with open(CELLS, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  amp {amp:>5.1f} seed {seed:>6d}  texture {tex:>6.2f}  "
              f"mean rho {(rm + rf) / 2:+.3f}", flush=True)

    done = load_done()
    print("\n" + "=" * 66)
    print("GENERATOR C, seed-averaged")
    print("=" * 66)
    print(f"{'amp':>7}{'texture':>10}{'n':>4}{'mean rho':>11}{'SD':>8}"
          f"{'% ceiling':>11}")
    rows = []
    for a in AMPS:
        cells = [r for (aa, _), r in done.items() if aa == a]
        if not cells:
            continue
        t = float(np.mean([c["texture"] for c in cells]))
        v = np.array([c["mean_rho"] for c in cells])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        rows.append({"amp": a, "texture": t, "n": len(v),
                     "mean_rho": float(v.mean()), "sd": float(sd)})
        print(f"{a:>7.1f}{t:>10.2f}{len(v):>4d}{v.mean():>11.3f}{sd:>8.3f}"
              f"{v.mean() / ceiling:>10.0%}")

    if len(rows) >= 3:
        r, p = spearman_permutation([x["texture"] for x in rows],
                                    [x["mean_rho"] for x in rows],
                                    n_perm=20000)
        print(f"\ntexture vs validity within generator C: Spearman {r:+.3f}, "
              f"p {p:.4f}")

    # ---- agreement with generator B's curve ------------------------------
    dr = json.load(open(os.path.join(OUT, "dose_response.json")))
    ts = {x["name"]: x for x in
          json.load(open(os.path.join(OUT, "texture_stats_canonical.json")))}
    bt, bv = [], []
    for x in dr["texture_sweep_extended"]:
        key = f"generator B amp={x['mottle_amplitude']:g}"
        if key in ts:
            bt.append(ts[key]["background_texture_masked"])
            bv.append(x["mean_rho"])
    order = np.argsort(bt)
    bt = np.array(bt)[order]
    bv = np.array(bv)[order]

    print("\n" + "=" * 66)
    print("DOES GENERATOR C FALL ON GENERATOR B's CURVE?")
    print("=" * 66)
    print(f"{'texture':>10}{'C observed':>13}{'B curve here':>15}"
          f"{'difference':>13}")
    diffs = []
    for x in rows:
        if bt.min() <= x["texture"] <= bt.max():
            pred = float(np.interp(x["texture"], bt, bv))
            d = x["mean_rho"] - pred
            diffs.append(d)
            print(f"{x['texture']:>10.2f}{x['mean_rho']:>13.3f}"
                  f"{pred:>15.3f}{d:>+13.3f}")
        else:
            print(f"{x['texture']:>10.2f}{x['mean_rho']:>13.3f}"
                  f"{'(outside B range)':>15}{'':>13}")
    if diffs:
        print(f"\nmean absolute difference from B's curve: "
              f"{np.mean(np.abs(diffs)):.3f}")
        print(f"for scale, generator B's own seed SD is about 0.04")

    json.dump({"rows": rows, "ceiling": ceiling,
               "b_curve": {"texture": bt.tolist(), "mean_rho": bv.tolist()}},
              open(os.path.join(OUT, "generator_c.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'generator_c.json')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Non-reducibility, rebuilt at n=32.

Three benchmarks from three generator families, matched on the measured
background texture statistic, scored against the neutral clinical references.
If benchmark validity were a function of that statistic, three benchmarks at the
same value would agree. The n=22 version found 0.121, 0.434 and 0.189.

Neutral references are Medetec and AZH. FUSeg is excluded here because eleven of
the 32 methods were trained on its training split, which inflates their scores
against it and makes it a poor reference for this pool.

Writes results/nonreduce32.json.
"""

import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import deep as D                                  # noqa: E402
from woundbench.combined import CombinedRegistry                  # noqa: E402
from woundbench.generator_a import build as build_a               # noqa: E402
from woundbench.generator_b import GenBConfig, GeneratorB         # noqa: E402
from woundbench.generator_c import build as build_c               # noqa: E402
from woundbench.imagestats import texture_of_set                  # noqa: E402
from woundbench.metrics import spearman_permutation               # noqa: E402
from rank_validity import OUT, score_benchmark                    # noqa: E402

N_PER_CLASS = 15
SEED = 2024
REFS = ("medetec", "azh")
CELLS = os.path.join(OUT, "nonreduce32_cells.jsonl")


def benchmarks():
    """The three matched-texture benchmarks, one per generator family."""
    def gen_a():
        i, m, _ = build_a(n_per_class=N_PER_CLASS)
        return i, m

    def gen_b():
        recs = GeneratorB(seed=SEED,
                          config=replace(GenBConfig(), mottle_amplitude=0.0)
                          ).generate(N_PER_CLASS)
        return [r[2] for r in recs], [r[3] for r in recs]

    def gen_c():
        i, m, _ = build_c(seed=SEED, n_per_class=N_PER_CLASS,
                          noise_amplitude=0.0)
        return i, m

    return [("generator A, uniform drape", gen_a),
            ("generator B, texture ablated", gen_b),
            ("generator C, no noise", gen_c)]


def main():
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(OUT, "deep")).load()
    thr = json.load(open(os.path.join(OUT, "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    names = reg.available()
    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    print(f"pool: {len(names)} methods; references: {', '.join(REFS)}",
          flush=True)

    done = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r

    for label, build in benchmarks():
        if label in done:
            continue
        imgs, masks = build()
        tex = texture_of_set(imgs, masks)[0]
        summ, _ = score_benchmark(label, imgs, masks, reg, names, log=None)
        v = [summ[n]["mean_dice"] for n in names]
        rhos = {k: spearman_permutation(v, cache[k], n_perm=20000)
                for k in REFS}
        row = {"name": label, "texture": float(tex),
               "rho": {k: rhos[k][0] for k in REFS},
               "p": {k: rhos[k][1] for k in REFS},
               "mean_rho": float(np.mean([rhos[k][0] for k in REFS])),
               "dice": {n: summ[n]["mean_dice"] for n in names}}
        open(CELLS, "a").write(json.dumps(row) + "\n")
        done[label] = row
        print(f"  {label:<32} texture {tex:.2f}  "
              f"mean rho {row['mean_rho']:+.3f}", flush=True)

    rows = [done[l] for l, _ in benchmarks() if l in done]
    if len(rows) < 3:
        print("not all three built yet")
        return

    print("\n" + "=" * 78)
    print(f"THREE BENCHMARKS AT MATCHED TEXTURE, n = {len(names)}")
    print("=" * 78)
    print(f"{'benchmark':<34}{'texture':>9}" +
          "".join(f"{('rho ' + k):>13}" for k in REFS) + f"{'mean':>9}")
    for r in rows:
        print(f"{r['name']:<34}{r['texture']:>9.2f}" +
              "".join(f"{r['rho'][k]:>+13.3f}" for k in REFS) +
              f"{r['mean_rho']:>+9.3f}")

    t = [r["texture"] for r in rows]
    v = [r["mean_rho"] for r in rows]
    print(f"\ntexture spread across the three: {max(t) - min(t):.2f}")
    print(f"validity spread at matched texture: {max(v) - min(v):.3f}")
    print("\nIf validity were a function of the measured statistic, three")
    print("benchmarks within %.2f of each other on it would not differ by %.3f."
          % (max(t) - min(t), max(v) - min(v)))

    json.dump({"rows": [{k: x[k] for k in x if k != "dice"} for x in rows],
               "pool_size": len(names), "references": list(REFS)},
              open(os.path.join(OUT, "nonreduce32.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'nonreduce32.json')}")


if __name__ == "__main__":
    main()

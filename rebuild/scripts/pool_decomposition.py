#!/usr/bin/env python3
"""What halves synthetic benchmark validity: the method pool or the reference?

The 98% figure was generator B against Medetec and FUSeg over the original 22
methods. The 49% figure was generator B against Medetec and AZH over 32 methods.
Two things changed at once, so the comparison cannot say which one matters.

This scores generator B (texture amplitudes 11, 28, 40; seeds 2024, 7, 101) with
all 32 methods, stores per-method Dice, and then computes validity in a full
2 x 2 x 3 design:

  pool:       original 22 | 18 untrained | 32 (all)
  reference:  Medetec+FUSeg | Medetec+AZH

each normalised by the clinical-to-clinical ceiling measured over the SAME pool
and the SAME reference pair, so the ceiling never shifts under the comparison.

Writes results/pool_decomposition.json and results/POOL_DECOMPOSITION.md.
Resumable via results/pool_decomposition_cells.jsonl.
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402
from woundbench.generator_b import GenBConfig, GeneratorB         # noqa: E402
from woundbench.methods import METHOD_NAMES                       # noqa: E402
from woundbench.metrics import dice, spearman_permutation         # noqa: E402

OUT = os.path.join(ROOT, "results")
CELLS = os.path.join(OUT, "pool_decomposition_cells.jsonl")
AMPS = [11.0, 28.0, 40.0]
SEEDS = [2024, 7, 101]
N_PER_CLASS = 15


def registry():
    import pickle
    from woundbench import deep as D
    from woundbench.combined import CombinedRegistry
    import torch
    torch.set_num_threads(int(os.environ.get("WB_THREADS", "2")))
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(OUT, "deep")).load()
    thr = json.load(open(os.path.join(OUT, "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    return reg, reg.available()


def done():
    d = {}
    if os.path.exists(CELLS):
        for line in open(CELLS):
            if line.strip():
                r = json.loads(line)
                d[(r["amp"], r["seed"])] = r
    return d


def build(budget):
    import time
    t0 = time.time()
    D = done()
    only = os.environ.get("SEEDS_ONLY")
    seeds = [int(x) for x in only.split(",")] if only else SEEDS
    todo = [(a, s) for a in AMPS for s in seeds if (a, s) not in D]
    if not todo:
        return True
    reg, names = registry()
    for amp, seed in todo:
        recs = GeneratorB(seed=seed, config=replace(GenBConfig(),
                                                    mottle_amplitude=amp)
                          ).generate(N_PER_CLASS)
        acc = {n: [] for n in names}
        for r in recs:
            out = reg.run_all(r[2], names=names)
            for n in names:
                acc[n].append(dice(out[n], r[3]))
        row = {"amp": amp, "seed": seed, "names": names,
               "dice": [float(np.mean(acc[n])) for n in names]}
        open(CELLS, "a").write(json.dumps(row) + "\n")
        print(f"  amp {amp:>4.0f} seed {seed:>5d} scored", flush=True)
        if time.time() - t0 > budget:
            return False
    return True


def ref_vectors(names):
    out = {}
    for k in ("medetec", "fuseg", "azh"):
        d = json.load(open(os.path.join(OUT, "perimage32", k + ".json")))
        assert d["names"] == names
        out[k] = np.asarray(d["dice"], float).mean(axis=0)
    return out


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 540
    if not build(budget) or os.environ.get("SEEDS_ONLY"):
        print("stopping after build step")
        return
    D = done()
    names = next(iter(D.values()))["names"]
    refs = ref_vectors(names)
    trained = lambda n: (ALL_METHOD_FAMILY[n] in ("learned", "CNN")
                         or ALL_METHOD_FAMILY[n].startswith("deep"))
    pools = {
        "original 22": np.array([i for i, n in enumerate(names)
                                 if n in METHOD_NAMES]),
        "18 untrained": np.array([i for i, n in enumerate(names)
                                  if not trained(n)]),
        "all 32": np.arange(len(names)),
    }
    assert len(pools["original 22"]) == 22
    refpairs = {"Medetec + FUSeg": ("medetec", "fuseg"),
                "Medetec + AZH": ("medetec", "azh")}

    md = ["# What halves synthetic benchmark validity: pool or reference?\n",
          "Generator B, three texture amplitudes, three seeds each. Validity is "
          "the mean of the benchmark's Spearman agreement with the two clinical "
          "references, as a percentage of the agreement between those two "
          "references measured over the same method pool.\n"]
    res = {}
    for pname, idx in pools.items():
        for rname, (ra, rb) in refpairs.items():
            ceil = spearman_permutation(refs[ra][idx], refs[rb][idx],
                                        n_perm=20000)
            per_amp = {}
            for amp in AMPS:
                vals = []
                for seed in SEEDS:
                    v = np.asarray(D[(amp, seed)]["dice"])[idx]
                    a = spearman_permutation(v, refs[ra][idx], n_perm=2000)[0]
                    b = spearman_permutation(v, refs[rb][idx], n_perm=2000)[0]
                    vals.append((a + b) / 2)
                per_amp[amp] = {"mean": float(np.mean(vals)),
                                "sd": float(np.std(vals, ddof=1)),
                                "pct": float(np.mean(vals) / ceil[0])}
            best = max(per_amp.values(), key=lambda x: x["mean"])
            res[f"{pname} | {rname}"] = {"ceiling": ceil[0], "ceiling_p": ceil[1],
                                         "per_amp": per_amp, "best": best}

    md.append("| method pool | references | ceiling | best validity | % of ceiling |")
    md.append("|---|---|---|---|---|")
    for key, r in res.items():
        p, rr = key.split(" | ")
        md.append(f"| {p} | {rr} | {r['ceiling']:+.3f} (p {r['ceiling_p']:.4f}) | "
                  f"{r['best']['mean']:+.3f} (sd {r['best']['sd']:.3f}) | "
                  f"{r['best']['pct']:.0%} |")
    md.append("")
    md.append("Per texture amplitude, mean over three seeds:\n")
    md.append("| method pool | references | amp 11 | amp 28 | amp 40 |")
    md.append("|---|---|---|---|---|")
    for key, r in res.items():
        p, rr = key.split(" | ")
        cells = " | ".join(f"{r['per_amp'][a]['mean']:+.3f} ({r['per_amp'][a]['pct']:.0%})"
                           for a in AMPS)
        md.append(f"| {p} | {rr} | {cells} |")
    md.append("")

    json.dump(res, open(os.path.join(OUT, "pool_decomposition.json"), "w"),
              indent=1)
    open(os.path.join(OUT, "POOL_DECOMPOSITION.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

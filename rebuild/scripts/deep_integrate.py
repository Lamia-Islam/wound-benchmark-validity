#!/usr/bin/env python3
"""Sweep the deep decision thresholds and rescore every clinical reference at n=32.

Thresholds are swept on FUSeg training images 400 to 550, which none of the deep
models were trained on and which is disjoint from the FUSeg validation slice used
for evaluation everywhere else.

Then all five clinical references are rescored with the 32-method pool and cached
in results/clinical_ranking_cache32.json. The original 22-method cache is left
untouched so the two can be compared.

Resumable: thresholds and each reference are written as they complete.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import deep as D                                  # noqa: E402
from woundbench.combined import ALL_METHOD_NAMES, CombinedRegistry  # noqa: E402
from woundbench.metrics import dice                               # noqa: E402
from woundbench.postproc import postprocess_prob                  # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark  # noqa: E402
from kvasir_clinical import SETS as KV_SETS, load_kvasir          # noqa: E402

CKPT = os.path.join(OUT, "deep")
THR_PATH = os.path.join(OUT, "deep_thresholds.json")
CACHE32 = os.path.join(OUT, "clinical_ranking_cache32.json")
GRID = np.arange(0.05, 0.91, 0.05)


def build_registry():
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = DeepZooLoad()
    thr = json.load(open(THR_PATH)) if os.path.exists(THR_PATH) else {}
    return CombinedRegistry(classical, zoo, thr), zoo, thr


def DeepZooLoad():
    return D.DeepZoo(CKPT).load()


def sweep_thresholds(zoo):
    """One threshold per deep model, on held-out FUSeg training images."""
    thr = json.load(open(THR_PATH)) if os.path.exists(THR_PATH) else {}
    todo = [n for n in zoo.ready if n not in thr]
    if not todo:
        return thr
    fu = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train")
    X, Y = load_pairs(os.path.join(fu, "images"), os.path.join(fu, "labels"),
                      400, 550)
    print(f"threshold sweep on {len(X)} held-out FUSeg training images",
          flush=True)
    for n in todo:
        probs = [zoo.probability(n, im) for im in X]
        best, best_d = 0.5, -1.0
        for t in GRID:
            d = float(np.mean([dice(postprocess_prob(p, t), y)
                               for p, y in zip(probs, Y)]))
            if d > best_d:
                best, best_d = float(t), d
        thr[n] = best
        json.dump(thr, open(THR_PATH, "w"), indent=2)
        print(f"  {n:<18} threshold {best:.2f}  Dice {best_d:.3f}", flush=True)
    return thr


def clinical_sets():
    med = (os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "images"),
           os.path.join(DATA, "Medetec_foot_ulcer_224", "train", "labels"),
           0, None)
    fus = (os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                        "images"),
           os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                        "labels"), 0, 120)
    azh = (os.path.join(DATA, "wound_dataset", "test", "images"),
           os.path.join(DATA, "wound_dataset", "test", "labels"), 0, 150)
    # disjoint slice of the same dataset: the within-dataset control that makes
    # every other agreement number interpretable
    azh_tr = (os.path.join(DATA, "wound_dataset", "train", "images"),
              os.path.join(DATA, "wound_dataset", "train", "labels"), 0, 150)
    return {"medetec": med, "fuseg": fus, "azh": azh,
            "azh_train_slice": azh_tr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    zoo = DeepZooLoad()
    print(f"{len(zoo.ready)}/10 deep models loaded: {', '.join(zoo.ready)}",
          flush=True)
    if len(zoo.ready) < 10:
        print("not all architectures are trained; run fit_deep.py first")

    thr = sweep_thresholds(zoo)
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    reg = CombinedRegistry(classical, zoo, thr)
    names = reg.available()
    print(f"\npool: {len(names)} methods\n", flush=True)

    cache = json.load(open(CACHE32)) if os.path.exists(CACHE32) else {}
    want = args.only.split(",") if args.only else None

    for key, (idir, ldir, a, b) in clinical_sets().items():
        if key in cache or (want and key not in want):
            continue
        X, Y = load_pairs(idir, ldir, a, b)
        print(f"{key}: {len(X)} pairs", flush=True)
        summ, _ = score_benchmark(key, X, Y, reg, names)
        cache[key] = [summ[n]["mean_dice"] for n in names]
        cache[key + "__lesion_fraction"] = float(np.mean([m.mean() for m in Y]))
        json.dump(cache, open(CACHE32, "w"), indent=2)
        print(f"  cached {key}\n", flush=True)

    for key, base in KV_SETS.items():
        if key in cache or (want and key not in want):
            continue
        X, Y = load_kvasir(base)
        print(f"{key}: {len(X)} pairs", flush=True)
        summ, _ = score_benchmark(key, X, Y, reg, names)
        cache[key] = [summ[n]["mean_dice"] for n in names]
        cache[key + "__lesion_fraction"] = float(np.median([m.mean()
                                                            for m in Y]))
        json.dump(cache, open(CACHE32, "w"), indent=2)
        print(f"  cached {key}\n", flush=True)

    cache["__method_names"] = names
    json.dump(cache, open(CACHE32, "w"), indent=2)
    done = [k for k in cache if isinstance(cache[k], list) and k != "__method_names"]
    print(f"cached references: {sorted(done)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-method mean scores for every method group on every set.

Groups: the 32 local methods (18 untrained U, 14 FUSeg-trained F), and from the
Kaggle run: 10 ImageNet-pretrained architectures on FUSeg (P), nnU-Net on FUSeg
(N), SAM and MedSAM with ground-truth box prompts (S), and the 14 trained
methods retrained on Kvasir (K).

load() returns {metric: {set: {method: mean}}} for metric in dice, nsd,
dice_noempty, nsd_noempty, covering the 12 references and, for Dice, the 21
synthetic sets.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402

RES = os.path.join(ROOT, "results")
REFS = ["medetec", "co2wounds", "fuseg", "azh", "azh_train_slice",
        "kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb",
        "etis_larib", "ph2", "tn3k_test"]


def _trained(n):
    f = ALL_METHOD_FAMILY[n]
    return f in ("learned", "CNN") or f.startswith("deep")


def groups():
    names = json.load(open(os.path.join(RES, "perimage32", "medetec.json")))["names"]
    g = {"U": [n for n in names if not _trained(n)],
         "F": [n for n in names if _trained(n)]}
    for key, grp in (("P", "pretrained"), ("S", "sam"), ("K", "kvasir"),
                     ("N", "nnunet")):
        f = os.path.join(RES, "kaggle", "scores", grp, "medetec.json")
        if os.path.exists(f):
            g[key] = json.load(open(f))["names"]
    return g


EXCL = json.load(open(os.path.join(RES, "exclusions_v5.json")))
CLEAN = os.environ.get("WB_CLEAN", "1") == "1"


def _means(dice, nsd, empty, ref=None):
    D, N = np.asarray(dice, float), np.asarray(nsd, float)
    empty = np.asarray(empty, bool)
    if CLEAN and ref in EXCL:
        drop = {e[0] for e in EXCL[ref]}
        k = np.array([i not in drop for i in range(len(D))])
        D, N, empty = D[k], N[k], empty[k]
    keep = ~empty
    return {"dice": D.mean(0), "nsd": N.mean(0),
            "dice_noempty": D[keep].mean(0), "nsd_noempty": N[keep].mean(0)}


def load():
    out = {m: {} for m in ("dice", "nsd", "dice_noempty", "nsd_noempty")}

    def put(s, names, vals):
        for m, v in vals.items():
            out[m].setdefault(s, {}).update(dict(zip(names, map(float, v))))

    for r in REFS:
        d = json.load(open(os.path.join(RES, "perimage32", r + ".json")))
        put(r, d["names"], _means(d["dice"], d["nsd"], d["gt_empty"], r))
    for f in glob.glob(os.path.join(RES, "kaggle", "scores", "*", "*.json")):
        d = json.load(open(f))
        put(d["set"], d["names"], _means(d["dice"], d["nsd"], d["gt_empty"], d["set"]))
    # the 32 local methods on synthetic sets (mean Dice only)
    for line in open(os.path.join(RES, "pool_decomposition_cells.jsonl")):
        if line.strip():
            c = json.loads(line)
            s = f"genB_amp{int(c['amp'])}_s{c['seed']}"
            out["dice"].setdefault(s, {}).update(dict(zip(c["names"], c["dice"])))
    nr = {"generator A, uniform drape": "nrA", "generator B, texture ablated": "nrB0",
          "generator C, no noise": "nrC0", "generator C, noise 3.4": "nrC34"}
    for line in open(os.path.join(RES, "nonreduce32_cells.jsonl")):
        c = json.loads(line)
        out["dice"].setdefault(nr[c["name"]] + "_s2024", {}).update(c["dice"])
    p = os.path.join(RES, "nonreduce32_seeds.jsonl")
    if os.path.exists(p):
        for line in open(p):
            c = json.loads(line)
            out["dice"].setdefault(c["set"], {}).update(c["dice"])
    return out


def vec(table, s, names):
    return np.array([table[s][n] for n in names])

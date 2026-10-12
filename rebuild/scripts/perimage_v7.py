#!/usr/bin/env python3
"""Shared per-image access for the V7 analysis (run 1 + run 2 merged).

Extends perimage_v6 (same references, exclusions, resampling units) with the
run-2 models:
  pretrained seeds 1, 2   results/kaggle2/scores/pretrained_s{1,2}   (10 original
                          architectures; manet_r18_in_s2 diverged, not saved)
  modern2 seeds 0, 1, 2   results/kaggle2/scores/modern2_s{0,1,2}   (8 new
                          architectures; manet_r50_s0 and _s2 diverged)
  sam_v2_oracle           MedSAM, official preprocessing, reference box
  sam_v2_nnbox            SAM and MedSAM (official) with nnU-Net boxes

Columns
  every individual model under its own name (run-1 names unchanged; seed-0
  pretrained models keep the run-1 name, e.g. unet_r18_in), plus one
  seed-averaged column per architecture:
      <base>_in@seeds   mean over available seeds of the original ten
      <base>@seeds      mean over available seeds of the eight new ones
  Seed averaging is done per image (mean of per-image scores over the seeds
  that exist), so method means equal the mean of the seed means.

Groups (GROUPS)
  U, F, K, S          as in perimage_v6
  M                   V6 definition: 10 pretrained seed 0 + nnunet_2d (11)
  P10s                10 original architectures, seed-averaged
  N8s                 8 new architectures, seed-averaged
  M11s                P10s + nnunet_2d (seed-averaged version of M)
  M19                 P10s + N8s + nnunet_2d (main expanded modern set)
  P10_s0, P10_s1, P10_s2, N8_s0, N8_s1, N8_s2   single-seed groups (only
                      models that exist; P10_s2 has 9, N8_s0 and N8_s2 have 7)
  S2o, S2n            sam_v2_oracle, sam_v2_nnbox
Clinical loads apply perimage_v6 exclusions and check that every file is
aligned with perimage32 (same n, same gt_empty, same gt_fraction).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perimage_v6 as V6                                           # noqa: E402
from perimage_v6 import (CONTROL, COLON, EXCL, MODALITY, REFS, RES,  # noqa: E402,F401
                         SOURCE, WOUND, unit_of)

K1 = os.path.join(RES, "kaggle", "scores")
K2 = os.path.join(RES, "kaggle2", "scores")
SYNTH = [f"genB_amp{a}_s{s}" for a in (11, 28, 40) for s in (2024, 7, 101)] + \
        [f"{n}_s{s}" for n in ("nrA", "nrB0", "nrC0", "nrC34") for s in (2024, 7, 101)]


def _j(path):
    return json.load(open(path))


# Source files per score folder; seed-0 pretrained and nnU-Net come from run 1.
SRC = {
    "pretrained_s0": (K1, "pretrained"), "pretrained_s1": (K2, "pretrained_s1"),
    "pretrained_s2": (K2, "pretrained_s2"), "modern2_s0": (K2, "modern2_s0"),
    "modern2_s1": (K2, "modern2_s1"), "modern2_s2": (K2, "modern2_s2"),
    "nnunet": (K1, "nnunet"), "kvasir": (K1, "kvasir"), "sam": (K1, "sam"),
    "sam_v2_oracle": (K2, "sam_v2_oracle"), "sam_v2_nnbox": (K2, "sam_v2_nnbox"),
}

ORIG10 = [n[:-3] for n in _j(os.path.join(K1, "pretrained", "medetec.json"))["names"]]
NEW8 = [n[:-3] for n in _j(os.path.join(K2, "modern2_s1", "medetec.json"))["names"]]
assert len(ORIG10) == 10 and len(NEW8) == 8, (ORIG10, NEW8)


def _seed_name(base, seed, new):
    if new:
        return f"{base}_s{seed}"
    return f"{base}_in" if seed == 0 else f"{base}_in_s{seed}"


def _avail(folder):
    return _j(os.path.join(*SRC[folder], "medetec.json"))["names"]


SEEDCOLS = {}            # seed-mean column -> list of individual columns present
for b in ORIG10:
    SEEDCOLS[f"{b}_in@seeds"] = [_seed_name(b, s, False) for s in (0, 1, 2)
                                 if _seed_name(b, s, False) in _avail(f"pretrained_s{s}")]
for b in NEW8:
    SEEDCOLS[f"{b}@seeds"] = [_seed_name(b, s, True) for s in (0, 1, 2)
                              if _seed_name(b, s, True) in _avail(f"modern2_s{s}")]

G6 = V6.GROUPS
GROUPS = {k: list(G6[k]) for k in ("U", "F", "K", "M", "S")}
GROUPS["P10s"] = [f"{b}_in@seeds" for b in ORIG10]
GROUPS["N8s"] = [f"{b}@seeds" for b in NEW8]
GROUPS["M11s"] = GROUPS["P10s"] + ["nnunet_2d"]
GROUPS["M19"] = GROUPS["P10s"] + GROUPS["N8s"] + ["nnunet_2d"]
for s in (0, 1, 2):
    GROUPS[f"P10_s{s}"] = list(_avail(f"pretrained_s{s}"))
    GROUPS[f"N8_s{s}"] = list(_avail(f"modern2_s{s}"))
GROUPS["S2o"] = list(_avail("sam_v2_oracle"))
GROUPS["S2n"] = list(_avail("sam_v2_nnbox"))
assert GROUPS["M"] == GROUPS["P10_s0"] + ["nnunet_2d"], GROUPS["M"]

RAW_FOLDERS = list(SRC)
INDIV = []
for f in RAW_FOLDERS:
    INDIV += _avail(f)
LOCAL = V6._local_names()
NAMES_ALL = LOCAL + INDIV + list(SEEDCOLS)
assert len(NAMES_ALL) == len(set(NAMES_ALL)), "duplicate column names"
assert all(n in NAMES_ALL for g in GROUPS.values() for n in g)


def _folder(f, s):
    j = _j(os.path.join(*SRC[f], s + ".json"))
    assert j["set"] == s, (f, s, j["set"])
    assert j["n"] == len(j["dice"]) == len(j["nsd"]), (f, s)
    return j


def _add_seedmeans(cols):
    for c, members in SEEDCOLS.items():
        cols[c] = np.mean(np.stack([cols[m] for m in members], 1), 1)


def load(ref, metric="dice"):
    """Clinical reference or control. Returns the perimage_v6 dict layout with
    X over NAMES_ALL, after exclusions."""
    assert ref in REFS or ref == CONTROL, ref
    d = _j(os.path.join(RES, "perimage32", ref + ".json"))
    n = len(d["ids"])
    cols = {}
    A = np.asarray(d[metric], float)
    for k, nm in enumerate(d["names"]):
        cols[nm] = A[:, k]
    for f in RAW_FOLDERS:
        j = _folder(f, ref)
        assert j["n"] == n, (f, ref, j["n"], n)
        assert j["gt_empty"] == d["gt_empty"], (f, ref, "gt_empty differs")
        assert np.allclose(j["gt_fraction"], d["gt_fraction"], atol=1e-9), (f, ref)
        B = np.asarray(j[metric], float)
        for k, nm in enumerate(j["names"]):
            cols[nm] = B[:, k]
    _add_seedmeans(cols)
    X = np.stack([cols[nm] for nm in NAMES_ALL], 1)
    assert np.isfinite(X).all(), (ref, metric, "non-finite score")
    keep = np.ones(n, bool)
    for e in EXCL.get(ref, []):
        keep[e[0]] = False
    ids = [i for i, k in zip(d["ids"], keep) if k]
    return {"X": X[keep], "names": list(NAMES_ALL),
            "empty": np.asarray(d["gt_empty"], bool)[keep],
            "frac": np.asarray(d["gt_fraction"], float)[keep],
            "units": np.array([unit_of(ref, i) for i in ids]), "ids": ids}


SYNTH_FOLDERS = ["pretrained_s0", "pretrained_s1", "pretrained_s2", "modern2_s0",
                 "modern2_s1", "modern2_s2", "nnunet", "kvasir"]
SYNTH_NAMES = []
for f in SYNTH_FOLDERS:
    SYNTH_NAMES += _avail(f)
SYNTH_NAMES += list(SEEDCOLS)


def load_synth(s, metric="dice"):
    """Synthetic set s: X over SYNTH_NAMES (trained Kaggle models only; the
    untrained and from-scratch methods have no per-image synthetic scores)."""
    assert s in SYNTH, s
    cols, n = {}, None
    for f in SYNTH_FOLDERS:
        j = _folder(f, s)
        n = j["n"] if n is None else n
        assert j["n"] == n, (f, s)
        B = np.asarray(j[metric], float)
        for k, nm in enumerate(j["names"]):
            cols[nm] = B[:, k]
    _add_seedmeans(cols)
    X = np.stack([cols[nm] for nm in SYNTH_NAMES], 1)
    assert np.isfinite(X).all(), (s, metric)
    return {"X": X, "names": list(SYNTH_NAMES)}


def cols(names_subset, names=None):
    names = NAMES_ALL if names is None else names
    return np.array([names.index(n) for n in names_subset])


if __name__ == "__main__":
    print({k: len(v) for k, v in GROUPS.items()})
    print({k: len(v) for k, v in SEEDCOLS.items() if len(v) < 3})
    for r in REFS + [CONTROL]:
        D = load(r)
        Dn = load(r, "nsd")
        print(f"{r:<16} images {len(D['ids']):>3} units {len(set(D['units'])):>3} "
              f"cols {D['X'].shape[1]}  M19 mean Dice {D['X'][:, cols(GROUPS['M19'])].mean():.3f}")
    for s in SYNTH:
        load_synth(s); load_synth(s, "nsd")
    print("synthetic sets ok:", len(SYNTH))
    # V6 compatibility: identical columns to perimage_v6 for its groups
    for r in REFS + [CONTROL]:
        a, b = V6.load(r), load(r)
        assert np.array_equal(a["X"], b["X"][:, cols(V6.NAMES_ALL)]), r
    print("V6 columns reproduced exactly")

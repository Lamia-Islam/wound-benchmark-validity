#!/usr/bin/env python3
"""Shared per-image access for the V6 reanalysis. Every V6 script uses this so
that exclusions, method groups and resampling units are identical everywhere.

load(ref, metric="dice") -> dict with
    X        float array, images x methods (duplicated images already removed)
    names    method names, columns of X, in the order of NAMES_ALL
    empty    bool per image, reference mask empty
    frac     reference lesion fraction per image
    units    resampling unit per image (photograph for Medetec and AZH,
             image otherwise); see unit_of()
    ids      image ids

GROUPS: U 18 untrained, F 14 from scratch on FUSeg, K 14 on Kvasir-SEG,
M 11 modern (10 ImageNet-pretrained architectures + nnU-Net), S 2 box-prompted
SAM/MedSAM (oracle, never pooled).
REFS: the 11 non-control references; CONTROL = azh_train_slice.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402

RES = os.path.join(ROOT, "results")
REFS = ["medetec", "co2wounds", "fuseg", "azh", "kvasir_disjoint",
        "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib", "ph2",
        "tn3k_test"]
CONTROL = "azh_train_slice"
MODALITY = {"medetec": "wound", "co2wounds": "wound", "fuseg": "wound",
            "azh": "wound", "kvasir_disjoint": "colonoscopy",
            "kvasir_sessile": "colonoscopy", "cvc_clinicdb": "colonoscopy",
            "cvc_colondb": "colonoscopy", "etis_larib": "colonoscopy",
            "ph2": "dermoscopy", "tn3k_test": "ultrasound"}
SOURCE = {"medetec": "Medetec", "co2wounds": "CO2Wounds", "fuseg": "AZH",
          "azh": "AZH", "kvasir_disjoint": "Kvasir", "kvasir_sessile": "Kvasir",
          "cvc_clinicdb": "CVC", "cvc_colondb": "CVC", "etis_larib": "ETIS",
          "ph2": "PH2", "tn3k_test": "TN3K"}
WOUND = [r for r in REFS if MODALITY[r] == "wound"]
COLON = [r for r in REFS if MODALITY[r] == "colonoscopy"]
EXCL = json.load(open(os.path.join(RES, "exclusions_v5.json")))
KAGGLE = {"M": ["pretrained", "nnunet"], "K": ["kvasir"], "S": ["sam"]}


def _trained(n):
    f = ALL_METHOD_FAMILY[n]
    return f in ("learned", "CNN") or f.startswith("deep")


def _local_names():
    return json.load(open(os.path.join(RES, "perimage32", "medetec.json")))["names"]


def groups():
    ln = _local_names()
    g = {"U": [n for n in ln if not _trained(n)], "F": [n for n in ln if _trained(n)]}
    for key, dirs in KAGGLE.items():
        g[key] = []
        for d in dirs:
            g[key] += json.load(open(os.path.join(RES, "kaggle", "scores", d, "medetec.json")))["names"]
    return g


GROUPS = groups()
NAMES_ALL = sum((GROUPS[k] for k in ("U", "F", "K", "M", "S")), [])


def unit_of(ref, img_id):
    """Resampling unit. Medetec stores four variants of each photograph
    (suffix _1.._3); AZH and its train slice store several patches of one
    photograph (hash_0, hash_1, ...). Other datasets: one unit per image."""
    if ref == "medetec":
        return re.sub(r"_\d+$", "", img_id)
    if ref in ("azh", CONTROL):
        return re.sub(r"_\d+$", "", img_id)
    return img_id


def load(ref, metric="dice"):
    d = json.load(open(os.path.join(RES, "perimage32", ref + ".json")))
    cols = [np.asarray(d[metric], float)]
    names = list(d["names"])
    for key in ("K", "M", "S"):
        for sub in KAGGLE[key]:
            g = json.load(open(os.path.join(RES, "kaggle", "scores", sub, ref + ".json")))
            assert g["n"] == len(d["ids"]), (ref, sub)
            cols.append(np.asarray(g[metric], float))
            names += g["names"]
    X = np.concatenate(cols, axis=1)
    order = [names.index(n) for n in NAMES_ALL]
    X = X[:, order]
    keep = np.ones(len(d["ids"]), bool)
    for e in EXCL.get(ref, []):
        keep[e[0]] = False
    ids = [i for i, k in zip(d["ids"], keep) if k]
    return {"X": X[keep], "names": list(NAMES_ALL),
            "empty": np.asarray(d["gt_empty"], bool)[keep],
            "frac": np.asarray(d["gt_fraction"], float)[keep],
            "units": np.array([unit_of(ref, i) for i in ids]), "ids": ids}


def cols(names_subset):
    return np.array([NAMES_ALL.index(n) for n in names_subset])


if __name__ == "__main__":
    print({k: len(v) for k, v in GROUPS.items()})
    for r in REFS + [CONTROL]:
        D = load(r)
        print(f"{r:<16} images {len(D['ids']):>3} units {len(set(D['units'])):>3} "
              f"empty {int(D['empty'].sum())} methods {D['X'].shape[1]}")

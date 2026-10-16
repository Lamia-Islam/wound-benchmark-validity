#!/usr/bin/env python3
"""Rebuild the parts of the Kaggle kit that the audit reads, from the public raw
data, with the kit's own export code (scripts/kaggle_export.py):
  refs/<set>/NNNN_img.png, NNNN_mask.png   for the wound references and control
  train/fuseg/NNNN_img.png ...             FUSeg training images 0 to 549
  manifest.json                            fuseg_train (files, eval_duplicates)
                                           from results/v7/fuseg_train_index.json
CO2Wounds-V2 is not in the public wound-segmentation repository and is skipped.

Usage: python scripts/v7/kit_partial_v7.py OUT_DIR   (then WB_KIT=OUT_DIR)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import kaggle_export as KE                                        # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
man = {}
KE.REFS = ["medetec", "fuseg", "azh", "azh_train_slice"]
KE.export_refs(out, man)
imgs, masks = KE.P.wound_pairs(os.path.join(KE.FUSEG_TRAIN, "images"), os.path.join(KE.FUSEG_TRAIN, "labels"), 0, 550)
idx = json.load(open(os.path.join(ROOT, "results", "v7", "fuseg_train_index.json")))
assert [os.path.basename(p) for p in imgs] == idx["files"]
od = os.path.join(out, "train", "fuseg")
os.makedirs(od, exist_ok=True)
for k, (ip, mp) in enumerate(zip(imgs, masks)):
    im, m = KE.P.load_wound(ip, mp)
    KE.write_pair(od, k, im, m)
man["fuseg_train"] = {"n": len(imgs), "eval_duplicates": idx["eval_duplicates"], "files": idx["files"]}
man["_note"] = "partial kit rebuilt by kit_partial_v7.py; synth and kvasir parts absent"
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("done", {k: v["n"] for k, v in man["refs"].items()})

#!/usr/bin/env python3
"""Build the Kaggle kit: code plus exactly the images the GPU run needs.

Everything is written as lossless PNG in the form the local scripts see it,
so a method scored on Kaggle sees the same pixels as the 32 methods scored
here:

  refs/<reference>/NNNN_img.png, NNNN_mask.png
      the 12 clinical references, loaded with the loaders perimage32.py used
      (same images, same order, same resampling and mask binarisation)
  synth/<set>/NNNN_img.png, NNNN_mask.png
      generator B cells (amplitudes 11, 28, 40 x seeds 2024, 7, 101) and the
      four matched-texture benchmarks of Table 8 at seeds 2024, 7, 101
  train/fuseg/NNNN_img.png ...   FUSeg training images 0 to 549, native size
  train/kvasir/NNNN_img.png ...  Kvasir-SEG images outside both Kvasir
                                 references, long side resampled to 256
  manifest.json   ids, texture of every synthetic set, FUSeg training images
                  that are byte-identical to an evaluation image (excluded)

Usage: python scripts/kaggle_export.py OUT_DIR
"""

import glob
import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import perimage32 as P                                            # noqa: E402
from woundbench.generator_a import build as build_a               # noqa: E402
from woundbench.generator_b import GenBConfig, GeneratorB         # noqa: E402
from woundbench.generator_c import build as build_c               # noqa: E402
from woundbench.imagestats import texture_of_set                  # noqa: E402

PER = os.path.join(ROOT, "results", "perimage32")
REFS = ["medetec", "co2wounds", "fuseg", "azh", "azh_train_slice",
        "kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb",
        "etis_larib", "ph2", "tn3k_test"]
SEEDS = [2024, 7, 101]
AMPS = [11.0, 28.0, 40.0]
FUSEG_TRAIN = os.path.join(ROOT, "data", "wound-segmentation", "data",
                           "Foot Ulcer Segmentation Challenge", "train")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def write_pair(d, k, bgr, mask01):
    cv2.imwrite(os.path.join(d, f"{k:04d}_img.png"), bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(os.path.join(d, f"{k:04d}_mask.png"),
                (mask01 > 0).astype(np.uint8) * 255,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def export_refs(out, man):
    loaders = dict(P.LOADERS)
    loaders["extra"] = P.make_extra_loader(127)
    man["refs"] = {}
    for name in REFS:
        d = json.load(open(os.path.join(PER, name + ".json")))
        load = loaders[d["loader"]]
        od = os.path.join(out, "refs", name)
        os.makedirs(od, exist_ok=True)
        ids, gt_ok = [], 0
        for k, (ip, mp) in enumerate(zip(d["image_paths"], d["mask_paths"])):
            im, m = load(os.path.join(ROOT, ip), os.path.join(ROOT, mp))
            write_pair(od, k, im, m)
            ids.append(d["ids"][k])
            gt_ok += abs(float(m.mean()) - d["gt_fraction"][k]) < 1e-9
        assert gt_ok == len(ids), f"{name}: masks differ from the scored ones"
        man["refs"][name] = {"ids": ids, "n": len(ids)}
        print(f"  {name}: {len(ids)}", flush=True)


def synth_sets():
    for amp in AMPS:
        for s in SEEDS:
            def f(amp=amp, s=s):
                r = GeneratorB(seed=s, config=replace(GenBConfig(),
                               mottle_amplitude=amp)).generate(15)
                return [x[2] for x in r], [x[3] for x in r]
            yield f"genB_amp{int(amp)}_s{s}", f
    for s in SEEDS:
        def fa(s=s):
            i, m, _ = build_a(seed=s, n_per_class=15)
            return i, m

        def fb(s=s):
            r = GeneratorB(seed=s, config=replace(GenBConfig(),
                           mottle_amplitude=0.0)).generate(15)
            return [x[2] for x in r], [x[3] for x in r]

        def fc0(s=s):
            i, m, _ = build_c(seed=s, n_per_class=15, noise_amplitude=0.0)
            return i, m

        def fc3(s=s):
            i, m, _ = build_c(seed=s, n_per_class=15, noise_amplitude=3.4)
            return i, m
        yield f"nrA_s{s}", fa
        yield f"nrB0_s{s}", fb
        yield f"nrC0_s{s}", fc0
        yield f"nrC34_s{s}", fc3


def export_synth(out, man):
    man["synth"] = {}
    for name, f in synth_sets():
        imgs, masks = f()
        od = os.path.join(out, "synth", name)
        os.makedirs(od, exist_ok=True)
        for k, (im, m) in enumerate(zip(imgs, masks)):
            write_pair(od, k, im, m)
        tex = float(texture_of_set(imgs, masks)[0])
        man["synth"][name] = {"n": len(imgs), "texture": tex}
        print(f"  {name}: {len(imgs)} texture {tex:.3f}", flush=True)


def export_fuseg(out, man):
    imgs, masks = P.wound_pairs(os.path.join(FUSEG_TRAIN, "images"),
                                os.path.join(FUSEG_TRAIN, "labels"), 0, 550)
    # byte-identical copies of any evaluation image are left out of training
    ev = set()
    for name in REFS:
        d = json.load(open(os.path.join(PER, name + ".json")))
        ev |= {md5(os.path.join(ROOT, p)) for p in d["image_paths"]}
    od = os.path.join(out, "train", "fuseg")
    os.makedirs(od, exist_ok=True)
    dup = []
    for k, (ip, mp) in enumerate(zip(imgs, masks)):
        if md5(ip) in ev:
            dup.append(k)
        im, m = P.load_wound(ip, mp)
        write_pair(od, k, im, m)
    man["fuseg_train"] = {"n": len(imgs), "eval_duplicates": dup,
                          "files": [os.path.basename(p) for p in imgs]}
    print(f"  fuseg train: {len(imgs)}, {len(dup)} duplicates of evaluation "
          f"images: {dup}", flush=True)


def export_kvasir(out, man):
    full = os.path.join(P.KV, "Kvasir-SEG")
    used = set()
    for name in ("kvasir_disjoint", "kvasir_sessile"):
        d = json.load(open(os.path.join(PER, name + ".json")))
        used |= {P.stem(p) for p in d["image_paths"]}
    sess = {P.stem(p) for p in glob.glob(os.path.join(
        P.KV, "sessile-main-Kvasir-SEG", "images", "*"))}
    pool = [p for p in sorted(glob.glob(os.path.join(full, "images", "*")))
            if P.stem(p) not in used and P.stem(p) not in sess]
    od = os.path.join(out, "train", "kvasir")
    os.makedirs(od, exist_ok=True)
    stems = []
    for k, ip in enumerate(pool):
        mp = P.kvasir_mask_for(full, ip)
        im, m = P.load_kvasir_one(ip, mp)
        write_pair(od, k, im, m)
        stems.append(P.stem(ip))
    man["kvasir_train"] = {"n": len(pool), "files": stems}
    print(f"  kvasir train: {len(pool)} (excluded {len(used)} evaluation and "
          f"{len(sess)} sessile)", flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    man = {}
    print("references", flush=True)
    export_refs(out, man)
    print("synthetic sets", flush=True)
    export_synth(out, man)
    print("training sets", flush=True)
    export_fuseg(out, man)
    export_kvasir(out, man)
    shutil.copytree(os.path.join(ROOT, "woundbench"),
                    os.path.join(out, "woundbench"), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()

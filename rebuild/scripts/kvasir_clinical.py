#!/usr/bin/env python3
"""Add Kvasir-SEG as a clinical reference from a different modality.

Everything in this project so far is chronic wound photography. Kvasir-SEG is
colonoscopy: different instrument, different illumination, different tissue,
different annotators, different pathology. It shares nothing with the wound
datasets except being clinical images with expert lesion masks.

Two things are computed here.

1. The rank agreement between Kvasir and every other clinical reference. This
   asks whether "which segmentation method is good" is a property of the method
   or of the imaging domain.

2. Kvasir's contrast verdict, placed against the framing curve that was built
   entirely from cropped wound images. Kvasir-SEG's median lesion fraction is
   0.106, almost exactly Medetec's 0.102, so the curve makes a prediction about
   a modality it has never seen.

Mask handling differs from the wound datasets and this matters. Medetec masks
are 0/1 and FUSeg 0/255, so the project binarises on any non-zero value. Kvasir
masks are JPEG-compressed and carry anti-aliased edge values of 1, 2, 3 and so
on, so any-non-zero would inflate every lesion. Kvasir is thresholded at 127.

Writes results/kvasir_clinical.json and extends the ranking cache.
"""

import glob
import itertools
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.methods import METHOD_NAMES                   # noqa: E402
from woundbench.metrics import spearman_permutation           # noqa: E402
from rank_validity import OUT, score_benchmark                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KV = os.path.join(ROOT, "data", "kvasir")
SETS = {
    "kvasir": os.path.join(KV, "Kvasir-SEG"),
    "kvasir_sessile": os.path.join(KV, "sessile-main-Kvasir-SEG"),
}
N = 120
MAX_SIDE = 256          # matches Medetec resolution; keeps scoring tractable


def load_kvasir(base, n=N):
    """Image/mask pairs with a mid-range mask threshold, long side capped."""
    ims = sorted(glob.glob(os.path.join(base, "images", "*")))[:n]
    X, Y = [], []
    for p in ims:
        stem = os.path.splitext(os.path.basename(p))[0]
        cand = glob.glob(os.path.join(base, "masks", stem + ".*"))
        if not cand:
            continue
        im = cv2.imread(p)
        m = cv2.imread(cand[0], cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            continue
        if max(im.shape[:2]) > MAX_SIDE:
            s = MAX_SIDE / max(im.shape[:2])
            wh = (int(round(im.shape[1] * s)), int(round(im.shape[0] * s)))
            im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, wh, interpolation=cv2.INTER_NEAREST)
        X.append(im)
        Y.append((m > 127).astype(np.uint8))
    return X, Y


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    cpath = os.path.join(OUT, "clinical_ranking_cache.json")
    cache = json.load(open(cpath))

    for key, base in SETS.items():
        if key in cache:
            print(f"{key} already scored", flush=True)
            continue
        X, Y = load_kvasir(base)
        frac = float(np.median([m.mean() for m in Y]))
        print(f"\n{key}: {len(X)} pairs, median lesion fraction {frac:.4f}",
              flush=True)
        summ, _ = score_benchmark(key, X, Y, reg, names)
        cache[key] = [summ[n]["mean_dice"] for n in names]
        cache[key + "__lesion_fraction"] = frac
        json.dump(cache, open(cpath, "w"), indent=2)
        print(f"  cached {key}", flush=True)

    refs = [k for k in ("medetec", "fuseg", "azh", "kvasir", "kvasir_sessile")
            if k in cache and isinstance(cache[k], list)]

    print("\n" + "=" * 76)
    print("RANK AGREEMENT BETWEEN CLINICAL REFERENCES, WITHIN AND ACROSS MODALITY")
    print("=" * 76)
    print(f"{'pair':<34}{'rho':>9}{'p':>10}{'':>6}")
    wound = {"medetec", "fuseg", "azh"}
    within_w, within_k, across = [], [], []
    mat = {}
    for a, b in itertools.combinations(refs, 2):
        r, p = spearman_permutation(cache[a], cache[b], n_perm=20000)
        mat[f"{a}__{b}"] = {"rho": r, "p": p}
        aw, bw = a in wound, b in wound
        tag = "wound" if (aw and bw) else ("endoscopy" if (not aw and not bw)
                                           else "ACROSS")
        (within_w if (aw and bw) else within_k if (not aw and not bw)
         else across).append(r)
        print(f"{a + ' vs ' + b:<34}{r:>+9.3f}{p:>10.4f}  {tag}")

    print()
    if within_w:
        print(f"within wound photography   mean {np.mean(within_w):+.3f}  "
              f"({len(within_w)} pairs)")
    if within_k:
        print(f"within endoscopy           mean {np.mean(within_k):+.3f}  "
              f"({len(within_k)} pairs)")
    if across:
        print(f"across modalities          mean {np.mean(across):+.3f}  "
              f"({len(across)} pairs)")

    # ---- place Kvasir on the wound-built framing curve ----------------------
    law_path = os.path.join(OUT, "framing_law.json")
    if os.path.exists(law_path):
        law = json.load(open(law_path))
        curve = sorted(law["curve"], key=lambda r: r["achieved"])
        fx = [r["achieved"] for r in curve]
        fy = [r["trend"] for r in curve]
        dr = json.load(open(os.path.join(OUT, "dose_response.json")))
        sweep = dr["contrast_sweep"]
        xs = [r["measured_sat_delta"] for r in sweep]

        print("\n" + "=" * 76)
        print("ENDOSCOPY PLACED ON A CURVE BUILT ENTIRELY FROM WOUND IMAGES")
        print("=" * 76)
        print(f"{'dataset':<18}{'lesion frac':>13}{'predicted':>12}"
              f"{'measured':>11}{'residual':>11}")
        placed = {}
        for k in ("kvasir", "kvasir_sessile"):
            if k not in cache:
                continue
            f0 = cache[k + "__lesion_fraction"]
            agree = [spearman_permutation([r["per_method_dice"][n]
                                           for n in names], cache[k],
                                          n_perm=4000)[0] for r in sweep]
            meas = spearman_permutation(xs, agree, n_perm=20000)[0]
            pred = float(np.interp(f0, fx, fy))
            placed[k] = {"lesion_fraction": f0, "predicted": pred,
                         "measured": meas, "residual": meas - pred}
            print(f"{k:<18}{f0:>13.4f}{pred:>+12.3f}{meas:>+11.3f}"
                  f"{meas - pred:>+11.3f}")
        print("\nThe curve was fitted on cropped foot ulcer photographs and has")
        print("never seen an endoscope. A small residual would mean the framing")
        print("effect is a property of benchmark references, not of wounds.")
    else:
        placed = {}

    json.dump({"matrix": mat, "refs": refs, "placed_on_wound_curve": placed,
               "within_wound": within_w, "within_endoscopy": within_k,
               "across_modality": across},
              open(os.path.join(OUT, "kvasir_clinical.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'kvasir_clinical.json')}")


if __name__ == "__main__":
    main()

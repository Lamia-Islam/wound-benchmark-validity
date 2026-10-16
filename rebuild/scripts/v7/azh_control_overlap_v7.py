#!/usr/bin/env python3
"""Do the AZH test images and the control images (AZH train slice) show the
same photograph?

The control is meant to be a second, disjoint sample of the AZH source. Two
checks:
  pixel   decoded pixels identical (md5 of the decoded array)
  crop    the SIFT rule of azh_overlap_v7.py (ratio 0.75, RANSAC similarity
          3 px, at least 25 inliers), every test image against every control
          image, brute-force matcher (deterministic)

Output: results/v7/azh_control_overlap_v7.json
"""
import hashlib
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
RAW = os.path.join(ROOT, "data", "wound-segmentation", "data", "wound_dataset")
MIN_INLIERS = 25
cv2.setNumThreads(1)
CTRL = None


def feats(img):
    k, d = cv2.SIFT_create().detectAndCompute(img, None)
    return (np.float32([p.pt for p in k]), d) if d is not None else (np.zeros((0, 2), np.float32), None)


def best(args):
    name, img = args
    pa, da = feats(img)
    bf = cv2.BFMatcher()
    out = []
    if da is None or len(da) < 8:
        return name, out
    for cname, pc, dc in CTRL:
        if dc is None or len(dc) < 8:
            continue
        good = [m for m, n in (x for x in bf.knnMatch(da, dc, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            continue
        T, inl = cv2.estimateAffinePartial2D(pa[[m.queryIdx for m in good]], pc[[m.trainIdx for m in good]],
                                             method=cv2.RANSAC, ransacReprojThreshold=3.0)
        n = int(inl.sum()) if inl is not None else 0
        if n >= MIN_INLIERS:
            out.append({"control": cname, "inliers": n})
    return name, out


def main():
    global CTRL
    t0 = time.time()
    ids_t = json.load(open(os.path.join(RES, "perimage32", "azh.json")))["ids"]
    ids_c = json.load(open(os.path.join(RES, "perimage32", "azh_train_slice.json")))["ids"]
    imt = {i: cv2.imread(os.path.join(RAW, "test", "images", i + ".png")) for i in ids_t}
    imc = {i: cv2.imread(os.path.join(RAW, "train", "images", i + ".png")) for i in ids_c}
    h = {i: hashlib.md5(im.tobytes() + str(im.shape).encode()).hexdigest() for i, im in imc.items()}
    inv = {}
    for i, x in h.items():
        inv.setdefault(x, []).append(i)
    pixel = {i: inv[x] for i, im in imt.items()
             if (x := hashlib.md5(im.tobytes() + str(im.shape).encode()).hexdigest()) in inv}
    CTRL = [(i,) + feats(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for i, im in imc.items()]
    with Pool(int(os.environ.get("WB_PROCS", "2"))) as p:
        crop = {k: v for k, v in p.map(best, [(i, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)) for i, im in imt.items()],
                                       chunksize=5) if v}
    out = {"rule": f"pixel identity; SIFT ratio 0.75, RANSAC similarity 3 px, >= {MIN_INLIERS} inliers, brute-force matcher",
           "n_test": len(ids_t), "n_control": len(ids_c),
           "pixel_identical": pixel, "crop_matches": crop,
           "n_test_pixel_identical": len(pixel),
           "n_test_with_crop_match": len(crop),
           "control_images_matched": sorted({c["control"] for v in crop.values() for c in v} |
                                            {c for v in pixel.values() for c in v}),
           "runtime_s": round(time.time() - t0)}
    out["n_control_images_matched"] = len(out["control_images_matched"])
    json.dump(out, open(os.path.join(RES, "v7", "azh_control_overlap_v7.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("crop_matches",)})


if __name__ == "__main__":
    main()

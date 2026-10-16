#!/usr/bin/env python3
"""Are AZH test images crops of FUSeg photographs?

Same rule as the control-set exclusions (exclusions_v5.json): SIFT features,
Lowe ratio 0.75, similarity transform by RANSAC (reprojection 3 px), a match is
at least 25 geometrically consistent inliers.

Validation of the procedure: the 150 control images (AZH train slice) are
checked the same way; the 47 recorded control exclusions must be found again.

Output: results/v7/azh_overlap_v7.json
"""
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
RAW = os.path.join(ROOT, "data", "wound-segmentation", "data")
FUS = os.path.join(RAW, "Foot Ulcer Segmentation Challenge")
MIN_INLIERS = 25
cv2.setNumThreads(1)


def gray(buf_or_path):
    if isinstance(buf_or_path, bytes):
        return cv2.imdecode(np.frombuffer(buf_or_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.imread(buf_or_path, cv2.IMREAD_GRAYSCALE)


def feats(img):
    k, d = cv2.SIFT_create().detectAndCompute(img, None)
    if d is None:
        return np.zeros((0, 2), np.float32), None
    return np.float32([p.pt for p in k]), d


FUS_FEATS = None


def load_fuseg():
    out = []
    for split in ("train", "validation"):
        d = os.path.join(FUS, split, "images")
        for f in sorted(os.listdir(d)):
            out.append((f"{split}/{f}",) + feats(gray(os.path.join(d, f))))
    return out


def best_match(args):
    name, img = args
    pa, da = feats(img)
    best = (0, None, None)
    if da is None or len(da) < 8:
        return name, best
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    for fname, pf, df in FUS_FEATS:
        if df is None or len(df) < 8:
            continue
        good = [m for m, n in (x for x in flann.knnMatch(da, df, k=2) if len(x) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            continue
        A = pa[[m.queryIdx for m in good]]
        B = pf[[m.trainIdx for m in good]]
        T, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        n = int(inl.sum()) if inl is not None else 0
        if n > best[0]:
            best = (n, fname, float(np.hypot(T[0, 0], T[1, 0])) if T is not None else None)
    return name, best


def main():
    global FUS_FEATS
    t0 = time.time()
    FUS_FEATS = load_fuseg()
    print(f"FUSeg features: {len(FUS_FEATS)} images, {time.time() - t0:.0f}s", flush=True)
    ids_test = json.load(open(os.path.join(RES, "perimage32", "azh.json")))["ids"]
    ids_ctrl = json.load(open(os.path.join(RES, "perimage32", "azh_train_slice.json")))["ids"]
    test = [(i, gray(os.path.join(RAW, "wound_dataset", "test", "images", i + ".png"))) for i in ids_test]
    assert all(im is not None for _, im in test), "missing AZH test image"
    z = zipfile.ZipFile(os.path.join(RAW, "wound_dataset", "azh_wound_care_center_dataset_patches.zip"))
    names = {os.path.basename(n): n for n in z.namelist() if ("/" + n.replace("\\", "/")).find("/train/images/") >= 0 and not n.endswith("/")}
    ctrl = [(i, gray(z.read(names[i + ".png"]))) for i in ids_ctrl]
    with Pool(2) as p:
        r_test = dict(p.map(best_match, test, chunksize=5))
        print(f"test done {time.time() - t0:.0f}s", flush=True)
        r_ctrl = dict(p.map(best_match, ctrl, chunksize=5))
    excl = {e[1].replace(".png", ""): e[2] for e in json.load(open(os.path.join(RES, "exclusions_v5.json")))["azh_train_slice"]}
    ctrl_hit = {k for k, v in r_ctrl.items() if v[0] >= MIN_INLIERS}
    found = sorted(set(excl) & ctrl_hit)
    out = {
        "rule": f"SIFT, ratio 0.75, RANSAC similarity 3 px, >= {MIN_INLIERS} inliers; FUSeg train (810) + validation (200)",
        "validation_control": {"recorded_exclusions": len(excl), "found_again": len(found),
                               "new_hits_not_recorded": sorted(ctrl_hit - set(excl))},
        "azh_test": {"n": len(test),
                     "matches": {k: {"inliers": v[0], "fuseg": v[1], "scale": v[2]}
                                 for k, v in r_test.items() if v[0] >= MIN_INLIERS},
                     "max_inliers_nonmatch": max([v[0] for v in r_test.values() if v[0] < MIN_INLIERS] or [0])},
        "runtime_s": round(time.time() - t0),
    }
    m = out["azh_test"]["matches"]
    out["azh_test"]["n_match"] = len(m)
    out["azh_test"]["n_match_train"] = sum(v["fuseg"].startswith("train/") for v in m.values())
    out["azh_test"]["n_match_validation"] = sum(v["fuseg"].startswith("validation/") for v in m.values())
    json.dump(out, open(os.path.join(RES, "v7", "azh_overlap_v7.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "azh_test"}, indent=1))
    print({k: v for k, v in out["azh_test"].items() if k != "matches"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Are the AZH patches crops of the FUSeg images?

Both come from the AZH Wound and Vascular Center. The repository README says the
wound_dataset was "preprocessed with cropping and zero-padding" and that the same
data was later published as the MICCAI Foot Ulcer Segmentation Challenge. If the
patches are literally crops of the challenge frames, then AZH and FUSeg are the
same clinical images under two different framing conventions, which makes the
pair a natural experiment rather than two independent datasets.

Test: for a sample of AZH patches, template-match against every FUSeg frame at a
range of scales and report the best normalised cross-correlation. A genuine crop
scores near 1.0; unrelated images sit far below.
"""
import glob, os, sys, json
import numpy as np, cv2

ROOT = "/home/claude/wound-recovery/rebuild/data/wound-segmentation/data"
AZH = os.path.join(ROOT, "wound_dataset")
FUS = os.path.join(ROOT, "Foot Ulcer Segmentation Challenge")

def load_gray(p, side=None):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if side: g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA)
    return g

def trim_pad(g, thr=8):
    """Strip uniform black padding so the real content is what gets matched."""
    m = g > thr
    if not m.any(): return g
    ys, xs = np.where(m)
    return g[ys.min():ys.max()+1, xs.min():xs.max()+1]

fus_paths = sorted(glob.glob(os.path.join(FUS, "train", "images", "*.png"))) + \
            sorted(glob.glob(os.path.join(FUS, "validation", "images", "*.png")))
azh_paths = sorted(glob.glob(os.path.join(AZH, "train", "images", "*.png")))
print(f"{len(fus_paths)} FUSeg frames, {len(azh_paths)} AZH patches", flush=True)

WORK = 256
fus = [(os.path.basename(p), trim_pad(load_gray(p, WORK))) for p in fus_paths]
print("FUSeg frames loaded and de-padded", flush=True)

rng = np.random.default_rng(0)
sample = [azh_paths[i] for i in rng.choice(len(azh_paths), 25, replace=False)]

rows = []
for ap in sample:
    a0 = trim_pad(load_gray(ap))
    best = (-1.0, None, None)
    for frac in (1.00, 0.85, 0.70, 0.55, 0.40, 0.30):
        for name, f in fus:
            th = max(8, int(round(f.shape[0] * frac)))
            tw = max(8, int(round(f.shape[1] * frac)))
            if th > f.shape[0] or tw > f.shape[1]: continue
            t = cv2.resize(a0, (tw, th), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
            v = float(r.max())
            if v > best[0]: best = (v, name, frac)
    rows.append({"azh": os.path.basename(ap), "ncc": best[0],
                 "fuseg": best[1], "scale": best[2]})
    print(f"  {os.path.basename(ap)[:16]}  best NCC {best[0]:.3f}  "
          f"vs {best[1]}  at {best[2]:.2f}", flush=True)

v = np.array([r["ncc"] for r in rows])
print("\n" + "=" * 64)
print(f"median best NCC {np.median(v):.3f}   min {v.min():.3f}   max {v.max():.3f}")
print(f"above 0.90: {int((v > 0.90).sum())}/{len(v)}   "
      f"above 0.75: {int((v > 0.75).sum())}/{len(v)}")
print("NCC near 1.0 means the patch is literally a crop of that FUSeg frame.")
json.dump(rows, open("/home/claude/wound-recovery/rebuild/results/azh_overlap.json", "w"), indent=2)

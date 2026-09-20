#!/usr/bin/env python3
"""Fit the four learned methods on FUSeg training images.

Manuscript 3.5: pixel classifiers trained on 264,292 pixels sampled from 250
FUSeg training images, balanced lesion / non-lesion.
Manuscript 3.6: U-Net trained on 393 FUSeg training images at 128 x 128.
Manuscript 5.4: the decision threshold is swept on FUSeg training images
250-400, which are disjoint from every evaluation split.

Writes fitted models to results/models.pkl.
"""

import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench import learned as L          # noqa: E402
from woundbench.methods import MethodRegistry  # noqa: E402
from woundbench.metrics import dice, binarise  # noqa: E402
from woundbench.postproc import postprocess_prob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "wound-segmentation", "data")
FUSEG_TRAIN = os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "train")
OUT = os.path.join(ROOT, "results")


def load_pairs(img_dir, lab_dir, start=0, stop=None):
    imgs = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                  if p.lower().endswith((".png", ".jpg", ".jpeg")))
    labs = sorted(p for p in glob.glob(os.path.join(lab_dir, "*"))
                  if p.lower().endswith((".png", ".jpg", ".jpeg")))
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    lab_by = {stem(p): p for p in labs}
    pairs = [(i, lab_by[stem(i)]) for i in imgs if stem(i) in lab_by]
    pairs = pairs[start:stop]
    X = [cv2.imread(i) for i, _ in pairs]
    Y = [binarise(cv2.imread(l, cv2.IMREAD_GRAYSCALE)) for _, l in pairs]
    keep = [k for k in range(len(X)) if X[k] is not None and Y[k] is not None]
    return [X[k] for k in keep], [Y[k] for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=L.UNET_EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    img_dir = os.path.join(FUSEG_TRAIN, "images")
    lab_dir = os.path.join(FUSEG_TRAIN, "labels")

    print("loading FUSeg training images 0-250 (pixel classifiers)", flush=True)
    px_imgs, px_masks = load_pairs(img_dir, lab_dir, 0, L.N_TRAIN_IMAGES_PIXEL)
    print(f"  {len(px_imgs)} images", flush=True)

    reg = MethodRegistry()
    t0 = time.time()
    reg.fit_pixel_classifiers(px_imgs, px_masks, seed=args.seed)
    print(f"  pixel classifiers fitted in {time.time() - t0:.1f} s", flush=True)

    print(f"loading FUSeg training images 0-{L.N_TRAIN_IMAGES_UNET} (U-Net)",
          flush=True)
    un_imgs, un_masks = load_pairs(img_dir, lab_dir, 0, L.N_TRAIN_IMAGES_UNET)
    print(f"  {len(un_imgs)} images", flush=True)

    t0 = time.time()
    reg.fit_unet(un_imgs, un_masks, seed=args.seed, epochs=args.epochs)
    print(f"  U-Net trained in {time.time() - t0:.0f} s "
          f"(manuscript reports 470 s)", flush=True)

    # --- threshold sweep on FUSeg training images 250-400 (manuscript 5.4) ---
    print("sweeping decision threshold on FUSeg training images 250-400",
          flush=True)
    sw_imgs, sw_masks = load_pairs(img_dir, lab_dir, 250, 400)
    print(f"  {len(sw_imgs)} held-out images", flush=True)

    probs = [reg.probability("logreg_all", im) for im in sw_imgs]
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    best, best_t = -1.0, 0.9
    print(f"  {'thr':>5s}{'meanDice':>10s}")
    for t in grid:
        d = float(np.mean([dice(postprocess_prob(p, float(t)), m)
                           for p, m in zip(probs, sw_masks)]))
        print(f"  {t:>5.2f}{d:>10.3f}", flush=True)
        if d > best:
            best, best_t = d, float(t)
    print(f"  selected threshold {best_t:.2f} (Dice {best:.3f}); "
          f"manuscript reports 0.9", flush=True)

    reg.learned_threshold = best_t
    with open(os.path.join(OUT, "models.pkl"), "wb") as f:
        pickle.dump({"registry": reg,
                     "swept_threshold": best_t,
                     "sweep_best_dice": best,
                     "seed": args.seed}, f)
    print(f"saved {os.path.join(OUT, 'models.pkl')}", flush=True)


if __name__ == "__main__":
    main()

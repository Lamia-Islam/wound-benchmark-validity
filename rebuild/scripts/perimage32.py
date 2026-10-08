#!/usr/bin/env python3
"""Per-image Dice and NSD for all 32 methods on every clinical reference.

Earlier scoring kept only the mean Dice per method. This script keeps the full
per-image matrices, so later analyses (empty ground truth, tiny lesions, ties)
can be computed without rerunning inference.

The two Kvasir references are made disjoint here. The 196 sessile images are all
inside the 1000 Kvasir-SEG images, and the old "first 120 sorted files of each"
selection shared 31 images. kvasir_disjoint takes the first 120 Kvasir-SEG
images whose stem is not a sessile stem.

Output: results/perimage32/<ref>.json with
    names            32 method names
    ids              image ids (filename stems)
    dice, nsd        [n_images][32] matrices (nsd at tau = 2.0)
    gt_fraction      lesion fraction of the ground-truth mask per image
    gt_empty         True where the ground-truth mask has no lesion pixel

Resumable: a finished reference is skipped, and within a reference progress is
checkpointed to <ref>.partial.json every 30 images.

Usage:
    WB_THREADS=2 python scripts/perimage32.py                 # all built-ins
    WB_THREADS=2 python scripts/perimage32.py --only medetec,fuseg
    WB_THREADS=2 python scripts/perimage32.py --extra NAME --mask-threshold 127
        scores ROOT/data/extra/NAME/{images,masks}, paired by filename stem,
        mask > T, long side resampled to 256 if it exceeds 512.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from woundbench.metrics import dice, nsd                      # noqa: E402
from rank_validity import DATA, OUT                           # noqa: E402

PER = os.path.join(OUT, "perimage32")
KV = os.path.join(ROOT, "data", "kvasir")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
CHECKPOINT_EVERY = 30
N_KVASIR = 120


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


# ---------------------------------------------------------------------------
# loaders: each takes ONE image path and ONE mask path, returns (bgr, mask01)
# ---------------------------------------------------------------------------

def _resample(im, m, max_side):
    s = max_side / max(im.shape[:2])
    wh = (int(round(im.shape[1] * s)), int(round(im.shape[0] * s)))
    im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, wh, interpolation=cv2.INTER_NEAREST)
    return im, m


def load_wound(ip, mp):
    """Same as rank_validity.load_pairs: native size, mask on any non-zero."""
    im = cv2.imread(ip)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if im is None or m is None:
        return None
    return im, (m > 0).astype(np.uint8)


def load_kvasir_one(ip, mp, max_side=256, threshold=127):
    """Same as kvasir_clinical.load_kvasir: long side > 256 resampled to 256,
    INTER_AREA for the image, INTER_NEAREST for the mask, then mask > 127."""
    im = cv2.imread(ip)
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if im is None or m is None:
        return None
    if max(im.shape[:2]) > max_side:
        im, m = _resample(im, m, max_side)
    return im, (m > threshold).astype(np.uint8)


def make_extra_loader(threshold):
    def load(ip, mp):
        im = cv2.imread(ip)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if im is None or m is None:
            return None
        if m.shape[:2] != im.shape[:2]:
            m = cv2.resize(m, (im.shape[1], im.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        if max(im.shape[:2]) > 512:
            im, m = _resample(im, m, 256)
        return im, (m > threshold).astype(np.uint8)
    return load


LOADERS = {"wound": load_wound, "kvasir": load_kvasir_one}


# ---------------------------------------------------------------------------
# reference specs: name -> dict(images=[...], masks=[...], loader=str)
# ---------------------------------------------------------------------------

def wound_pairs(img_dir, lab_dir, start=0, stop=None):
    """The exact pairing and slicing rule of rank_validity.load_pairs."""
    ext = (".png", ".jpg", ".jpeg")
    imgs = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                  if p.lower().endswith(ext))
    labs = sorted(p for p in glob.glob(os.path.join(lab_dir, "*"))
                  if p.lower().endswith(ext))
    lab_by = {stem(p): p for p in labs}
    pairs = [(i, lab_by[stem(i)]) for i in imgs if stem(i) in lab_by][start:stop]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def kvasir_mask_for(base, image_path):
    cand = sorted(glob.glob(os.path.join(base, "masks", stem(image_path) + ".*")))
    return cand[0] if cand else None


def kvasir_specs():
    full = os.path.join(KV, "Kvasir-SEG")
    sess = os.path.join(KV, "sessile-main-Kvasir-SEG")
    full_imgs = sorted(glob.glob(os.path.join(full, "images", "*")))
    sess_imgs = sorted(glob.glob(os.path.join(sess, "images", "*")))
    sess_stems = {stem(p) for p in sess_imgs}
    assert len(full_imgs) == 1000 and len(sess_imgs) == 196
    assert sess_stems <= {stem(p) for p in full_imgs}

    out = {}
    for name, base, pool in (
            ("kvasir_disjoint", full,
             [p for p in full_imgs if stem(p) not in sess_stems]),
            ("kvasir_sessile", sess, sess_imgs)):
        imgs, masks = [], []
        for p in pool:
            mp = kvasir_mask_for(base, p)
            if mp is None:
                continue
            imgs.append(p)
            masks.append(mp)
            if len(imgs) == N_KVASIR:
                break
        out[name] = {"images": imgs, "masks": masks, "loader": "kvasir"}

    a = {stem(p) for p in out["kvasir_disjoint"]["images"]}
    b = {stem(p) for p in out["kvasir_sessile"]["images"]}
    assert not (a & b), f"Kvasir references overlap on {len(a & b)} images"
    assert len(a) == N_KVASIR and len(b) == N_KVASIR
    return out


def builtin_specs():
    specs = {}
    W = [
        ("medetec", "Medetec_foot_ulcer_224/train", 0, None),
        ("fuseg", "Foot Ulcer Segmentation Challenge/validation", 0, 120),
        ("azh", "wound_dataset/test", 0, 150),
        ("azh_train_slice", "wound_dataset/train", 0, 150),
    ]
    for name, sub, a, b in W:
        imgs, masks = wound_pairs(os.path.join(DATA, sub, "images"),
                                  os.path.join(DATA, sub, "labels"), a, b)
        specs[name] = {"images": imgs, "masks": masks, "loader": "wound"}
    specs.update(kvasir_specs())
    return specs


def extra_spec(name):
    base = os.path.join(ROOT, "data", "extra", name)
    imgs = sorted(p for p in glob.glob(os.path.join(base, "images", "*"))
                  if p.lower().endswith(IMG_EXT))
    masks = sorted(p for p in glob.glob(os.path.join(base, "masks", "*"))
                   if p.lower().endswith(IMG_EXT))
    by = {}
    for p in masks:
        by.setdefault(stem(p), p)
    pi = [p for p in imgs if stem(p) in by]
    if not pi:
        raise SystemExit(f"no image/mask pairs under {base}")
    return {"images": pi, "masks": [by[stem(p)] for p in pi], "loader": "extra"}


# ---------------------------------------------------------------------------

def build_registry():
    import torch
    torch.set_num_threads(int(os.environ.get("WB_THREADS", "2")))
    from woundbench import deep as D
    from woundbench.combined import CombinedRegistry
    classical = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(OUT, "deep")).load()
    thr = json.load(open(os.path.join(OUT, "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    names = reg.available()
    assert len(names) == 32, f"only {len(names)} methods available"
    return reg, names


def score_reference(name, spec, reg, names, loader, budget=None, log=print):
    """Score one reference, checkpointing. Returns True when complete."""
    final = os.path.join(PER, name + ".json")
    part = os.path.join(PER, name + ".partial.json")
    if os.path.exists(final):
        log(f"{name}: done already")
        return True

    imgs, masks = spec["images"], spec["masks"]
    st = {"ids": [], "dice": [], "nsd": [], "gt_fraction": [],
          "gt_empty": [], "skipped": [], "next": 0}
    if os.path.exists(part):
        st = json.load(open(part))
        assert st["names"] == names
        log(f"{name}: resuming at {st['next']}/{len(imgs)}")

    def save(path):
        blob = dict(st)
        blob["names"] = names
        blob["reference"] = name
        blob["loader"] = spec["loader"]
        blob["image_paths"] = [os.path.relpath(p, ROOT) for p in imgs]
        blob["mask_paths"] = [os.path.relpath(p, ROOT) for p in masks]
        blob["nsd_tau"] = 2.0
        tmp = path + ".tmp"
        json.dump(blob, open(tmp, "w"))
        os.replace(tmp, path)

    t0 = time.time()
    since = 0
    for k in range(st["next"], len(imgs)):
        pair = loader(imgs[k], masks[k])
        if pair is None:
            st["skipped"].append(os.path.relpath(imgs[k], ROOT))
        else:
            im, gt = pair
            out = reg.run_all(im, names=names)
            st["ids"].append(stem(imgs[k]))
            st["dice"].append([dice(out[n], gt) for n in names])
            st["nsd"].append([nsd(out[n], gt) for n in names])
            st["gt_fraction"].append(float(gt.mean()))
            st["gt_empty"].append(bool(gt.sum() == 0))
        st["next"] = k + 1
        since += 1
        if since >= CHECKPOINT_EVERY:
            save(part)
            since = 0
            log(f"  {name}: {k + 1}/{len(imgs)} ({time.time() - t0:.0f}s)")
            if budget is not None and time.time() > budget:
                log(f"  {name}: time budget reached, rerun to resume")
                return False

    del st["next"]
    save(final)
    if os.path.exists(part):
        os.remove(part)
    log(f"{name}: {len(st['ids'])} images scored, "
        f"{sum(st['gt_empty'])} empty-GT, saved {final}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of built-in references")
    ap.add_argument("--extra", default=None,
                    help="score ROOT/data/extra/NAME instead of the built-ins")
    ap.add_argument("--mask-threshold", type=float, default=127,
                    help="with --extra: mask pixels > T are lesion")
    ap.add_argument("--budget", type=float, default=None,
                    help="stop at the next checkpoint after this many seconds")
    args = ap.parse_args()
    os.makedirs(PER, exist_ok=True)

    if args.extra:
        specs = {args.extra: extra_spec(args.extra)}
        loaders = {args.extra: make_extra_loader(args.mask_threshold)}
    else:
        specs = builtin_specs()
        if args.only:
            want = args.only.split(",")
            specs = {k: v for k, v in specs.items() if k in want}
        loaders = {k: LOADERS[v["loader"]] for k, v in specs.items()}

    todo = [k for k in specs
            if not os.path.exists(os.path.join(PER, k + ".json"))]
    if not todo:
        print("all requested references already scored")
        return
    reg, names = build_registry()
    budget = time.time() + args.budget if args.budget else None
    for k in todo:
        print(f"{k}: {len(specs[k]['images'])} pairs", flush=True)
        if not score_reference(k, specs[k], reg, names, loaders[k], budget,
                               log=lambda s: print(s, flush=True)):
            sys.exit(3)


if __name__ == "__main__":
    main()

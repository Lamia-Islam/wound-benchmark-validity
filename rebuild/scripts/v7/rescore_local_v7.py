#!/usr/bin/env python3
"""Rescore the 32 local methods under alternative post-processing and boundary
tolerances, in the layout of the Kaggle rescore files (kaggle2/k2_rescore.py).

Why: the Kaggle run rescored every trained network (results/kaggle2/rescore/).
The 32 local methods (18 untrained + 14 trained from scratch on FUSeg, the
perimage32 names) need the same rescoring so that pooled robustness analyses
can use one file layout for every method.

Methods and models are loaded exactly as scripts/perimage32.py loads them
(perimage32.build_registry: results/models.pkl, results/deep/, results/
deep_thresholds.json) and run per image in the same order as
CombinedRegistry.run_all.

Method types and variants (per image, per method):

  Probability-output methods (14, all trained from scratch on FUSeg):
  logreg_all, gbm_all, logreg_colour (pixel classifiers, threshold
  registry.learned_threshold = 0.9), unet (threshold registry.unet_threshold
  = 0.5) and the ten smp networks (thresholds from deep_thresholds.json).
  With p the probability map at native size (cast to float32) and t the
  method's own threshold:
    P0  postprocess_prob(p, t)            the study pipeline (Gaussian sigma 2
                                          on p, threshold, small components)
    P1  postprocess((p >= t))             the untrained methods' pipeline
                                          (5 px close, 5 px open, small comps)
    P2  remove_small_components((p >= t)) small-component removal only
  These are the definitions of k2_rescore.score_image.
  grid_dice: Dice of postprocess_prob(p, g) for the 18 grid thresholds g read
  from a Kaggle rescore file (0.05 to 0.90 step 0.05). Stored for these 14
  methods only; "grid_names" lists them in grid_dice order.

  Binary-mask methods (18 untrained classical methods, woundbench.classical):
  the raw mask is CLASSICAL_METHODS[name](bgr), the value MethodRegistry.run
  passes to postprocess. So:
    P0  postprocess(raw)                  the study pipeline
    P1  postprocess(raw)                  identical to P0 by definition (the
                                          untrained pipeline IS the study
                                          pipeline for these methods); copied
    P2  remove_small_components(raw)      small-component removal only
  Nothing else is added. Every classical method has a separable raw mask, so
  no method needs P1 = P2 = P0; "variant_note" records the mapping per method.

  A method that raises on an image gets an all-zero mask for every variant
  (and zero grid masks), exactly as CombinedRegistry.run_all does for P0.
  Counts are stored in "failures".

Metrics per variant: Dice (woundbench.metrics.dice) and NSD at tau = 1, 2, 3,
5 px and at 1% of the image diagonal, from ONE surface_distances call per
(pred, gt), with the conventions of woundbench.metrics.nsd (both empty = 1, one
empty = 0). nsd_multi and selftest_nsd are copied from kaggle2/k2_rescore.py.

Sets: the 11 references and the control azh_train_slice, with images, masks
and loaders taken from results/perimage32/<set>.json (image_paths,
mask_paths, loader; extra loader with mask threshold 127 as in
scripts/kaggle_export.py), so the image order is that of perimage32. No
exclusions are applied. fuseg_r256: the fuseg reference images as perimage32
loads them, resampled so the long side is 256 when it exceeds 256 (INTER_AREA
image, INTER_NEAREST on the 0/255 mask, then > 127), exactly the rule of
kaggle2/k2_common.fuseg_r256_dir; same order as fuseg.

Checks (stored in each output under "checks" and printed):
  - gt_fraction of every image equals perimage32's (sets other than r256).
  - loaded image and mask equal the Kaggle kit's refs/<set>/NNNN_img.png and
    mask (for fuseg_r256: the k2_common rule applied to kit refs/fuseg), when
    the kit folder /home/claude/kaggle_kit exists; also gt_fraction equals the
    Kaggle rescore file of the same set.
  - P0 reproduction: P0 dice and P0 nsd2 against results/perimage32/<set>.json
    dice and nsd (tau 2) for every image and method; max abs difference.
    Not possible for fuseg_r256 (perimage32 has no such set).
  - nsd2 against woundbench.metrics.nsd directly, every variant and method on
    every 10th image; and the synthetic NSD self-test at start.

Values are stored unrounded (the Kaggle files round to 5 decimals).

Output: results/v7/rescore_local/<set>.json, written atomically; within a set
progress is checkpointed to <set>.partial.json every 15 images, so the script
is resumable. Log lines go to stdout.

Usage (from the rebuild folder):
  WB_THREADS=2 python scripts/v7/rescore_local_v7.py [--only a,b] [--limit N]
  python scripts/v7/rescore_local_v7.py --summary   (check table, summary.json)
  (--limit N scores only the first N images of each set into a scratch
  folder given by --out, for testing.)
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import perimage32 as P                                            # noqa: E402
from woundbench.classical import CLASSICAL_METHODS                # noqa: E402
from woundbench.methods import LEARNED_NAMES                      # noqa: E402
from woundbench.metrics import dice, nsd, surface_distances       # noqa: E402
from woundbench.postproc import (postprocess, postprocess_prob,   # noqa: E402
                                 remove_small_components)

PER = os.path.join(ROOT, "results", "perimage32")
OUTDIR = os.path.join(ROOT, "results", "v7", "rescore_local")
KAGGLE_RS = os.path.join(ROOT, "results", "kaggle2", "rescore")
KIT = "/home/claude/kaggle_kit"
SETS = ["medetec", "co2wounds", "fuseg", "azh", "kvasir_disjoint",
        "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib", "ph2",
        "tn3k_test", "azh_train_slice", "fuseg_r256"]
VARIANTS = ["P0", "P1", "P2"]
METRICS = ["dice", "nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1"]
TAUS = [1.0, 2.0, 3.0, 5.0]
DIAG_FRAC = 0.01
CHECKPOINT_EVERY = 15
NSD_DIRECT_EVERY = 10


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ------------------------------------------------- metrics (from k2_rescore)

def nsd_multi(pred, gt, taus):
    """Copied from kaggle2/k2_rescore.py. NSD at several tolerances from one
    surface_distances call. Same conventions as woundbench.metrics.nsd: both
    empty = 1, one empty = 0."""
    a, b = surface_distances(pred, gt)
    if len(a) == 0 and len(b) == 0:
        return [1.0] * len(taus)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return [0.0] * len(taus)
    den = len(a) + len(b)
    return [float(((a <= tau).sum() + (b <= tau).sum()) / den) for tau in taus]


def variant_metrics(mask, gt, diag_tau):
    return [dice(mask, gt)] + nsd_multi(mask, gt, TAUS + [diag_tau])


def selftest_nsd(n=300, seed=0):
    """Copied from kaggle2/k2_rescore.py: nsd_multi at every tau must equal
    woundbench.metrics.nsd exactly on random blobs, empty, single-pixel and
    full masks. Returns the number of comparisons; raises on any mismatch."""
    rng = np.random.default_rng(seed)

    def blob(h, w):
        r = rng.random()
        if r < 0.08:
            return np.zeros((h, w), np.uint8)
        if r < 0.12:
            m = np.zeros((h, w), np.uint8)
            m[rng.integers(h), rng.integers(w)] = 1
            return m
        if r < 0.14:
            return np.ones((h, w), np.uint8)
        f = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0),
                             float(rng.uniform(1, 8)))
        return (f > np.quantile(f, rng.uniform(0.5, 0.995))).astype(np.uint8)

    checks = 0
    for i in range(n):
        h, w = int(rng.integers(8, 260)), int(rng.integers(8, 260))
        p, g = blob(h, w), blob(h, w)
        diag = DIAG_FRAC * float(np.hypot(h, w))
        taus = TAUS + [diag]
        got = nsd_multi(p, g, taus)
        for tau, v in zip(taus, got):
            ref = nsd(p, g, tau)
            if v != ref:
                raise AssertionError(f"NSD mismatch case {i} tau {tau}: {v} != {ref}")
            checks += 1
        if got[1] != nsd(p, g):
            raise AssertionError(f"NSD(2) mismatch case {i}")
        checks += 1
    return checks


# ------------------------------------------------------------------ data

def r256(im, m01):
    """kaggle2/k2_common.fuseg_r256_dir applied to one (bgr, mask01) pair. The
    kit mask file is mask01 * 255, so the mask is resized as 0/255 and then
    binarised at > 127."""
    m = (m01 > 0).astype(np.uint8) * 255
    s = 256 / max(im.shape[:2])
    if s < 1:
        wh = (round(im.shape[1] * s), round(im.shape[0] * s))
        im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, wh, interpolation=cv2.INTER_NEAREST)
    return im, (m > 127).astype(np.uint8)


def load_set(name):
    """Returns (ids, images, masks, meta). Images in perimage32 order."""
    base = "fuseg" if name == "fuseg_r256" else name
    d = json.load(open(os.path.join(PER, base + ".json")))
    loaders = dict(P.LOADERS)
    loaders["extra"] = P.make_extra_loader(127)
    load = loaders[d["loader"]]
    X, Y = [], []
    gt_ok = 0
    for k, (ip, mp) in enumerate(zip(d["image_paths"], d["mask_paths"])):
        pair = load(os.path.join(ROOT, ip), os.path.join(ROOT, mp))
        if pair is None:
            raise RuntimeError(f"{name}: image {k} failed to load")
        im, m = pair
        gt_ok += abs(float(m.mean()) - d["gt_fraction"][k]) < 1e-12
        if name == "fuseg_r256":
            im, m = r256(im, m)
        X.append(im)
        Y.append(m)
    if gt_ok != len(X) or len(X) != len(d["ids"]):
        raise RuntimeError(f"{name}: {gt_ok}/{len(X)} masks match perimage32 "
                           f"gt_fraction, {len(d['ids'])} ids")
    meta = {"loader": d["loader"], "image_paths": d["image_paths"],
            "mask_paths": d["mask_paths"], "perimage32_source": base,
            "resampling": ("long side to 256 if above, INTER_AREA image, "
                           "INTER_NEAREST 0/255 mask, > 127 (k2_common."
                           "fuseg_r256_dir)") if name == "fuseg_r256" else
            "perimage32 loader"}
    return list(d["ids"]), X, Y, meta


def kit_check(name, X, Y):
    """Pixel identity with the Kaggle kit images, and gt_fraction identity
    with a Kaggle rescore file of the same set. Returns a dict."""
    out = {}
    src = os.path.join(KIT, "refs", "fuseg" if name == "fuseg_r256" else name)
    if os.path.isdir(src):
        same = 0
        for k, (im, m) in enumerate(zip(X, Y)):
            ki = cv2.imread(os.path.join(src, f"{k:04d}_img.png"))
            km = cv2.imread(os.path.join(src, f"{k:04d}_mask.png"),
                            cv2.IMREAD_GRAYSCALE)
            if name == "fuseg_r256":
                # k2_common rule on the kit files (mask written as 0/255)
                s = 256 / max(ki.shape[:2])
                if s < 1:
                    wh = (round(ki.shape[1] * s), round(ki.shape[0] * s))
                    ki = cv2.resize(ki, wh, interpolation=cv2.INTER_AREA)
                    km = cv2.resize(km, wh, interpolation=cv2.INTER_NEAREST)
                km = ((km > 127) * 255).astype(np.uint8)
            km = (km > 0).astype(np.uint8)
            same += (ki is not None and ki.shape == im.shape
                     and np.array_equal(ki, im) and np.array_equal(km, m))
        n_kit = len([f for f in os.listdir(src) if f.endswith("_img.png")])
        out["kit_identical_pairs"] = int(same)
        out["kit_n"] = int(n_kit)
    kf = os.path.join(KAGGLE_RS, "pretrained_s0", name + ".json")
    if os.path.exists(kf):
        kd = json.load(open(kf))
        gf = [float(m.mean()) for m in Y]
        out["kaggle_rescore_n"] = int(kd["n"])
        out["kaggle_gt_fraction_max_abs_diff"] = (
            float(max(abs(a - b) for a, b in zip(gf, kd["gt_fraction"])))
            if kd["n"] == len(gf) else None)
    return out


# ---------------------------------------------------------------- methods

def build():
    reg, names = P.build_registry()          # sets torch threads from WB_THREADS
    classical, zoo = reg.classical, reg.zoo
    kinds, thrs = {}, {}
    for n in names:
        if n in zoo.models:
            kinds[n], thrs[n] = "prob", float(reg.thresholds.get(n, 0.5))
        elif n in LEARNED_NAMES:
            kinds[n] = "prob"
            thrs[n] = float(classical.unet_threshold if n == "unet"
                            else classical.learned_threshold)
        elif n in CLASSICAL_METHODS:
            kinds[n], thrs[n] = "mask", None
        else:
            raise KeyError(n)
    if not classical.fitted:
        raise RuntimeError("learned methods in models.pkl are not fitted")
    return reg, names, kinds, thrs


def method_output(reg, name, kind, bgr):
    """('prob', p float32) or ('mask', raw uint8). Mirrors CombinedRegistry.run
    and MethodRegistry.run up to the post-processing step."""
    if kind == "prob":
        if name in reg.zoo.models:
            p = reg.zoo.probability(name, bgr)
        else:
            p = reg.classical.probability(name, bgr)
        return p.astype(np.float32)
    return CLASSICAL_METHODS[name](bgr)


def variant_masks(kind, out, t):
    if kind == "prob":
        raw = (out >= t).astype(np.uint8)
        return [postprocess_prob(out, t), postprocess(raw),
                remove_small_components(raw)]
    m0 = postprocess(out)
    return [m0, m0, remove_small_components(out)]


VARIANT_NOTE = {
    "prob": ("P0 = postprocess_prob(p, t) (study pipeline); P1 = postprocess(p >= t)"
             " (untrained pipeline); P2 = remove_small_components(p >= t)"),
    "mask": ("raw = woundbench.classical.CLASSICAL_METHODS[name](bgr); P0 = "
             "postprocess(raw) (study pipeline); P1 = postprocess(raw), identical "
             "to P0 because the untrained pipeline is the study pipeline for this "
             "method; P2 = remove_small_components(raw)"),
}


# ------------------------------------------------------------------ scoring

def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def p0_check(name, ids, names, V):
    """Max |P0 dice - perimage32 dice| and |P0 nsd2 - perimage32 nsd|."""
    if name == "fuseg_r256":
        return {"available": False,
                "reason": "perimage32 has no fuseg_r256 set"}
    d = json.load(open(os.path.join(PER, name + ".json")))
    assert d["names"] == names and d["ids"] == ids
    v = np.asarray(V, dtype=float)                     # [img][method][3][6]
    dd = np.abs(v[:, :, 0, 0] - np.asarray(d["dice"], float))
    dn = np.abs(v[:, :, 0, 2] - np.asarray(d["nsd"], float))
    per_method = {}
    for j, n in enumerate(names):
        per_method[n] = {"max_abs_diff_dice": float(dd[:, j].max()),
                         "max_abs_diff_nsd2": float(dn[:, j].max()),
                         "images_dice_diff_gt_1e-6": int((dd[:, j] > 1e-6).sum()),
                         "images_nsd2_diff_gt_1e-6": int((dn[:, j] > 1e-6).sum())}
    return {"available": True, "n_images": int(v.shape[0]),
            "n_methods": int(v.shape[1]),
            "max_abs_diff_dice": float(dd.max()),
            "max_abs_diff_nsd2": float(dn.max()),
            "pass_1e-6": bool(max(dd.max(), dn.max()) <= 1e-6),
            "per_method": per_method}


def score_set(name, reg, names, kinds, thrs, grid, outdir, limit=None):
    final = os.path.join(outdir, name + ".json")
    part = os.path.join(outdir, name + ".partial.json")
    if os.path.exists(final):
        log(f"{name}: done already")
        return
    ids, X, Y, meta = load_set(name)
    if limit:
        ids, X, Y = ids[:limit], X[:limit], Y[:limit]
    checks = kit_check(name, X, Y) if not limit else {}
    log(f"{name}: {len(X)} images, checks {checks}")
    grid_names = [n for n in names if kinds[n] == "prob"]

    st = {"V": [], "G": [], "fail": {n: 0 for n in names},
          "nsd_direct_max": 0.0, "nsd_direct_n": 0, "next": 0}
    if os.path.exists(part):
        st = json.load(open(part))
        assert st["names"] == names, "method list changed"
        log(f"{name}: resuming at {st['next']}/{len(X)}")
    t0 = time.time()
    start = st["next"]
    since = 0
    for k in range(start, len(X)):
        im, gt = X[k], Y[k]
        diag_tau = DIAG_FRAC * float(np.hypot(*gt.shape[:2]))
        vi, gi = [], []
        for n in names:
            kind, t = kinds[n], thrs[n]
            try:
                out = method_output(reg, n, kind, im)
                masks = variant_masks(kind, out, t)
            except Exception:
                st["fail"][n] += 1
                out = None
                z = np.zeros(im.shape[:2], np.uint8)
                masks = [z, z, z]
            vm = [variant_metrics(m, gt, diag_tau) for m in masks]
            vi.append(vm)
            if kind == "prob":
                if out is None:
                    gi.append([dice(np.zeros(im.shape[:2], np.uint8), gt)]
                              * len(grid))
                else:
                    gi.append([dice(postprocess_prob(out, float(g)), gt)
                               for g in grid])
            if k % NSD_DIRECT_EVERY == 0:
                for m, row in zip(masks, vm):
                    st["nsd_direct_max"] = max(st["nsd_direct_max"],
                                               abs(row[2] - nsd(m, gt)))
                    st["nsd_direct_n"] += 1
        st["V"].append(vi)
        st["G"].append(gi)
        st["next"] = k + 1
        since += 1
        if since >= CHECKPOINT_EVERY:
            blob = dict(st)
            blob["names"] = names
            _dump(part, blob)
            since = 0
            el = time.time() - t0
            log(f"  {name}: {k + 1}/{len(X)} ({el:.0f}s, "
                f"{el / (k + 1 - start):.1f} s/img)")

    chk = p0_check(name, ids, names, st["V"]) if not limit else {
        "available": False, "reason": "--limit test run"}
    if limit and name != "fuseg_r256":
        d = json.load(open(os.path.join(PER, name + ".json")))
        v = np.asarray(st["V"], float)
        chk = {"available": True, "limit": limit,
               "max_abs_diff_dice": float(np.abs(v[:, :, 0, 0] - np.asarray(d["dice"][:limit])).max()),
               "max_abs_diff_nsd2": float(np.abs(v[:, :, 0, 2] - np.asarray(d["nsd"][:limit])).max())}
    checks.update({"p0_reproduction": chk,
                   "nsd2_vs_woundbench_nsd": {
                       "comparisons": st["nsd_direct_n"],
                       "max_abs_diff": st["nsd_direct_max"],
                       "sample": f"every {NSD_DIRECT_EVERY}th image, all "
                                 "methods and variants"}})
    obj = {"set": name, "names": names, "ids": ids, "n": len(X),
           "method_kind": [kinds[n] for n in names],
           "thresholds": [thrs[n] for n in names],
           "variants": VARIANTS, "metrics": METRICS, "var": st["V"],
           "grid_thresholds": [float(g) for g in grid],
           "grid_names": grid_names, "grid_dice": st["G"],
           "grid_note": ("grid_dice[image][j] is for grid_names[j]: only the "
                         "probability-output methods; omitted for the 18 "
                         "binary-mask methods, which have no probability map"),
           "variant_note": {n: VARIANT_NOTE[kinds[n]] for n in names},
           "failures": st["fail"],
           "gt_empty": [bool(g.sum() == 0) for g in Y],
           "gt_fraction": [float(g.mean()) for g in Y],
           "exclusions_applied": False, "values_rounded": False,
           "checks": checks, **meta}
    _dump(final, obj)
    if os.path.exists(part):
        os.remove(part)
    v = np.asarray(st["V"], float)
    log(f"{name}: saved {final} ({time.time() - t0:.0f}s this run); mean Dice "
        f"P0/P1/P2 " + "/".join(f"{v[:, :, j, 0].mean():.4f}" for j in range(3))
        + f"; P0 check: " + (f"max|dDice| {chk['max_abs_diff_dice']:.2e}, "
                             f"max|dNSD2| {chk['max_abs_diff_nsd2']:.2e}"
                             if chk.get("available") else chk.get("reason", ""))
        + f"; nsd direct max diff {st['nsd_direct_max']:.1e} "
          f"({st['nsd_direct_n']} comparisons); failures "
        + str({n: c for n, c in st["fail"].items() if c}))


def summary(outdir=OUTDIR):
    """Markdown table of the checks and mean Dice per variant, read from the
    finished outputs; also written to <outdir>/summary.json."""
    rows, js = [], {}
    hdr = ("| set | images | methods | P0 max abs diff Dice | P0 max abs diff NSD2 "
           "| within 1e-6 | kit pixel-identical pairs | NSD2 vs metrics.nsd "
           "(comparisons, max diff) | mean Dice P0 / P1 / P2, untrained 18 "
           "| mean Dice P0 / P1 / P2, from-scratch 14 | failures |")
    rows.append(hdr)
    rows.append("|" + "---|" * 11)
    for s in SETS:
        f = os.path.join(outdir, s + ".json")
        if not os.path.exists(f):
            rows.append(f"| {s} | not done |" + " |" * 9)
            continue
        d = json.load(open(f))
        v = np.asarray(d["var"], float)
        mk = np.asarray(d["method_kind"])
        c = d["checks"]
        p0 = c["p0_reproduction"]
        nd = c["nsd2_vs_woundbench_nsd"]
        mU = [float(v[:, mk == "mask", j, 0].mean()) for j in range(3)]
        mF = [float(v[:, mk == "prob", j, 0].mean()) for j in range(3)]
        fl = {n: k for n, k in d["failures"].items() if k}
        js[s] = {"n_images": d["n"], "n_methods": len(d["names"]),
                 "n_untrained": int((mk == "mask").sum()),
                 "n_from_scratch": int((mk == "prob").sum()),
                 "p0": {k: p0.get(k) for k in ("available", "max_abs_diff_dice",
                                               "max_abs_diff_nsd2", "pass_1e-6",
                                               "reason")},
                 "kit_identical_pairs": c.get("kit_identical_pairs"),
                 "kit_n": c.get("kit_n"),
                 "kaggle_gt_fraction_max_abs_diff":
                     c.get("kaggle_gt_fraction_max_abs_diff"),
                 "nsd2_direct": nd, "mean_dice_untrained": mU,
                 "mean_dice_from_scratch": mF, "failures": fl}
        if p0.get("available"):
            p0s = (f"{p0['max_abs_diff_dice']:.1e} | {p0['max_abs_diff_nsd2']:.1e}"
                   f" | {'yes' if max(p0['max_abs_diff_dice'], p0['max_abs_diff_nsd2']) <= 1e-6 else 'NO'}")
        else:
            p0s = "n/a | n/a | n/a (no perimage32 set)"
        rows.append(
            f"| {s} | {d['n']} | {len(d['names'])} | {p0s} | "
            f"{c.get('kit_identical_pairs')}/{c.get('kit_n')} | "
            f"{nd['comparisons']}, {nd['max_abs_diff']:.1e} | "
            + " / ".join(f"{x:.4f}" for x in mU) + " | "
            + " / ".join(f"{x:.4f}" for x in mF) + f" | {fl or 'none'} |")
    _dump(os.path.join(outdir, "summary.json"), js)
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--summary", action="store_true",
                    help="print the check table from finished outputs and exit")
    args = ap.parse_args()
    if args.summary:
        summary(args.out)
        return
    if args.limit and os.path.abspath(args.out) == OUTDIR:
        raise SystemExit("--limit needs --out pointing to a scratch folder")
    sets = args.only.split(",") if args.only else SETS
    n = selftest_nsd(300)
    log(f"NSD self-test: {n} comparisons with woundbench.metrics.nsd, all equal")
    kf = json.load(open(os.path.join(KAGGLE_RS, "pretrained_s0", "fuseg.json")))
    grid = [float(g) for g in kf["grid_thresholds"]]
    assert len(grid) == 18
    assert np.allclose(grid, np.round(np.arange(0.05, 0.91, 0.05), 2))
    reg, names, kinds, thrs = build()
    log(f"{len(names)} methods: {sum(k == 'prob' for k in kinds.values())} "
        f"probability-output, {sum(k == 'mask' for k in kinds.values())} binary; "
        f"thresholds { {n: t for n, t in thrs.items() if t is not None} }; "
        f"torch threads {os.environ.get('WB_THREADS', '2')}")
    T0 = time.time()
    for s in sets:
        ts = time.time()
        score_set(s, reg, names, kinds, thrs, grid, args.out, args.limit)
        log(f"{s}: {time.time() - ts:.0f}s; total {time.time() - T0:.0f}s")
    log("all requested sets done")


if __name__ == "__main__":
    main()

"""Shared interface for the phase-2 GPU run (kit 2). Every k2_* stage imports
this module; nothing else defines paths, names or model loading.

Layout under OUT (default /kaggle/working/out):
  ckpt_pretrained/<base>_in.pt          seed-0 pretrained (copied from the previous run)
  thresholds_pretrained.json            {name: threshold}            (copied)
  ckpt_kvasir/<base>_kv.pt, thresholds_kvasir.json, kvasir_learned.pkl   (copied)
  nnunet/results/...                    nnU-Net model from the previous run (copied)
  ckpt_seeds/<base>_in_s{1,2}.pt        thresholds_seeds.json          (stage seeds)
  ckpt_modern2/<base2>_s{0,1,2}.pt      thresholds_modern2.json        (stage modern2)
  train_report_k2.json                  {name: {"sweep_dice": d, "threshold": t, "seconds": s}}
  extra_sets/fuseg_r256/NNNN_img.png    FUSeg reference resampled to long side 256
  woundseg/refs/NNNN_img.png ...        fifth wound dataset (stage woundseg)
  woundseg/meta.json
  scores/<group>/<set>.json             per-image scores, kit_run layout
  rescore/<family>/<set>.json           post-processing / tolerance variants
  nnunet_masks/<set>.npz                nnU-Net binary masks for SAM prompts

A checkpoint file holds the model state_dict only. Threshold files map model
name to the decision threshold chosen on FUSeg training images 400 to 549.
"""
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import kit_run as K                                               # noqa: E402
from woundbench.metrics import dice, nsd, surface_distances       # noqa: E402,F401
from woundbench.postproc import (postprocess, postprocess_prob,   # noqa: E402,F401
                                 remove_small_components)

OUT = K.OUT
DRY = K.DRY
DEV = K.DEV
log = K.log
SEEDS_NEW = [1, 2]          # extra seeds for the original ten architectures
SEEDS2 = [0, 1, 2]          # seeds for the eight new architectures

# original ten (seed 0 already trained in run 1): (base, smp arch, encoder)
SPECS = K.SPECS
# eight new modern architectures: (base, smp arch, encoder, extra kwargs)
SPECS2 = [
    ("unet_r50", "Unet", "resnet50", {}),
    ("deeplabv3p_r50", "DeepLabV3Plus", "resnet50", {}),
    ("upernet_convnext_t", "UPerNet", "tu-convnext_tiny.fb_in1k", {}),
    ("unet_swin_t", "Unet", "tu-swin_tiny_patch4_window7_224", {"img_size": 256}),
    ("segformer_mitb2", "Segformer", "mit_b2", {}),
    ("unet_effb3", "Unet", "tu-efficientnet_b3", {}),
    ("fpn_regnety016", "FPN", "tu-regnety_016", {}),
    ("manet_r50", "MAnet", "resnet50", {}),
]


def name_pre(base, seed):
    """Pretrained original-ten model name; seed 0 keeps the run-1 name."""
    return f"{base}_in" if seed == 0 else f"{base}_in_s{seed}"


def name_m2(base, seed):
    return f"{base}_s{seed}"


def build_smp(arch, enc, kwargs=None, weights=None):
    import segmentation_models_pytorch as smp
    return getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3,
                              classes=1, **(kwargs or {}))


# ------------------------------------------------------------ previous run

NN_FINAL = os.path.join("nnunet", "results", "Dataset501_FUSeg",
                        "nnUNetTrainer_WB__nnUNetPlans__2d", "fold_all",
                        "checkpoint_final.pth")


def _prev_ok(d):
    """A run-1 output folder is accepted only if it holds the finished nnU-Net
    model, the DONE_nnunet marker and the full Medetec scores of run 1."""
    try:
        if not (os.path.isfile(os.path.join(d, "DONE_nnunet"))
                and os.path.isfile(os.path.join(d, NN_FINAL))):
            return False
        j = json.load(open(os.path.join(d, "scores", "pretrained", "medetec.json")))
        return j.get("n") == (3 if DRY else 152)
    except Exception:
        return False


def find_prev():
    """Output folder of the run-1 notebook version, attached as an input."""
    env = os.environ.get("PREV_OUT")
    if env and os.path.isdir(env):
        return env
    root = os.environ.get("PREV_SEARCH", "/kaggle/input")
    cands = []
    for r, dirs, _ in os.walk(root, followlinks=True):
        if "ckpt_pretrained" in dirs:
            cands.append(r)
        # do not descend into checkpoint, nnU-Net or image folders
        dirs[:] = [x for x in dirs if not x.startswith(("ckpt", "nnunet", "synth", "refs", "train"))]
    cands = sorted(cands)
    log(f"  run-1 output candidates: {cands}")
    good = [c for c in cands if _prev_ok(c)]
    if not good:
        return None
    if len(good) > 1:
        log(f"  several valid run-1 outputs, using the first: {good}")
    return good[0]


def restore_prev():
    """Copy run-1 checkpoints, thresholds and the nnU-Net model into OUT.
    Copies go to a temporary name first and are renamed when complete, so an
    interrupted copy is never mistaken for a finished one. Returns a dict of
    what was found. Safe to call repeatedly."""
    prev = find_prev()
    found = {"prev": prev}
    if not prev:
        log("  previous run output not found; seed-0 models must be retrained")
        return found
    items = ["ckpt_pretrained", "ckpt_kvasir", "kvasir_learned.pkl",
             "thresholds_pretrained.json", "thresholds_kvasir.json",
             os.path.join("nnunet", "results")]
    for it in items:
        src, dst = os.path.join(prev, it), os.path.join(OUT, it)
        if os.path.exists(dst):
            found[it] = True
            continue
        tmp = dst + ".copying"
        if os.path.isdir(tmp):
            shutil.rmtree(tmp)
        elif os.path.exists(tmp):
            os.remove(tmp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, tmp)
            os.replace(tmp, dst)
            found[it] = True
        elif os.path.isfile(src):
            shutil.copy(src, tmp)
            os.replace(tmp, dst)
            found[it] = True
        else:
            found[it] = False
    log(f"  restored from {prev}: " + ", ".join(f"{k}={v}" for k, v in found.items() if k != "prev"))
    return found


# --------------------------------------------------------------- data sets

def fuseg_r256_dir():
    """FUSeg reference resampled so the long side is 256 (same images, same
    order as refs/fuseg). Built once into a temporary folder and renamed when
    complete, so a partial folder is never used."""
    d = os.path.join(OUT, "extra_sets", "fuseg_r256")
    src = sorted(glob.glob(os.path.join(KIT, "refs", "fuseg", "*_img.png")))
    if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*_img.png"))) == len(src):
        return d
    tmp = d + ".building"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(tmp)
    for f in src:
        im = cv2.imread(f)
        m = cv2.imread(f.replace("_img.png", "_mask.png"), cv2.IMREAD_GRAYSCALE)
        s = 256 / max(im.shape[:2])
        if s < 1:
            wh = (round(im.shape[1] * s), round(im.shape[0] * s))
            im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, wh, interpolation=cv2.INTER_NEAREST)
        b = os.path.basename(f)
        cv2.imwrite(os.path.join(tmp, b), im)
        cv2.imwrite(os.path.join(tmp, b.replace("_img", "_mask")), ((m > 127) * 255).astype(np.uint8))
    os.replace(tmp, d)
    return d


def woundseg_dir():
    d = os.path.join(OUT, "woundseg", "refs")
    return d if os.path.isdir(d) and glob.glob(os.path.join(d, "*_img.png")) else None


def ref_sets(include_extra=True):
    """(kind, name, dir) for the 12 clinical sets of run 1 (11 references +
    control), plus fuseg_r256 and woundseg when available."""
    out = [("refs", n, os.path.join(KIT, "refs", n)) for n in K.MAN["refs"]]
    if include_extra:
        out.append(("refs", "fuseg_r256", fuseg_r256_dir()))
        w = woundseg_dir()
        if w:
            out.append(("refs", "woundseg", w))
    if DRY:
        out = out[:2] + [s for s in out if s[1] in ("fuseg_r256", "woundseg")]
    return out


def synth_sets():
    """(kind, name, dir) for the 21 synthetic sets (regenerated and checked by
    kit_run.synth_root on first use)."""
    return [s for s in K.eval_sets() if s[0] == "synth"]


def score_sets(group, names, predict, sets):
    """Score every set in `sets` for a method group; resumable per set.
    predict(bgr, gt) -> {name: uint8 mask}. Same JSON layout as kit_run."""
    od = os.path.join(OUT, "scores", group)
    os.makedirs(od, exist_ok=True)
    for kind, name, d in sets:
        path = os.path.join(od, name + ".json")
        if os.path.exists(path):
            continue
        X, Y = K.load_eval(d)
        D, N = [], []
        t0 = time.time()
        for im, gt in zip(X, Y):
            pred = predict(im, gt)
            D.append([dice(pred[n], gt) for n in names])
            N.append([nsd(pred[n], gt) for n in names])
        tmp = path + ".tmp"
        json.dump({"group": group, "set": name, "kind": kind, "names": names,
                   "n": len(X), "dice": D, "nsd": N,
                   "gt_empty": [bool(g.sum() == 0) for g in Y],
                   "gt_fraction": [float(g.mean()) for g in Y]}, open(tmp, "w"))
        os.replace(tmp, path)
        md = np.mean(D, axis=0) if D else np.zeros(1)
        log(f"  {group}/{name}: {len(X)} images, {time.time() - t0:.0f}s, "
            f"mean Dice {md.min():.3f} to {md.max():.3f}")


# ------------------------------------------------------------- model zoo

def _thr(file):
    return K.load_json(file, {})


def load_family(family):
    """Load one trained family. Returns {name: (model, threshold, kind)} where
    kind 'pt' = letterboxed 256 px with ImageNet normalisation (kit_run.pt_prob)
    and kind 'sq128' = square 128 px, /255 (kit_run.sq_prob). Missing
    checkpoints are skipped with a log line.
    families: pretrained_s0, pretrained_s1, pretrained_s2, modern2_s0,
    modern2_s1, modern2_s2, kvasir_nets."""
    out = {}
    if family.startswith("pretrained_s"):
        seed = int(family[-1])
        ck = "ckpt_pretrained" if seed == 0 else "ckpt_seeds"
        thr = _thr("thresholds_pretrained.json" if seed == 0 else "thresholds_seeds.json")
        specs = [(name_pre(b, seed), a, e, {}) for b, a, e in SPECS]
        kind = "pt"
    elif family.startswith("modern2_s"):
        seed = int(family[-1])
        ck, thr, kind = "ckpt_modern2", _thr("thresholds_modern2.json"), "pt"
        specs = [(name_m2(b, seed), a, e, kw) for b, a, e, kw in SPECS2]
    elif family == "kvasir_nets":
        ck, thr, kind = "ckpt_kvasir", _thr("thresholds_kvasir.json"), "sq128"
        specs = [(b + "_kv", a, e, {}) for b, a, e in SPECS]
    else:
        raise KeyError(family)
    for name, arch, enc, kw in specs:
        p = os.path.join(OUT, ck, name + ".pt")
        if not os.path.exists(p) or name not in thr:
            log(f"  {family}: {name} missing (checkpoint or threshold), skipped")
            continue
        try:
            m = build_smp(arch, enc, kw).to(DEV)
            m.load_state_dict(torch.load(p, map_location=DEV))
            m.eval()
            out[name] = (m, float(thr[name]), kind)
        except Exception as e:
            log(f"  {family}: {name} failed to load ({type(e).__name__}: {e}), skipped")
    return out


@torch.no_grad()
def prob(entry, bgr):
    m, _, kind = entry
    return K.pt_prob(m, bgr) if kind == "pt" else K.sq_prob(m, bgr)

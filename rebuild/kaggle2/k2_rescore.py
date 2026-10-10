"""Stage rescore: re-score every trained probability-output model on the
clinical sets under alternative post-processing and boundary tolerances.

Models (one "family" per output folder):
  pretrained_s0..s2, modern2_s0..s2, kvasir_nets   via k2_common.load_family
  kvasir_unet   the small U-Net of the Kvasir-trained main pool, from
                OUT/kvasir_learned.pkl (reg.probability("unet", im),
                threshold reg.unet_threshold), named "unet_kv"
  nnunet        run-1 nnU-Net 2D model restored to OUT/nnunet/results,
                named "nnunet_2d", threshold 0.5

The three Kvasir pixel classifiers in kvasir_learned.pkl (logreg_all,
gbm_all, logreg_colour) are deliberately excluded: this stage is about the
trained networks, and those classifiers already use their own swept threshold
on a different grid (0.30 to 0.95).

Sets: k2_common.ref_sets() (11 references + control + fuseg_r256 + woundseg
when present). No synthetic sets.

Per image and model, with p the probability map at native size and t the
model's own threshold:
  grid   Dice of postprocess_prob(p, g) for every g in kit_run.GRID
  P0     postprocess_prob(p, t)                       (the study's pipeline)
  P1     postprocess((p >= t).astype(uint8))          (untrained methods' pipeline)
  P2     remove_small_components((p >= t).astype(uint8))
  metrics per variant: Dice and NSD at tau = 1, 2, 3, 5 px and 1% of the
  image diagonal, from ONE surface_distances call per (pred, gt), with the
  conventions of woundbench.metrics.nsd.

Output OUT/rescore/<family>/<set>.json, resumable per (family, set).
nnU-Net extras: OUT/nnunet_heldout.json and OUT/nnunet_masks/<set>.npz
(P0 masks for the SAM stage; read them with load_nnunet_masks(set)).
"""
import glob
import json
import os
import pickle
import shutil
import time
import traceback

import cv2
import numpy as np
import torch

import k2_common as C
from k2_common import (K, OUT, DRY, DEV, log, dice, nsd, surface_distances,
                       postprocess, postprocess_prob, remove_small_components)

FAMILIES = ["pretrained_s0", "pretrained_s1", "pretrained_s2",
            "modern2_s0", "modern2_s1", "modern2_s2", "kvasir_nets",
            "kvasir_unet", "nnunet"]
VARIANTS = ["P0", "P1", "P2"]
METRICS = ["dice", "nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1"]
TAUS = [1.0, 2.0, 3.0, 5.0]
DIAG_FRAC = 0.01
NN_NAME, NN_THR = "nnunet_2d", 0.5
# run-1 score group used for the consistency check
CHECK = {"pretrained_s0": "pretrained", "kvasir_nets": "kvasir",
         "kvasir_unet": "kvasir", "nnunet": "nnunet"}
RS = os.path.join(OUT, "rescore")
MASKS = os.path.join(OUT, "nnunet_masks")


# --------------------------------------------------------------- metrics

def nsd_multi(pred, gt, taus):
    """NSD at several tolerances from one surface_distances call. Same
    conventions as woundbench.metrics.nsd: both empty = 1, one empty = 0."""
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
    """nsd_multi at every tau must equal woundbench.metrics.nsd exactly, on
    random blobs, empty, single-pixel and full masks. Returns the number of
    (case, tau) comparisons; raises on any mismatch."""
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
        if got[1] != nsd(p, g):          # default tau of the study (2 px)
            raise AssertionError(f"NSD(2) mismatch case {i}")
        checks += 1
    return checks


def score_image(p, t, gt):
    """Grid Dice [18] and variant metrics [3][6] for one probability map.
    Also returns the P0 mask."""
    p = p.astype(np.float32)
    grid = [dice(postprocess_prob(p, float(g)), gt) for g in K.GRID]
    raw = (p >= t).astype(np.uint8)
    m0 = postprocess_prob(p, t)
    masks = [m0, postprocess(raw), remove_small_components(raw)]
    diag_tau = DIAG_FRAC * float(np.hypot(*gt.shape[:2]))
    return grid, [variant_metrics(m, gt, diag_tau) for m in masks], m0


def _r(x):
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return round(float(x), 5)


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def rescore_set(family, names, thrs, prob_fn, set_name, d, keep_p0=False):
    """prob_fn(k, bgr) -> list of probability maps (native size), one per name.
    Writes OUT/rescore/<family>/<set>.json; returns the P0 masks
    [img][model] when keep_p0, else None."""
    X, Y = K.load_eval(d)
    G, V, P0 = [], [], []
    t0 = time.time()
    for k, (im, gt) in enumerate(zip(X, Y)):
        probs = prob_fn(k, im)
        gi, vi, pi = [], [], []
        for p, t in zip(probs, thrs):
            if p.shape != gt.shape:
                raise ValueError(f"{set_name} image {k}: probability {p.shape} "
                                 f"vs mask {gt.shape}")
            g, v, m0 = score_image(p, t, gt)
            gi.append(g)
            vi.append(v)
            pi.append(m0)
        G.append(gi)
        V.append(vi)
        if keep_p0:
            P0.append(pi)
    obj = {"family": family, "set": set_name, "names": list(names),
           "thresholds": [float(t) for t in thrs], "n": len(X),
           "grid_thresholds": [float(g) for g in K.GRID],
           "grid_dice": _r(G), "variants": VARIANTS, "metrics": METRICS,
           "var": _r(V),
           "gt_empty": [bool(g.sum() == 0) for g in Y],
           "gt_fraction": [float(g.mean()) for g in Y]}
    _dump(os.path.join(RS, family, set_name + ".json"), obj)
    v = np.asarray(V, dtype=float).reshape(len(X), len(names), 3, 6) if X else None
    msg = "" if v is None else (" mean Dice P0/P1/P2 " + "/".join(
        f"{v[:, :, j, 0].mean():.3f}" for j in range(3)))
    log(f"  rescore {family}/{set_name}: {len(X)} images x {len(names)} models, "
        f"{time.time() - t0:.0f}s,{msg}")
    return P0 if keep_p0 else None


# ------------------------------------------------------------ families

def _done(family, sets):
    return all(os.path.exists(os.path.join(RS, family, s[1] + ".json")) for s in sets)


def _run_sets(family, names, thrs, prob_fn, sets):
    for kind, sname, d in sets:
        path = os.path.join(RS, family, sname + ".json")
        if os.path.exists(path):
            try:
                old = json.load(open(path))
                if old["names"] == list(names) and old["thresholds"] == [float(t) for t in thrs]:
                    continue
                log(f"  rescore {family}/{sname}: model list or thresholds changed "
                    f"({len(old['names'])} -> {len(names)} models), recomputing")
            except Exception:
                log(f"  rescore {family}/{sname}: unreadable result, recomputing")
        try:
            rescore_set(family, names, thrs, lambda k, im: prob_fn(im), sname, d)
        except Exception:
            log(f"  rescore {family}/{sname} FAILED:\n{traceback.format_exc()}")


def rescore_zoo(family, sets):
    # always load: a family re-scored while its training stage was incomplete
    # is recomputed once more checkpoints exist (names are compared per set)
    fam = C.load_family(family)
    if not fam:
        log(f"  rescore {family}: no model could be loaded, skipped")
        return
    names = list(fam)
    thrs = [fam[n][1] for n in names]
    log(f"  rescore {family}: {len(names)} models")

    def prob_fn(im):
        return [C.prob(fam[n], im) for n in names]
    try:
        _run_sets(family, names, thrs, prob_fn, sets)
    finally:
        del fam
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


def rescore_kvasir_unet(sets):
    family = "kvasir_unet"
    if _done(family, sets):
        log(f"  rescore {family}: all sets done")
        return
    pk = os.path.join(OUT, "kvasir_learned.pkl")
    if not os.path.exists(pk):
        log(f"  rescore {family}: {pk} missing, skipped")
        return
    reg = pickle.load(open(pk, "rb"))
    if reg.unet is None:
        log(f"  rescore {family}: registry has no U-Net, skipped")
        return
    thr = float(reg.unet_threshold)
    log(f"  rescore {family}: unet_kv, threshold {thr:.2f}")
    _run_sets(family, ["unet_kv"], [thr],
              lambda im: [reg.probability("unet", im)], sets)


# -------------------------------------------------------------- nnU-Net

def _mask_path(set_name):
    return os.path.join(MASKS, set_name + ".npz")


def save_nnunet_masks(set_name, masks):
    """masks: list of uint8 0/1 arrays. Stored bit-packed with their shapes."""
    os.makedirs(MASKS, exist_ok=True)
    arrs = {f"m{k:05d}": np.packbits((m > 0).astype(np.uint8).ravel())
            for k, m in enumerate(masks)}
    shapes = np.array([m.shape[:2] for m in masks], np.int64).reshape(-1, 2)
    tmp = _mask_path(set_name) + ".tmp.npz"
    np.savez_compressed(tmp, shapes=shapes, **arrs)
    os.replace(tmp, _mask_path(set_name))


def load_nnunet_masks(set_name):
    """nnU-Net P0 masks (postprocess_prob(p, 0.5)) for one set, in the image
    order of kit_run.load_eval: a list of uint8 0/1 arrays at native size."""
    z = np.load(_mask_path(set_name))
    out = []
    for k, (h, w) in enumerate(z["shapes"]):
        bits = np.unpackbits(z[f"m{k:05d}"], count=int(h) * int(w))
        out.append(bits.reshape(int(h), int(w)).astype(np.uint8))
    return out


def _nn_env():
    """Environment and trainer registration exactly as kit_run.stage_nnunet."""
    import nnunetv2
    base = os.path.join(OUT, "nnunet")
    work = os.environ.get("NNUNET_WORK", base)
    env = dict(os.environ,
               nnUNet_raw=os.path.join(work, "raw"),
               nnUNet_preprocessed=os.path.join(work, "preprocessed"),
               nnUNet_results=os.path.join(base, "results"),
               NNUNET_EPOCHS="1" if DRY else os.environ.get("NNUNET_EPOCHS", "100"))
    if DRY:
        env["NNUNET_ITERS"] = "2"
        env["nnUNet_n_proc_DA"] = "1"
    tdir = os.path.join(os.path.dirname(nnunetv2.__file__), "training",
                        "nnUNetTrainer", "variants", "wb")
    os.makedirs(tdir, exist_ok=True)
    open(os.path.join(tdir, "__init__.py"), "w").close()
    open(os.path.join(tdir, "nnUNetTrainer_WB.py"), "w").write(K.TRAINER_SRC)
    return env, work


def _heldout_files():
    """Files of kit_run.fuseg_split()'s second element (held-out FUSeg
    training images 400 to 549 minus duplicates; 3 in DRY), with the masks
    from fuseg_split itself. The index rule is mirrored here because
    fuseg_split returns arrays only; the images are checked to be identical."""
    _, (Xsw, Ysw), _ = K.fuseg_split()
    dup = set(K.MAN["fuseg_train"]["eval_duplicates"])
    sw = [i for i in range(400, 550) if i not in dup]
    if DRY:
        sw = sw[:3]
    d = os.path.join(K.KIT, "train", "fuseg")
    files = sorted(glob.glob(os.path.join(d, "*_img.png")))
    files = [files[i] for i in sw if i < len(files)]
    if len(files) != len(Xsw) or not all(
            np.array_equal(cv2.imread(f), x) for f, x in zip(files, Xsw)):
        raise RuntimeError("held-out file list does not match kit_run.fuseg_split")
    return files, Ysw


def _nn_prob(pout, key, shape):
    """Channel 1 of 'probabilities', read exactly as kit_run.stage_nnunet."""
    z = np.load(os.path.join(pout, key + ".npz"))["probabilities"]
    p = np.asarray(z)[1].squeeze().astype(np.float32)
    if p.shape != shape[:2]:
        p = cv2.resize(p, (shape[1], shape[0]))
    return p


def rescore_nnunet(sets):
    family = "nnunet"
    heldout_json = os.path.join(OUT, "nnunet_heldout.json")
    todo = [s for s in sets
            if not os.path.exists(os.path.join(RS, family, s[1] + ".json"))
            or not os.path.exists(_mask_path(s[1]))]
    need_heldout = not os.path.exists(heldout_json)
    if not todo and not need_heldout:
        log(f"  rescore {family}: all sets, masks and held-out score done")
        return
    env, work = _nn_env()
    model_dir = os.path.join(env["nnUNet_results"], K.NN_NAME,
                             f"{K.TRAINER}__nnUNetPlans__2d", "fold_all")
    if not os.path.exists(os.path.join(model_dir, "checkpoint_final.pth")):
        log(f"  rescore {family}: no trained model in {model_dir}, skipped")
        return
    pin = os.path.join(work, "rescore_pred_in")
    pout = os.path.join(work, "rescore_pred_out")
    os.makedirs(pin, exist_ok=True)
    keys = []
    for kind, name, d in todo:
        files = sorted(glob.glob(os.path.join(d, "*_img.png")))
        if DRY:
            files = files[:3]
        for k, f in enumerate(files):
            key = f"{name}__{k:04d}"
            keys.append(key)
            dst = os.path.join(pin, key + "_0000.png")
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    hfiles, hY = [], []
    if need_heldout:
        hfiles, hY = _heldout_files()
        for k, f in enumerate(hfiles):
            key = f"heldout__{k:04d}"
            keys.append(key)
            dst = os.path.join(pin, key + "_0000.png")
            if not os.path.exists(dst):
                shutil.copy(f, dst)
    dev = ["-device", "cuda" if torch.cuda.is_available() else "cpu"]
    if not all(os.path.exists(os.path.join(pout, k + ".npz")) for k in keys):
        t0 = time.time()
        K.run(["nnUNetv2_predict", "-i", pin, "-o", pout, "-d", str(K.NN_ID),
               "-c", "2d", "-f", "all", "-tr", K.TRAINER, "--save_probabilities"]
              + dev + (["-npp", "1", "-nps", "1"] if DRY else []), env)
        log(f"  nnU-Net predicted {len(keys)} images in {time.time() - t0:.0f}s")

    ok = True
    if need_heldout:
        try:
            D = [dice(postprocess_prob(_nn_prob(pout, f"heldout__{k:04d}", y.shape),
                                       NN_THR), y) for k, y in enumerate(hY)]
            _dump(heldout_json,
                        {"name": NN_NAME, "threshold": NN_THR, "variant": "P0",
                         "n": len(D), "mean_dice": float(np.mean(D)) if D else None,
                         "dice": _r(D)})
            log(f"  nnU-Net held-out FUSeg ({len(D)} images): mean Dice "
                f"{np.mean(D) if D else float('nan'):.4f}")
        except Exception:
            ok = False
            log(f"  nnU-Net held-out score FAILED:\n{traceback.format_exc()}")
    for kind, name, d in todo:
        try:
            masks = rescore_set(
                family, [NN_NAME], [NN_THR],
                lambda k, im, name=name: [_nn_prob(pout, f"{name}__{k:04d}", im.shape)],
                name, d, keep_p0=True)
            save_nnunet_masks(name, [m[0] for m in masks])
        except Exception:
            ok = False
            log(f"  rescore {family}/{name} FAILED:\n{traceback.format_exc()}")
    if ok:
        shutil.rmtree(pin, ignore_errors=True)
        shutil.rmtree(pout, ignore_errors=True)
    else:
        log(f"  kept {pin} and {pout} for a resumed run (some sets failed)")


# --------------------------------------------------- consistency check

def _run1_scores(group, set_name):
    for root in (OUT, C.find_prev()):
        if root:
            p = os.path.join(root, "scores", group, set_name + ".json")
            if os.path.exists(p):
                return json.load(open(p)), p
    return None, None


def consistency_check(sets):
    """P0 Dice and P0 NSD(2) against the run-1 per-image scores."""
    res = {}
    for family, group in CHECK.items():
        dd, dn, n_img, used = 0.0, 0.0, 0, []
        for kind, sname, d in sets:
            rp = os.path.join(RS, family, sname + ".json")
            if not os.path.exists(rp):
                continue
            s1, src = _run1_scores(group, sname)
            if s1 is None:
                continue
            r = json.load(open(rp))
            if s1["n"] != r["n"]:
                log(f"  check {family}/{sname}: n {r['n']} vs run-1 {s1['n']}, skipped")
                continue
            for j, name in enumerate(r["names"]):
                if name not in s1["names"]:
                    continue
                i1 = s1["names"].index(name)
                for k in range(r["n"]):
                    dd = max(dd, abs(r["var"][k][j][0][0] - s1["dice"][k][i1]))
                    dn = max(dn, abs(r["var"][k][j][0][2] - s1["nsd"][k][i1]))
                    n_img += 1
            used.append(sname)
        if not used:
            log(f"  check {family}: no overlapping run-1 scores, skipped")
            continue
        res[family] = {"group": group, "sets": used, "comparisons": n_img,
                       "max_abs_diff_dice": dd, "max_abs_diff_nsd2": dn}
        flag = "" if max(dd, dn) < 1e-3 else "  <-- LARGE, check for a bug"
        log(f"  check {family} vs run-1 {group} ({n_img} image-model pairs, "
            f"{len(used)} sets): max |dDice| {dd:.2e}, max |dNSD2| {dn:.2e}{flag}")
    if res:
        _dump(os.path.join(RS, "consistency.json"), res)
    return res


# ---------------------------------------------------------------- stage

def stage_rescore():
    t0 = time.time()
    n = selftest_nsd(300)
    log(f"  NSD self-test: {n} comparisons with woundbench.metrics.nsd, all equal")
    C.restore_prev()
    sets = C.ref_sets()
    log(f"  rescore sets: {', '.join(s[1] for s in sets)}")
    for family in FAMILIES:
        try:
            if family == "kvasir_unet":
                rescore_kvasir_unet(sets)
            elif family == "nnunet":
                rescore_nnunet(sets)
            else:
                rescore_zoo(family, sets)
        except Exception:
            log(f"  rescore family {family} FAILED:\n{traceback.format_exc()}")
    try:
        consistency_check(sets)
    except Exception:
        log(f"  consistency check FAILED:\n{traceback.format_exc()}")
    log(f"  rescore stage finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    stage_rescore()

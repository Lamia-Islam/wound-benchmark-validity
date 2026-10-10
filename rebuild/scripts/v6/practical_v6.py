#!/usr/bin/env python3
"""V6 practical analysis: what the modern group (M) means for someone choosing
a wound-segmentation model.

Definitions (all Dice values are per-image clinical Dice from perimage_v6.load):
- Per-image Dice on an empty reference mask is 1 when the prediction is also
  empty and 0 otherwise (the stored convention, checked on AZH).
- Mean Dice: mean over ALL kept images, empty references included.
- Bootstrap: resample resampling units (photograph for Medetec and AZH, image
  otherwise) with replacement, B = 2000 draws, every image of a drawn unit
  enters with the unit's draw count as weight. The same draws are used for all
  models of one dataset (paired). Intervals are 2.5 and 97.5 percentiles.
  Generator: numpy default_rng([SEED, dataset index]).
- Top model: highest point-estimate mean Dice. "Indistinguishable from the top":
  the 95% paired interval of (top minus model) has lower bound <= 0.
- Selection loss A -> B: best mean Dice on B minus mean Dice on B of the model
  with the highest mean Dice on A. Random choice: mean over the 11 models of
  (best on B minus model on B).
- Synthetic chooser: generator B at amplitude 40 (and 11); per seed the mean
  Dice over all 45 images, then the mean over seeds 2024, 7, 101.
- Miss: reference non-empty and Dice < t (t = 0.1 main; 0.05, 0.2 sensitivity).
  Miss rate = misses / non-empty images. Dice when detected = mean Dice over
  non-empty images with Dice >= t.
- Spearman: scipy.stats.spearmanr (average ranks for ties) across the 11 M
  models between two wound datasets; reported as mean over the 6 pairs.
"""
import glob
import json
import os
import sys
from itertools import combinations

import cv2
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import perimage_v6 as P                                           # noqa: E402

SEED = 20261010
B = 2000
THRESH = (0.05, 0.1, 0.2)
WOUND = list(P.WOUND)
CTRL = P.CONTROL
SETS = WOUND + [CTRL]
M = list(P.GROUPS["M"])
PT = [n for n in M if n != "nnunet_2d"]
NN = "nnunet_2d"
assert len(M) == 11 and len(PT) == 10
KIT = "/home/claude/kaggle_kit"
RAW = os.path.join(ROOT, "data", "wound-segmentation", "data")
OUT = os.path.join(ROOT, "results", "v6")


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def boot_weights(units, rng):
    u, inv = np.unique(units, return_inverse=True)
    draws = rng.integers(0, len(u), size=(B, len(u)))
    counts = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    return counts[:, inv]                                           # B x images


DATA = {s: P.load(s) for s in P.REFS + [CTRL]}


def sub(s, names):
    return DATA[s]["X"][:, P.cols(names)]


# ---------------------------------------------------------------- 1 and 2
res = {"definitions": __doc__, "seed": SEED, "B": B, "models": M}
per_set = {}
means = {}
for k, s in enumerate(SETS):
    D = DATA[s]
    X = sub(s, M)
    rng = np.random.default_rng([SEED, k])
    W = boot_weights(D["units"], rng)
    BM = (W @ X) / W.sum(1, keepdims=True)                         # B x models
    mu = X.mean(0)
    means[s] = dict(zip(M, mu))
    top = int(np.argmax(mu))
    rows = {}
    n_ind = 0
    for j, n in enumerate(M):
        r = {"mean": r4(mu[j]), "ci": [r4(np.percentile(BM[:, j], 2.5)),
                                       r4(np.percentile(BM[:, j], 97.5))]}
        if j != top:
            d = BM[:, top] - BM[:, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            r["diff_from_top"] = r4(mu[top] - mu[j])
            r["diff_ci"] = [r4(lo), r4(hi)]
            r["indistinguishable"] = bool(lo <= 0)
            n_ind += bool(lo <= 0)
        rows[n] = r
    am = np.argmax(BM, 1)
    per_set[s] = {
        "n_images": int(len(X)), "n_units": int(len(set(D["units"]))),
        "n_empty": int(D["empty"].sum()), "models": rows, "top": M[top],
        "n_indistinguishable_from_top": int(n_ind),
        "share_draws_top_is_top": r4(np.mean(am == top)),
        "share_draws_top_by_model": {M[j]: r4(np.mean(am == j)) for j in range(len(M)) if np.any(am == j)},
        "best_minus_median": r4(mu[top] - np.median(mu)),
        "best_minus_worst": r4(mu[top] - mu.min()), "worst": M[int(np.argmin(mu))]}
res["per_set"] = per_set

# ---------------------------------------------------------------- 3
def loss_of(choice, b):
    v = means[b]
    return max(v.values()) - v[choice]


chosen = {a: max(means[a], key=means[a].get) for a in SETS}
mat = {a: {b: r4(loss_of(chosen[a], b)) for b in WOUND} for a in WOUND}
off = [loss_of(chosen[a], b) for a in WOUND for b in WOUND if a != b]
rand = {b: r4(np.mean([max(means[b].values()) - means[b][n] for n in M])) for b in WOUND}
synth = {}
for amp in (11, 40):
    sm = {}
    for n in M:
        vals = []
        for seed in (2024, 7, 101):
            for d in ("pretrained", "nnunet"):
                f = os.path.join(ROOT, "results", "kaggle", "scores", d, f"genB_amp{amp}_s{seed}.json")
                g = json.load(open(f))
                if n in g["names"]:
                    vals.append(float(np.mean(np.asarray(g["dice"], float)[:, g["names"].index(n)])))
        assert len(vals) == 3, (n, amp)
        sm[n] = float(np.mean(vals))
    ch = max(sm, key=sm.get)
    rk = [sm[n] for n in M]
    synth[f"amp{amp}"] = {
        "mean_dice": {n: r4(v) for n, v in sm.items()}, "chosen": ch,
        "loss": {b: r4(loss_of(ch, b)) for b in WOUND},
        "mean_loss": r4(np.mean([loss_of(ch, b) for b in WOUND])),
        "spearman_with_wound": {b: r4(spearmanr(rk, [means[b][n] for n in M])[0]) for b in WOUND}}
gap = {n: max(max(means[b].values()) - means[b][n] for b in WOUND) for n in M}
gap_where = {n: max(WOUND, key=lambda b: max(means[b].values()) - means[b][n]) for n in M}
res["selection"] = {
    "chosen_on": chosen, "matrix_rows_choose_on_cols_deploy_on": mat,
    "offdiag_mean": r4(np.mean(off)), "offdiag_min": r4(min(off)), "offdiag_max": r4(max(off)),
    "random_choice_expected_loss": rand, "random_mean_over_B": r4(np.mean(list(rand.values()))),
    "control_chooser_loss": {b: r4(loss_of(chosen[CTRL], b)) for b in WOUND},
    "synthetic": synth,
    "max_gap_from_best": {n: r4(v) for n, v in gap.items()},
    "max_gap_where": gap_where,
    "minimax_model": min(gap, key=gap.get), "minimax_gap": r4(min(gap.values()))}

# ---------------------------------------------------------------- 4
def miss_stats(s, names, t):
    D = DATA[s]
    X = sub(s, names)
    ne = ~D["empty"]
    miss = ne[:, None] & (X < t)
    det = ne[:, None] & (X >= t)
    mr = miss.sum(0) / ne.sum()
    dwd = np.array([X[det[:, j], j].mean() if det[:, j].any() else np.nan for j in range(len(names))])
    return mr, dwd


def mean_pair_rho(vals):
    rhos = {}
    for a, b in combinations(WOUND, 2):
        x, y = vals[a], vals[b]
        rho = np.nan if (np.ptp(x) == 0 or np.ptp(y) == 0) else spearmanr(x, y)[0]
        rhos[f"{a}|{b}"] = r4(rho)
    v = [x for x in rhos.values() if x is not None]
    return {"pairs": rhos, "mean": r4(np.mean(v)) if v else None, "n_defined": len(v)}


miss = {}
for t in THRESH:
    tab, mr_all, dwd_all = {}, {}, {}
    for s in SETS:
        mr, dwd = miss_stats(s, M, t)
        mr_all[s], dwd_all[s] = mr, dwd
        tab[s] = {n: {"miss_rate": r4(mr[j]), "dice_when_detected": r4(dwd[j])} for j, n in enumerate(M)}
        tab[s]["_pretrained10_mean_miss"] = r4(mr[:10].mean())
        tab[s]["_nnunet_miss"] = r4(mr[10])
    groups = {}
    for gk in ("U", "F", "K", "M", "S"):
        groups[gk] = {s: r4(miss_stats(s, P.GROUPS[gk], t)[0].mean()) for s in SETS}
    miss[str(t)] = {
        "table": tab, "group_mean_miss_rate": groups,
        "spearman_miss_rate": mean_pair_rho(mr_all),
        "spearman_dice_when_detected": mean_pair_rho(dwd_all)}
res["miss"] = miss
res["spearman_mean_dice"] = mean_pair_rho({s: np.array([means[s][n] for n in M]) for s in WOUND})
res["spearman_mean_dice_nonempty"] = mean_pair_rho(
    {s: sub(s, M)[~DATA[s]["empty"]].mean(0) for s in WOUND})

# ---------------------------------------------------------------- 5
def img_size(s):
    shapes = {cv2.imread(f).shape[:2] for f in sorted(glob.glob(os.path.join(KIT, "refs", s, "*_img.png")))}
    return sorted(shapes)


azh = {"image_sizes": {s: [list(x) for x in img_size(s)] for s in ("azh", "fuseg", CTRL)},
       "model_input": "pretrained: letterbox to 256 px long side (kit_run.py PT_SIZE); "
                      "nnU-Net: native pixels, spacing 1, patch 512x384 (nnunet_stdout.txt)"}
edges = [0, 0.01, 0.03, 0.10, 0.30, 1.0001]
for s in ("azh", "fuseg", CTRL):
    D = DATA[s]
    ne = ~D["empty"]
    X = sub(s, M)
    f = D["frac"][ne]
    side = img_size(s)[0][0] if len(img_size(s)) == 1 else None
    mp = (X[ne][:, :10] < 0.1).mean(1)          # share of the 10 pretrained models missing
    mn = (X[ne][:, 10] < 0.1).astype(float)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (f >= lo) & (f < hi)
        bins.append({"frac_bin": [lo, round(min(hi, 1.0), 2)], "n": int(k.sum()),
                     "pretrained_miss": r4(mp[k].mean()) if k.any() else None,
                     "nnunet_miss": r4(mn[k].mean()) if k.any() else None})
    emp = D["empty"]
    azh[s] = {
        "n_nonempty": int(ne.sum()), "n_empty": int(emp.sum()),
        "frac_quartiles_nonempty": [r4(q) for q in np.percentile(f, [25, 50, 75])],
        "native_lesion_px_median": r4(np.median(f) * side * side) if side else None,
        "pretrained_input_lesion_px_median": r4(np.median(f) * 256 * 256),
        "pretrained10_miss_rate_mean": r4(mp.mean()),
        "pretrained10_miss_rate_range": [r4(x) for x in
                                         ((X[ne][:, :10] < 0.1).mean(0).min(), (X[ne][:, :10] < 0.1).mean(0).max())],
        "nnunet_miss_rate": r4(mn.mean()),
        "miss_by_frac": bins,
        "median_frac_missed_by_nnunet": r4(np.median(f[mn == 1])) if mn.any() else None,
        "median_frac_detected_by_nnunet": r4(np.median(f[mn == 0])),
        "pretrained_mean_dice_nonempty": r4(X[ne][:, :10].mean()),
        "nnunet_mean_dice_nonempty": r4(X[ne][:, 10].mean()),
        "empty_ref_pred_nonempty_pretrained_mean": r4((X[emp][:, :10] == 0).mean()) if emp.any() else None,
        "empty_ref_pred_nonempty_nnunet": r4((X[emp][:, 10] == 0).mean()) if emp.any() else None,
        "pretrained10_mean_dice": r4(X[:, :10].mean()), "nnunet_mean_dice": r4(X[:, 10].mean())}
    # images missed by nnU-Net but detected by most pretrained models
    azh[s]["n_nnunet_miss_pretrained_majority_detect"] = int(((mn == 1) & (mp < 0.5)).sum())
    azh[s]["n_pretrained_majority_miss_nnunet_detect"] = int(((mn == 0) & (mp >= 0.5)).sum())


def sift_scales():
    """Scale of AZH patches relative to FUSeg frames, from the control-set
    images that exclusions_v5.json records as SIFT-matched crops of FUSeg.
    Similarity transform AZH -> FUSeg by RANSAC; scale > 1 means one AZH pixel
    covers more than one FUSeg pixel (AZH patch downsampled)."""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    out = []
    for idx, name, why in P.EXCL.get(CTRL, []):
        parts = why.split()
        split, fname = parts[3], parts[4]
        a = cv2.imread(os.path.join(RAW, "wound_dataset", "train", "images", name), 0)
        f = cv2.imread(os.path.join(RAW, "Foot Ulcer Segmentation Challenge", split, "images", fname), 0)
        if a is None or f is None:
            continue
        ka, da = sift.detectAndCompute(a, None)
        kf, df = sift.detectAndCompute(f, None)
        if da is None or df is None:
            continue
        good = [m for m, n in bf.knnMatch(da, df, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            continue
        pa = np.float32([ka[m.queryIdx].pt for m in good])
        pf = np.float32([kf[m.trainIdx].pt for m in good])
        T, inl = cv2.estimateAffinePartial2D(pa, pf, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if T is None or inl.sum() < 8:
            continue
        out.append({"azh": name, "fuseg": f"{split}/{fname}", "inliers": int(inl.sum()),
                    "scale": r4(np.hypot(T[0, 0], T[1, 0])), "fuseg_shape": list(f.shape)})
    return out


def black_share(s):
    """Per kept image: share of pixels with every channel <= 8 (zero padding).
    Kit ref files are in scoring order; exclusions are dropped as in load()."""
    files = sorted(glob.glob(os.path.join(KIT, "refs", s, "*_img.png")))
    drop = {e[0] for e in P.EXCL.get(s, [])}
    v = [float((cv2.imread(f).max(2) <= 8).mean()) for i, f in enumerate(files) if i not in drop]
    assert len(v) == len(DATA[s]["ids"]), s
    return np.array(v)


for s in ("azh", "fuseg", CTRL):
    D = DATA[s]
    ne = ~D["empty"]
    bs = black_share(s)[ne]
    X = sub(s, M)[ne]
    maj = (X[:, :10] < 0.1).mean(1) >= 0.5
    rho = spearmanr(bs, (X[:, :10] < 0.1).mean(1))[0]
    azh[s]["black_pixel_share"] = {
        "mean_all_nonempty": r4(bs.mean()),
        "share_images_with_any_padding_gt_1pct": r4((bs > 0.01).mean()),
        "mean_pretrained_majority_missed": r4(bs[maj].mean()) if maj.any() else None,
        "mean_other": r4(bs[~maj].mean()),
        "spearman_with_pretrained_miss_share": r4(rho)}
    f = D["frac"][ne]
    ms = (X[:, :10] < 0.1).mean(1)
    side = 224 if s != "fuseg" else 512
    azh[s]["frac_spearman_with_pretrained_miss_share"] = r4(spearmanr(f, ms)[0])
    azh[s]["frac_spearman_with_black_share"] = r4(spearmanr(f, bs)[0])
    azh[s]["content_px_median"] = r4(np.median(1 - bs) * side * side)
    azh[s]["lesion_share_of_content_spearman_with_pretrained_miss"] = r4(
        spearmanr(f / np.clip(1 - bs, 1e-6, None), ms)[0])
    q = np.quantile(f, [1 / 3, 2 / 3])
    b = np.digitize(f, q)
    strat = []
    for k in range(3):
        m = b == k
        hi = bs[m] > np.median(bs[m])
        strat.append({"frac_tercile": k, "n": int(m.sum()),
                      "pretrained_miss_more_padding": r4(ms[m][hi].mean()),
                      "pretrained_miss_less_padding": r4(ms[m][~hi].mean()),
                      "nnunet_miss_more_padding": r4((X[m][hi, 10] < 0.1).mean()),
                      "nnunet_miss_less_padding": r4((X[m][~hi, 10] < 0.1).mean())})
    azh[s]["miss_by_frac_tercile_and_padding"] = strat


sc = sift_scales()
v = np.array([r["scale"] for r in sc])
azh["sift_scale_azh_to_fuseg"] = {
    "n": len(sc), "median": r4(np.median(v)) if len(v) else None,
    "quartiles": [r4(q) for q in np.percentile(v, [25, 75])] if len(v) else None,
    "min": r4(v.min()) if len(v) else None, "max": r4(v.max()) if len(v) else None, "rows": sc}
res["azh_explanation"] = azh

# ---------------------------------------------------------------- 6
S = P.GROUPS["S"]
sam = {}
for s in P.REFS + [CTRL]:
    X = sub(s, S)
    ne = ~DATA[s]["empty"]
    mr, dwd = miss_stats(s, S, 0.1)
    sam[s] = {n: {"mean_dice": r4(X[:, j].mean()), "mean_dice_nonempty": r4(X[ne, j].mean()),
                  "miss_rate": r4(mr[j]), "dice_when_detected": r4(dwd[j])} for j, n in enumerate(S)}
    sam[s]["best_M_mean"] = r4(max(means[s].values())) if s in means else None
res["sam"] = {"note": "box prompts are tight ground-truth component boxes (up to 10 per image); "
                      "empty references get no box and an empty prediction, hence Dice 1",
              "scores": sam}

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "practical_v6.json"), "w"), indent=1)
print("written", os.path.join(OUT, "practical_v6.json"))

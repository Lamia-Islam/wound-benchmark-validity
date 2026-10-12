#!/usr/bin/env python3
"""V7 practical analysis: port of scripts/v6/practical_v6.py to perimage_v7,
extended to the 19 modern methods.

Definitions (unchanged from V6; per-image clinical Dice from perimage_v7.load,
which applies the V6 exclusions and returns the V6 columns unchanged)
- Per-image Dice on an empty reference mask is 1 when the prediction is also
  empty and 0 otherwise.
- Mean Dice: mean over ALL kept images, empty references included.
- Unit bootstrap: resample resampling units (photograph for Medetec and AZH,
  image otherwise) with replacement, B = 2000 draws; every image of a drawn
  unit enters with the unit's draw count as weight. Same draws for all models
  of one dataset (paired). numpy default_rng([20261010, k]) with k the index of
  the dataset in (medetec, co2wounds, fuseg, azh, azh_train_slice): exactly the
  V6 draws. Intervals are 2.5 and 97.5 percentiles.
- Top model: highest point-estimate mean Dice. "Indistinguishable from the top":
  the 95% paired interval of (top minus model) has lower bound <= 0.
- Selection loss A -> B: best mean Dice on B minus mean Dice on B of the model
  with the highest mean Dice on A. Random choice: mean over the candidates of
  (best on B minus candidate on B).
- Synthetic chooser: generator B at amplitude 11, 28 or 40; per synthetic seed
  the mean Dice over all 45 images, then the mean over seeds 2024, 7, 101.
  Seed-averaged candidates ('<arch>@seeds') use the per-image mean over their
  training seeds, so their synthetic mean is the mean of the per-seed means.
- Within d of the best: loss <= d (d = 0.01, 0.02), unrounded values.
- Miss: reference non-empty and Dice < t (t = 0.1 main; 0.05, 0.2
  sensitivity). Miss rate = misses / non-empty images. Computed per
  individual trained model (a seed-averaged per-image Dice is not a model and
  is not used for misses). Per architecture: mean, min and max over its
  available training seeds; interval of the seed mean from the same unit
  draws (miss rate in a draw = weighted misses / weighted non-empty images).
- Spearman: scipy.stats.spearmanr (average ranks) across candidates between
  two wound datasets; mean over the 6 pairs.

Candidate sets: M (V6: 10 pretrained seed 0 + nnU-Net), M11s (10 original
architectures seed-averaged + nnU-Net), M19 (M11s + 8 new architectures
seed-averaged), N8s, and single-seed sets M19_s0, M19_s1, M19_s2 (every
architecture that has that training seed, as the single model, + nnU-Net).

Output: results/v7/practical_v7.json (written by this script) and
results/v7/PRACTICAL_V7.md (written by scripts/v7/write_md_v7.py).
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
import perimage_v7 as P                                           # noqa: E402

SEED = 20261010
B = 2000
THRESH = (0.05, 0.1, 0.2)
WOUND = list(P.WOUND)
CTRL = P.CONTROL
SETS = WOUND + [CTRL]
SYN_SEEDS = (2024, 7, 101)
KIT = "/home/claude/kaggle_kit"
RAW = os.path.join(ROOT, "data", "wound-segmentation", "data")
OUT = os.path.join(ROOT, "results", "v7")
V6JSON = os.path.join(ROOT, "results", "v6", "practical_v6.json")
NN = "nnunet_2d"

G = {k: list(v) for k, v in P.GROUPS.items()}
for k in (0, 1, 2):
    G[f"M19_s{k}"] = G[f"P10_s{k}"] + G[f"N8_s{k}"] + [NN]


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def boot_weights(units, rng):
    u, inv = np.unique(units, return_inverse=True)
    draws = rng.integers(0, len(u), size=(B, len(u)))
    counts = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    return counts[:, inv]                                           # B x images


DATA = {s: P.load(s) for s in P.REFS + [CTRL]}
W = {s: boot_weights(DATA[s]["units"], np.random.default_rng([SEED, k])) for k, s in enumerate(SETS)}
SYN = {}


def sub(s, names):
    return DATA[s]["X"][:, P.cols(names)]


def syn_mean(n, amp, seed):
    s = f"genB_amp{amp}_s{seed}"
    if s not in SYN:
        SYN[s] = P.load_synth(s)
    j = SYN[s]["names"].index(n)
    return float(np.mean(SYN[s]["X"][:, j]))


# ======================================================================
# Part A: the V6 script, ported (loader replaced; logic unchanged)
# ======================================================================

def v6_port():
    M = list(G["M"])
    PT = [n for n in M if n != NN]
    assert len(M) == 11 and len(PT) == 10
    res = {"seed": SEED, "B": B, "models": M}
    per_set, means = {}, {}
    for k, s in enumerate(SETS):
        D = DATA[s]
        X = sub(s, M)
        rng = np.random.default_rng([SEED, k])
        Wk = boot_weights(D["units"], rng)
        BM = (Wk @ X) / Wk.sum(1, keepdims=True)
        mu = X.mean(0)
        means[s] = dict(zip(M, mu))
        top = int(np.argmax(mu))
        rows, n_ind = {}, 0
        for j, n in enumerate(M):
            r = {"mean": r4(mu[j]), "ci": [r4(np.percentile(BM[:, j], 2.5)), r4(np.percentile(BM[:, j], 97.5))]}
            if j != top:
                d = BM[:, top] - BM[:, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                r["diff_from_top"] = r4(mu[top] - mu[j]); r["diff_ci"] = [r4(lo), r4(hi)]
                r["indistinguishable"] = bool(lo <= 0); n_ind += bool(lo <= 0)
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

    def loss_of(choice, b):
        v = means[b]
        return max(v.values()) - v[choice]

    chosen = {a: max(means[a], key=means[a].get) for a in SETS}
    mat = {a: {b: r4(loss_of(chosen[a], b)) for b in WOUND} for a in WOUND}
    off = [loss_of(chosen[a], b) for a in WOUND for b in WOUND if a != b]
    rand = {b: r4(np.mean([max(means[b].values()) - means[b][n] for n in M])) for b in WOUND}
    synth = {}
    for amp in (11, 40):
        sm = {n: float(np.mean([syn_mean(n, amp, sd) for sd in SYN_SEEDS])) for n in M}
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
        groups = {gk: {s: r4(miss_stats(s, G[gk], t)[0].mean()) for s in SETS} for gk in ("U", "F", "K", "M", "S")}
        miss[str(t)] = {"table": tab, "group_mean_miss_rate": groups,
                        "spearman_miss_rate": mean_pair_rho(mr_all),
                        "spearman_dice_when_detected": mean_pair_rho(dwd_all)}
    res["miss"] = miss
    res["spearman_mean_dice"] = mean_pair_rho({s: np.array([means[s][n] for n in M]) for s in WOUND})
    res["spearman_mean_dice_nonempty"] = mean_pair_rho({s: sub(s, M)[~DATA[s]["empty"]].mean(0) for s in WOUND})

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
        mp = (X[ne][:, :10] < 0.1).mean(1)
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
        azh[s]["n_nnunet_miss_pretrained_majority_detect"] = int(((mn == 1) & (mp < 0.5)).sum())
        azh[s]["n_pretrained_majority_miss_nnunet_detect"] = int(((mn == 0) & (mp >= 0.5)).sum())

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

    S = G["S"]
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
    return res


_IMG = {}


def img_size(s):
    if s not in _IMG:
        _IMG[s] = sorted({cv2.imread(f).shape[:2] for f in sorted(glob.glob(os.path.join(KIT, "refs", s, "*_img.png")))})
    return _IMG[s]


_BS = {}


def black_share(s):
    """Per kept image: share of pixels with every channel <= 8."""
    if s not in _BS:
        files = sorted(glob.glob(os.path.join(KIT, "refs", s, "*_img.png")))
        drop = {e[0] for e in P.EXCL.get(s, [])}
        v = [float((cv2.imread(f).max(2) <= 8).mean()) for i, f in enumerate(files) if i not in drop]
        assert len(v) == len(DATA[s]["ids"]), s
        _BS[s] = np.array(v)
    return _BS[s]


def sift_scales():
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


def rr(x):
    """Round like V6's r4, recursively."""
    if isinstance(x, dict):
        return {k: rr(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rr(v) for v in x]
    return r4(x) if isinstance(x, float) else x


def P6_top(R6, s):
    return R6["per_set"][s]["top"]


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


def compare(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    common = set(la) & set(lb)
    bad = sorted(k for k in common if la[k] != lb[k])
    return {"n_leaves_v6": len(lb), "n_leaves_rerun": len(la), "n_common": len(common),
            "n_equal": len(common) - len(bad), "mismatched": bad[:50],
            "missing_in_rerun": sorted(set(lb) - set(la))[:50],
            "extra_in_rerun": sorted(set(la) - set(lb))[:50]}


# ======================================================================
# Part B: V7 extensions
# ======================================================================


def rf(x):
    """Part B stores unrounded values (V6 rounded to 4 decimals with r4; rounding
    again for the report would round twice)."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

def spread(C):
    out = {}
    for s in SETS:
        X = sub(s, C)
        BM = (W[s] @ X) / W[s].sum(1, keepdims=True)
        mu = X.mean(0)
        top = int(np.argmax(mu))
        ind = []
        for j in range(len(C)):
            if j != top and np.percentile(BM[:, top] - BM[:, j], 2.5) <= 0:
                ind.append(C[j])
        am = np.argmax(BM, 1)
        out[s] = {"n_candidates": len(C), "n_images": int(len(X)), "n_units": int(len(set(DATA[s]["units"]))),
                  "top": C[top], "top_mean": rf(mu[top]),
                  "top_ci": [rf(np.percentile(BM[:, top], 2.5)), rf(np.percentile(BM[:, top], 97.5))],
                  "indistinguishable_from_top": ind, "n_indistinguishable_from_top": len(ind),
                  "share_draws_top_is_top": rf(np.mean(am == top)),
                  "best_minus_median": rf(mu[top] - np.median(mu)),
                  "best_minus_worst": rf(mu[top] - mu.min()), "worst": C[int(np.argmin(mu))],
                  "means": {n: rf(v) for n, v in zip(C, mu)}}
    return out


def selection(C):
    means = {s: dict(zip(C, sub(s, C).mean(0))) for s in SETS}
    best = {b: max(means[b].values()) for b in SETS}

    def loss_of(choice, b):
        return best[b] - means[b][choice]

    chosen = {a: max(means[a], key=means[a].get) for a in SETS}
    mat = {a: {b: rf(loss_of(chosen[a], b)) for b in WOUND} for a in WOUND}
    off = [loss_of(chosen[a], b) for a in WOUND for b in WOUND if a != b]
    rand = {b: float(np.mean([best[b] - means[b][n] for n in C])) for b in WOUND}
    within = {}
    for d in (0.01, 0.02):
        within[str(d)] = {
            "other_wound_chooser": f"{sum(loss_of(chosen[a], b) <= d for a in WOUND for b in WOUND if a != b)}/12",
            "random_share_of_candidates_per_dataset": {b: rf(np.mean([best[b] - means[b][n] <= d for n in C])) for b in WOUND},
            "random_share_mean": rf(np.mean([np.mean([best[b] - means[b][n] <= d for n in C]) for b in WOUND])),
            "candidates_within_per_dataset": {b: [n for n in C if best[b] - means[b][n] <= d] for b in WOUND}}
    synth = {}
    for amp in (11, 28, 40):
        per_seed = {sd: {n: syn_mean(n, amp, sd) for n in C} for sd in SYN_SEEDS}
        sm = {n: float(np.mean([per_seed[sd][n] for sd in SYN_SEEDS])) for n in C}
        ch = max(sm, key=sm.get)
        srt = sorted(sm, key=sm.get, reverse=True)
        losses = {b: loss_of(ch, b) for b in WOUND}
        synth[f"amp{amp}"] = {
            "mean_dice": {n: rf(v) for n, v in sm.items()}, "chosen": ch, "chosen_synth_dice": rf(sm[ch]),
            "runner_up": srt[1], "runner_up_synth_dice": rf(sm[srt[1]]),
            "loss": {b: rf(v) for b, v in losses.items()}, "mean_loss": rf(np.mean(list(losses.values()))),
            "within_0.01": {b: bool(v <= 0.01) for b, v in losses.items()},
            "within_0.02": {b: bool(v <= 0.02) for b, v in losses.items()},
            "n_within_0.01": int(sum(v <= 0.01 for v in losses.values())),
            "n_within_0.02": int(sum(v <= 0.02 for v in losses.values())),
            "chosen_per_synth_seed": {str(sd): max(per_seed[sd], key=per_seed[sd].get) for sd in SYN_SEEDS},
            "loss_per_synth_seed_pick": {str(sd): {b: rf(loss_of(max(per_seed[sd], key=per_seed[sd].get), b)) for b in WOUND}
                                         for sd in SYN_SEEDS},
            "spearman_with_wound": {b: rf(spearmanr([sm[n] for n in C], [means[b][n] for n in C])[0]) for b in WOUND}}
    gap = {n: max(best[b] - means[b][n] for b in WOUND) for n in C}
    return {
        "n_candidates": len(C), "candidates": C,
        "best_per_dataset": {b: {"model": chosen[b], "mean_dice": rf(best[b])} for b in SETS},
        "chosen_on": chosen, "matrix_rows_choose_on_cols_deploy_on": mat,
        "offdiag_mean": rf(np.mean(off)), "offdiag_min": rf(min(off)), "offdiag_max": rf(max(off)),
        "random_choice_expected_loss": {b: rf(v) for b, v in rand.items()},
        "random_mean_over_B": rf(np.mean(list(rand.values()))),
        "control_chooser": chosen[CTRL],
        "control_chooser_loss": {b: rf(loss_of(chosen[CTRL], b)) for b in WOUND},
        "within": within, "synthetic": synth,
        "max_gap_from_best": {n: rf(v) for n, v in gap.items()},
        "minimax_model": min(gap, key=gap.get), "minimax_gap": rf(min(gap.values())),
        "spearman_mean_dice_6pairs": rf(np.mean([spearmanr([means[a][n] for n in C], [means[b][n] for n in C])[0]
                                                 for a, b in combinations(WOUND, 2)]))}


def individual_models():
    """Architecture -> {seed: model column}; nnU-Net is its own architecture."""
    arch = {}
    for col, mem in P.SEEDCOLS.items():
        base = col.split("@")[0]
        d = {}
        for m in mem:
            if m.endswith("_in"):
                d[0] = m
            else:
                d[int(m.rsplit("_s", 1)[1])] = m
        arch[base] = d
    arch[NN] = {0: NN}
    return arch


ARCH = individual_models()
ORIG = [c.split("@")[0] for c in G["P10s"]]
NEW = [c.split("@")[0] for c in G["N8s"]]


def miss_section():
    out = {"n_individual_models": sum(len(v) for v in ARCH.values()),
           "architectures": {a: {str(k): m for k, m in v.items()} for a, v in ARCH.items()}}
    for t in THRESH:
        tt = {}
        for s in SETS:
            D = DATA[s]
            ne = ~D["empty"]
            names = [m for a in ARCH for m in ARCH[a].values()]
            X = sub(s, names)
            Mi = (ne[:, None] & (X < t)).astype(float)             # images x models
            mr = Mi.sum(0) / ne.sum()
            Wn = W[s] @ ne.astype(float)                            # B
            BMR = (W[s] @ Mi) / Wn[:, None]                         # B x models
            col = {n: j for j, n in enumerate(names)}
            per_model = {n: {"miss_rate": rf(mr[j]),
                             "ci": [rf(np.percentile(BMR[:, j], 2.5)), rf(np.percentile(BMR[:, j], 97.5))]}
                         for n, j in col.items()}
            per_arch, arch_draw = {}, {}
            for a, sd in ARCH.items():
                js = [col[m] for m in sd.values()]
                v = mr[js]
                dr = BMR[:, js].mean(1)
                arch_draw[a] = dr
                per_arch[a] = {"n_seeds": len(js), "seeds": sorted(sd), "mean": rf(v.mean()),
                               "min": rf(v.min()), "max": rf(v.max()),
                               "ci_of_seed_mean": [rf(np.percentile(dr, 2.5)), rf(np.percentile(dr, 97.5))],
                               "per_seed": {str(k): rf(mr[col[m]]) for k, m in sd.items()}}
            grp = {}
            defs = {"P10_archmean": ORIG, "N8_archmean": NEW, "P18_archmean": ORIG + NEW}
            for gname, archs in defs.items():
                est = float(np.mean([mr[[col[m] for m in ARCH[a].values()]].mean() for a in archs]))
                dr = np.mean([arch_draw[a] for a in archs], 0)
                diff = dr - arch_draw[NN]
                grp[gname] = {"n_architectures": len(archs), "est": rf(est),
                              "ci": [rf(np.percentile(dr, 2.5)), rf(np.percentile(dr, 97.5))],
                              "minus_nnunet": rf(est - mr[col[NN]]),
                              "minus_nnunet_ci": [rf(np.percentile(diff, 2.5)), rf(np.percentile(diff, 97.5))],
                              "arch_mean_range": [rf(min(per_arch[a]["mean"] for a in archs)),
                                                  rf(max(per_arch[a]["mean"] for a in archs))],
                              "single_model_range": [rf(min(mr[col[m]] for a in archs for m in ARCH[a].values())),
                                                     rf(max(mr[col[m]] for a in archs for m in ARCH[a].values()))]}
            for k in (0, 1, 2):
                for gname, base in (("P10", f"P10_s{k}"), ("N8", f"N8_s{k}")):
                    js = [col[m] for m in G[base]]
                    dr = BMR[:, js].mean(1)
                    grp[f"{base}_modelmean"] = {"n_models": len(js), "est": rf(mr[js].mean()),
                                                "ci": [rf(np.percentile(dr, 2.5)), rf(np.percentile(dr, 97.5))],
                                                "range": [rf(mr[js].min()), rf(mr[js].max())]}
            tt[s] = {"n_nonempty": int(ne.sum()), "n_units_nonempty": int(len(set(D["units"][ne]))),
                     "n_units": int(len(set(D["units"]))),
                     "per_model": per_model, "per_architecture": per_arch, "groups": grp}
        out[str(t)] = tt
    return out


def padding_section():
    """V6 section 5 padding analysis repeated with every individual pretrained
    model of the 18 architectures (51 models), and separately for the
    original ten (29 models) and the new eight (22 models)."""
    out = {}
    sets = {"P10_all_seeds": [m for a in ORIG for m in ARCH[a].values()],
            "N8_all_seeds": [m for a in NEW for m in ARCH[a].values()]}
    sets["P18_all_seeds"] = sets["P10_all_seeds"] + sets["N8_all_seeds"]
    for s in WOUND + [CTRL]:
        D = DATA[s]
        ne = ~D["empty"]
        bs_all = black_share(s)
        bs = bs_all[ne]
        f = D["frac"][ne]
        mn = (sub(s, [NN])[ne][:, 0] < 0.1)
        o = {"black_share_mean_all_kept": rf(bs_all.mean()), "black_share_mean_nonempty": rf(bs.mean()),
             "n_kept": int(len(bs_all)), "n_nonempty": int(ne.sum())}
        for gname, names in sets.items():
            ms = (sub(s, names)[ne] < 0.1).mean(1)
            maj = ms >= 0.5
            g = {"n_models": len(names), "mean_miss_share": rf(ms.mean()),
                 "n_majority_miss_nnunet_detect": int((maj & ~mn).sum()),
                 "n_nnunet_miss_majority_detect": int((mn & ~maj).sum())}
            if s in ("azh", "fuseg", CTRL):
                g["spearman_black_share_vs_miss_share"] = rf(spearmanr(bs, ms)[0])
                g["spearman_frac_vs_miss_share"] = rf(spearmanr(f, ms)[0])
                q = np.quantile(f, [1 / 3, 2 / 3])
                b = np.digitize(f, q)
                strat = []
                for k in range(3):
                    m = b == k
                    hi = bs[m] > np.median(bs[m])
                    strat.append({"frac_tercile": k, "n": int(m.sum()),
                                  "miss_more_padding": rf(ms[m][hi].mean()),
                                  "miss_less_padding": rf(ms[m][~hi].mean())})
                g["miss_by_frac_tercile_and_padding"] = strat
            o[gname] = g
        out[s] = o
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"definitions": __doc__, "seed": SEED, "B": B}

    # ---------------- reproduction of V6
    R6 = json.loads(json.dumps(v6_port()))
    V6 = json.load(open(V6JSON))
    V6c = {k: v for k, v in V6.items() if k != "definitions"}
    cmpr = compare(R6, V6c)
    res["v6_reproduction"] = cmpr
    assert not cmpr["mismatched"] and not cmpr["missing_in_rerun"] and not cmpr["extra_in_rerun"], cmpr
    print("practical V6 reproduction:", cmpr["n_equal"], "of", cmpr["n_common"], "leaves identical", flush=True)
    res["v6"] = R6

    # ---------------- V7: spread and selection
    cand = {"M": G["M"], "M11s": G["M11s"], "M19": G["M19"], "N8s": G["N8s"], "P10s": G["P10s"],
            "M19_s0": G["M19_s0"], "M19_s1": G["M19_s1"], "M19_s2": G["M19_s2"]}
    res["candidate_sets"] = cand
    res["spread"] = {k: spread(v) for k, v in cand.items()}
    res["selection"] = {k: selection(v) for k, v in cand.items()}
    # consistency: selection for M equals the V6 selection numbers (V6 stored
    # 4-decimal values; V7 part B stores unrounded values)
    s6, s7 = R6["selection"], res["selection"]["M"]
    assert s6["matrix_rows_choose_on_cols_deploy_on"] == rr(s7["matrix_rows_choose_on_cols_deploy_on"])
    assert s6["offdiag_mean"] == rr(s7["offdiag_mean"])
    assert s6["random_choice_expected_loss"] == rr(s7["random_choice_expected_loss"])
    assert s6["max_gap_from_best"] == rr(s7["max_gap_from_best"]) and s6["minimax_model"] == s7["minimax_model"]
    print("random mean V6 (mean of rounded) vs V7 (unrounded):", s6["random_mean_over_B"], s7["random_mean_over_B"])
    for a in ("amp11", "amp40"):
        assert s6["synthetic"][a]["loss"] == rr(s7["synthetic"][a]["loss"])
        assert s6["synthetic"][a]["mean_dice"] == rr(s7["synthetic"][a]["mean_dice"])
        assert s6["synthetic"][a]["mean_loss"] == rr(s7["synthetic"][a]["mean_loss"])
        assert s6["synthetic"][a]["chosen"] == s7["synthetic"][a]["chosen"]
    for s in SETS:
        assert P6_top(R6, s) == res["spread"]["M"][s]["top"]
        assert R6["per_set"][s]["n_indistinguishable_from_top"] == res["spread"]["M"][s]["n_indistinguishable_from_top"]

    # synthetic pick across training seeds (M19_s0..2 vs seed-averaged M19)
    pk = {}
    for amp in ("amp11", "amp28", "amp40"):
        pk[amp] = {k: {"chosen": res["selection"][k]["synthetic"][amp]["chosen"],
                       "mean_loss": res["selection"][k]["synthetic"][amp]["mean_loss"],
                       "n_within_0.01": res["selection"][k]["synthetic"][amp]["n_within_0.01"],
                       "n_within_0.02": res["selection"][k]["synthetic"][amp]["n_within_0.02"]}
                   for k in ("M19", "M19_s0", "M19_s1", "M19_s2", "M11s", "M")}
    res["synthetic_pick_by_training_seed"] = pk
    # clinical best across training seeds
    res["clinical_best_by_training_seed"] = {
        b: {k: res["selection"][k]["best_per_dataset"][b] for k in ("M19", "M19_s0", "M19_s1", "M19_s2")}
        for b in SETS}

    # synthetic mean Dice of each individual model, and its spread over training seeds
    ssr = {}
    for amp in (11, 28, 40):
        ssr[f"amp{amp}"] = {}
        for a, sd in ARCH.items():
            v = {str(k): float(np.mean([syn_mean(m, amp, x) for x in SYN_SEEDS])) for k, m in sd.items()}
            ssr[f"amp{amp}"][a] = {"per_seed": v, "range": float(max(v.values()) - min(v.values())), "n_seeds": len(v)}
    res["synthetic_dice_by_training_seed"] = ssr
    # same quantity for the seed-averaged column equals the mean of the per-seed values
    for a, sd in ARCH.items():
        if a == NN:
            continue
        col = [c for c in G["M19"] if c.split("@")[0] == a][0]
        assert abs(res["selection"]["M19"]["synthetic"]["amp40"]["mean_dice"][col]
                   - np.mean(list(ssr["amp40"][a]["per_seed"].values()))) < 1e-12, a

    # ---------------- misses
    res["miss"] = miss_section()
    # consistency: seed-0 P10 model mean and nnU-Net equal the V6 values
    for t in THRESH:
        for s in SETS:
            assert rr(res["miss"][str(t)][s]["groups"]["P10_s0_modelmean"]["est"]) == \
                R6["miss"][str(t)]["table"][s]["_pretrained10_mean_miss"], (t, s)
            assert rr(res["miss"][str(t)][s]["per_model"][NN]["miss_rate"]) == R6["miss"][str(t)]["table"][s]["_nnunet_miss"]
            for n in G["M"]:
                assert rr(res["miss"][str(t)][s]["per_model"][n]["miss_rate"]) == R6["miss"][str(t)]["table"][s][n]["miss_rate"]
    res["padding"] = padding_section()
    for s in ("azh", "fuseg", CTRL):
        assert rr(res["padding"][s]["black_share_mean_nonempty"]) == R6["azh_explanation"][s]["black_pixel_share"]["mean_all_nonempty"]

    # ---------------- cross-check with the earlier V7 implementation (independent code,
    # results/v7/practical_v7.json, 4-decimal values)
    xc = {}
    old = os.path.join(ROOT, "results", "v7_morning", "practical_v7.json")
    if os.path.exists(old):
        O = json.load(open(old))
        o, m = O["M19"]["selection"], res["selection"]["M19"]
        xc["M19_matrix"] = o["matrix_rows_choose_on_cols_deploy_on"] == rr(m["matrix_rows_choose_on_cols_deploy_on"])
        xc["M19_offdiag_random_minimax"] = [o["offdiag_mean"], o["random_mean_over_B"], o["minimax_gap"]] == \
            rr([m["offdiag_mean"], m["random_mean_over_B"], m["minimax_gap"]])
        for a in ("amp11", "amp40"):
            xc[f"M19_synthetic_{a}"] = (o["synthetic"][a]["chosen"] == m["synthetic"][a]["chosen"]
                                       and o["synthetic"][a]["loss"] == rr(m["synthetic"][a]["loss"]))
        for k in (0, 1, 2):
            ps, mk = O["seed_aware"]["per_seed_candidate_sets"][f"s{k}"], res["selection"][f"M19_s{k}"]
            xc[f"M19_s{k}"] = ([ps["offdiag_mean"], ps["random_mean_over_B"], ps["synthetic"]["amp40"]["mean_loss"]] ==
                               rr([mk["offdiag_mean"], mk["random_mean_over_B"], mk["synthetic"]["amp40"]["mean_loss"]])
                               and ps["synthetic"]["amp40"]["loss"] == rr(mk["synthetic"]["amp40"]["loss"]))
        for t in THRESH:
            for s in SETS:
                oi = O["miss_intervals"]["M19"][str(t)][s]
                gi = res["miss"][str(t)][s]["groups"]["P18_archmean"]
                xc[f"miss_P18_{t}_{s}"] = ([oi["pretrained_mean_miss"]] + oi["pretrained_ci"] + [oi["diff_pretrained_minus_nnunet"]] + oi["diff_ci"]
                                          == rr([gi["est"]] + gi["ci"] + [gi["minus_nnunet"]] + gi["minus_nnunet_ci"]))
        xc["_all_equal"] = all(xc.values())
    res["crosscheck_old_v7"] = xc
    print("cross-check with earlier V7 implementation:", xc.get("_all_equal"), [k for k, v in xc.items() if not v])

    json.dump(res, open(os.path.join(OUT, "practical_v7.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "practical_v7.json"))


if __name__ == "__main__":
    main()

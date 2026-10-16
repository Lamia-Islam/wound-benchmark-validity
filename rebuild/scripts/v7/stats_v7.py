#!/usr/bin/env python3
"""V7 agreement statistics (run 1 + run 2 models), written to
results/v7/stats_v7.json and results/v7/STATS_V7.md.

An earlier, independent V7 implementation (built on an older perimage_v7 API
with "@avg" columns) is kept in scripts/v7_morning and results/v7_morning and is
used only as a cross-check.

Port of scripts/v6/stats_v6.py. stats_v6.py computes and writes results/v6 at
import time, so it cannot be imported; its functions (zr, corrmat, nmean, summ,
ci, comp_masks, exact_test, v5_method_takes, gather) and sections 1 to 7 are
copied unchanged in substance and run over more method sets. Section 0 checks
that every numeric leaf of results/v6/stats_v6.json is reproduced (abs or rel
diff <= 1e-12) and stops otherwise.

Data: scripts/perimage_v7.py (V6 images, exclusions and resampling units).
Method groups: U 18, F 14, K 14, M 11 (V6), S (never pooled),
  M11s = 10 pretrained architectures seed-averaged + nnU-Net 2D,
  M19  = M11s + 8 new architectures seed-averaged,
  N8s  = 8 new architectures seed-averaged.
Method sets: V6 sets U, F, K, M, U+F, U+K, U+M, ALL57 and V7 sets M11s, M19,
N8s, U+M11s, U+M19, ALL (= U+F+K+M19, 65 methods).

Random procedures (all seeds fixed):
  split halves     500 unit splits, numpy default_rng(1) (V6 stream; same
                   splits for every method set)
  by-image split   default_rng(2) (V6 stream)
  hierarchical     default_rng(6060): unit resamples first (V6), then method
                   resamples pool by pool, V6 pools in V6 order (identical
                   draws), then the V7 pools in V7_BOOT order
  method-only      stats_v5 design: default_rng(0) per pool, B 1000 and 2000
  seed analysis    architecture resampling B 2000 default_rng(7070);
                   hierarchical (architectures + units) B 1000: architectures
                   default_rng(7071), units = the section-3 unit draws
Run from the rebuild folder:  python scripts/v7/stats_v7.py
"""
import itertools
import json
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr
from sympy.utilities.iterables import multiset_permutations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage_v7 as V                                            # noqa: E402

import excl_cfg as C                                               # noqa: E402
OUT = C.V7OUT
T0 = time.time()

N_SPLIT = 500
B_HIER = 1000
SEED_SPLIT = 1
SEED_HIER = 6060
B_SEEDBOOT = 2000
SEED_SEEDBOOT = 7070
SEED_SEEDHIER = 7071
METRICS = ("dice", "nsd")
TOL = 1e-12

REFS = list(V.REFS)
DS = REFS + [V.CONTROL]
NR = len(REFS)
ND = len(DS)
I_AZH, I_CTRL = DS.index("azh"), DS.index(V.CONTROL)
LAB = [V.MODALITY[r] for r in REFS]
SRC = [V.SOURCE[r] for r in REFS]

V6_SETS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
           "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"],
           "ALL57": ["U", "F", "K", "M"]}
V7_SETS = {"M11s": ["M11s"], "M19": ["M19"], "N8s": ["N8s"],
           "U+M11s": ["U", "M11s"], "U+M19": ["U", "M19"], "ALL": ["U", "F", "K", "M19"]}
GROUP_SETS = {**V6_SETS, **V7_SETS}
V5_POOLS = ["U", "F", "K", "M", "U+F", "U+K", "U+M"]
V7_BOOT = ["ALL57", "M11s", "M19", "N8s", "U+M11s", "U+M19", "ALL"]
GNAMES = ("U", "F", "K", "M", "S", "M11s", "M19", "N8s", "P10s")
GCOLS = {g: V.cols(V.GROUPS[g]) for g in GNAMES}
# V7 versions of the V6 groups are the same columns as in perimage_v6
import perimage_v6 as V6L                                          # noqa: E402
for g in ("U", "F", "K", "M", "S"):
    assert V.GROUPS[g] == V6L.GROUPS[g], g


def set_cols(s):
    return np.concatenate([GCOLS[g] for g in GROUP_SETS[s]])


# ------------------------------------------------------------------ pairs
PI, PJ = map(np.array, zip(*itertools.combinations(range(NR), 2)))     # 55 pairs
NP = len(PI)
PNAME = [f"{REFS[i]}~{REFS[j]}" for i, j in zip(PI, PJ)]
SAME = np.array([LAB[i] == LAB[j] for i, j in zip(PI, PJ)])
WSRC = np.array([SRC[i] == SRC[j] for i, j in zip(PI, PJ)])
ACROSS = ~SAME
DSSM = SAME & ~WSRC
WOUNDP = np.array([LAB[i] == LAB[j] == "wound" for i, j in zip(PI, PJ)])
COLONP = np.array([LAB[i] == LAB[j] == "colonoscopy" for i, j in zip(PI, PJ)])
WOUND_DS = WOUNDP & ~WSRC
FA = np.array([{REFS[i], REFS[j]} == {"fuseg", "azh"} for i, j in zip(PI, PJ)])
assert SAME.sum() == 16 and WSRC.sum() == 3 and DSSM.sum() == 13 and ACROSS.sum() == 39
assert WOUNDP.sum() == 6 and WOUND_DS.sum() == 5 and COLONP.sum() == 10 and FA.sum() == 1
WITHIN_SRC_NAMES = [PNAME[k] for k in np.where(WSRC)[0]]
WOUND_REFS = [r for r in REFS if V.MODALITY[r] == "wound"]
COLON_REFS = [r for r in REFS if V.MODALITY[r] == "colonoscopy"]
OTHER_REFS = [r for r in REFS if V.MODALITY[r] not in ("wound", "colonoscopy")]


def comp_masks(domain):
    dom = np.array([LAB[i] == domain for i in range(NR)])
    ia, ib = dom[PI], dom[PJ]
    return {"in x in": ia & ib, "in x out": ia != ib,
            "out, same modality": ~ia & ~ib & SAME,
            "out, across modality": ~ia & ~ib & ~SAME}


COMP = {"wound": comp_masks("wound"), "colonoscopy": comp_masks("colonoscopy")}


# ------------------------------------------------------------- correlation
def zr(T):
    """Centred, unit-norm average ranks along the last axis."""
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def corrmat(T):
    """T[..., datasets, methods] -> Spearman matrix [..., datasets, datasets]."""
    Z = zr(T)
    return np.einsum("...in,...jn->...ij", Z, Z)


def nmean(A, mask):
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(A[..., mask], axis=-1)


def qmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def summ(R):
    """Every scalar used below, from a correlation matrix over DS."""
    rp = R[..., PI, PJ]
    o = {"modality_diff": nmean(rp, SAME) - nmean(rp, ACROSS),
         "modality_diff_excl_within_source": nmean(rp, DSSM) - nmean(rp, ACROSS),
         "same dataset": R[..., I_AZH, I_CTRL],
         "same source": nmean(rp, WSRC),
         "different source, same modality": nmean(rp, DSSM),
         "across modality": nmean(rp, ACROSS),
         "same modality (16 pairs)": nmean(rp, SAME),
         "wound pairs": nmean(rp, WOUNDP),
         "wound different-source pairs": nmean(rp, WOUND_DS),
         "fuseg~azh": rp[..., np.where(FA)[0][0]],
         "colonoscopy pairs": nmean(rp, COLONP)}
    for dom, ms in COMP.items():
        for k, m in ms.items():
            o[f"comp|{dom}|{k}"] = nmean(rp, m)
    return o


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def fp(r):
    """p value with the smallest attainable value marked."""
    return f"{r['p']:.5f}" + (" (min)" if abs(r["p"] - r["min_p"]) < 1e-12 else "")


def fci(est, lo, hi):
    return f"{f3(est)} [{f3(lo)}, {f3(hi)}]"


def fiv(iv):
    return "n/a" if iv is None else f"[{f3(iv[0])}, {f3(iv[1])}]"


def fl(x):
    return None if x is None or not np.isfinite(x) else float(x)


# ------------------------------------------------------------------- data
print("loading", flush=True)
DATA = {m: {} for m in METRICS}
for m in METRICS:
    for d in DS:
        DATA[m][d] = V.load(d, m)
for d in DS:
    assert DATA["dice"][d]["ids"] == DATA["nsd"][d]["ids"]
    assert list(DATA["dice"][d]["units"]) == list(DATA["nsd"][d]["units"])
NC = DATA["dice"][DS[0]]["X"].shape[1]
UNITS = {}
for d in DS:
    u = DATA["dice"][d]["units"]
    uniq, inv = np.unique(u, return_inverse=True)
    UNITS[d] = inv
MEANS = {m: np.stack([DATA[m][d]["X"].mean(0) for d in DS]) for m in METRICS}   # 12 x NC
MEANS["dice_noempty"] = np.stack([DATA["dice"][d]["X"][~DATA["dice"][d]["empty"]].mean(0)
                                  for d in DS])
USUM = {m: {d: np.zeros((UNITS[d].max() + 1, NC)) for d in DS} for m in METRICS}
UCNT = {d: np.bincount(UNITS[d]).astype(float) for d in DS}
for m in METRICS:
    for d in DS:
        np.add.at(USUM[m][d], UNITS[d], DATA[m][d]["X"])

V5 = json.load(open(os.path.join(ROOT, "results", "stats_v5.json")))
V6J = json.load(open(os.path.join(ROOT, "results", "v6", "stats_v6.json")))
S = {"definitions": {
    "data": "scripts/perimage_v7.py load(); V6 images, exclusions and units (photograph for Medetec and AZH incl. control, image otherwise)",
    "agreement": "Spearman (Pearson on average ranks) of per-method mean scores between two datasets over a method set",
    "pairs": "55 pairs of the 11 REFS; same modality 16 (of which 3 within source: " + ", ".join(WITHIN_SRC_NAMES) + "; 13 different source), across modality 39; control = azh vs azh_train_slice",
    "modality_diff": "mean same-modality pair agreement (16) minus mean across-modality agreement (39)",
    "modality_diff_excl_within_source": "mean of the 13 different-source same-modality pairs minus mean of the 39 across-modality pairs; relabeling test drops the 3 within-source pairs by identity (52 pairs)",
    "seed_average": "per image, mean of the per-image scores over the seeds that exist (perimage_v7 '<arch>@seeds'); method mean = mean of the seed means",
    "method_sets": {s: {"groups": g, "n": int(len(set_cols(s)))} for s, g in GROUP_SETS.items()},
    "groups": {g: list(V.GROUPS[g]) for g in GNAMES},
    "n_splits": N_SPLIT, "seed_split": SEED_SPLIT, "seed_image_split": SEED_SPLIT + 1,
    "B_hier": B_HIER, "seed_hier": SEED_HIER,
    "method_only": "stats_v5 design: default_rng(0) per pool, methods resampled within group, B 1000 and B 2000",
    "hier_pool_order": V5_POOLS + V7_BOOT,
    "seed_analysis_boot": {"B_method": B_SEEDBOOT, "seed_method": SEED_SEEDBOOT,
                           "B_hier": B_HIER, "seed_hier_arch": SEED_SEEDHIER,
                           "units": "section-3 unit draws (seed 6060)"}},
    "n_units": {d: int(UCNT[d].size) for d in DS}, "n_images": {d: int(UCNT[d].sum()) for d in DS}}

# full-data point estimates and check against stats_v5
POINT = {}
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        R = corrmat(MEANS[m][:, set_cols(s)])
        POINT[f"{m}|{s}"] = {k: float(v) for k, v in summ(R).items()}
        POINT[f"{m}|{s}"]["_R"] = R
chk = []
for m in ("dice", "nsd", "dice_noempty"):
    for s in V5_POOLS:
        h = V5["hier"][f"{m}|{s}"]
        for k in ("modality_diff", "same source", "different source, same modality",
                  "across modality", "same dataset"):
            chk.append(abs(h[k]["est"] - POINT[f"{m}|{s}"][k]))
for key, v in V5["comp"].items():
    m, s, dom = key.split("|")
    for k, e in v.items():
        chk.append(abs(e["est"] - POINT[f"{m}|{s}"][f"comp|{dom}|{k}"]))
S["check_point_estimates_vs_v5"] = {"n_compared": len(chk), "max_abs_diff": float(max(chk))}
print("point estimates vs v5: max abs diff", max(chk), flush=True)

# independent route for a few V7 point estimates: scipy.stats.spearmanr
ind = []
for m in METRICS:
    for s in ("M19", "U+M19", "N8s"):
        c = set_cols(s)
        vals = [spearmanr(MEANS[m][DS.index(a), c], MEANS[m][DS.index(b), c])[0]
                for a, b in itertools.combinations(WOUND_REFS, 2)]
        ind.append(abs(np.mean(vals) - POINT[f"{m}|{s}"]["wound pairs"]))
        ctl = spearmanr(MEANS[m][I_AZH, c], MEANS[m][I_CTRL, c])[0]
        ind.append(abs(ctl - POINT[f"{m}|{s}"]["same dataset"]))
S["check_independent_scipy"] = {"n_compared": len(ind), "max_abs_diff": float(max(ind)),
                                "what": "wound-pair mean and control via scipy.stats.spearmanr, sets M19, U+M19, N8s, dice and nsd"}
print("scipy route: max abs diff", max(ind), flush=True)

# ================================================== 1 + 2  split halves
print("split halves", flush=True)
rng = np.random.default_rng(SEED_SPLIT)
H1, H2 = {m: {} for m in METRICS}, {m: {} for m in METRICS}
for d in DS:
    nu = UCNT[d].size
    perm = np.argsort(rng.random((N_SPLIT, nu)), axis=1)
    A = np.zeros((N_SPLIT, nu), bool)
    A[np.arange(N_SPLIT)[:, None], perm[:, :nu // 2]] = True
    Af, Bf = A.astype(float), (~A).astype(float)
    for m in METRICS:
        H1[m][d] = (Af @ USUM[m][d]) / (Af @ UCNT[d])[:, None]
        H2[m][d] = (Bf @ USUM[m][d]) / (Bf @ UCNT[d])[:, None]

# by-image split (V6: ALL57 Dice); same splits reused for the V7 sets
rng_i = np.random.default_rng(SEED_SPLIT + 1)
IMG_SPLIT = {}
for d in DS:
    n = len(DATA["dice"][d]["X"])
    perm = np.argsort(rng_i.random((N_SPLIT, n)), axis=1)
    A = np.zeros((N_SPLIT, n), bool)
    A[np.arange(N_SPLIT)[:, None], perm[:, :n // 2]] = True
    IMG_SPLIT[d] = A


def rel_img(s, m="dice"):
    out = {}
    c = set_cols(s)
    for d in DS:
        X = DATA[m][d]["X"][:, c]
        A = IMG_SPLIT[d]
        h1 = (A.astype(float) @ X) / A.sum(1)[:, None]
        h2 = ((~A).astype(float) @ X) / (~A).sum(1)[:, None]
        r = (zr(h1) * zr(h2)).sum(-1)
        out[d] = float(np.median(2 * r / (1 + r)))
    return out


REL_IMG = rel_img("ALL57")
REL_IMG_V7 = {f"{m}|{s}": rel_img(s, m) for m in METRICS for s in ("ALL57", "ALL", "M", "M11s", "M19")}

REL, DIS = {}, {}
for m in METRICS:
    for s in GROUP_SETS:
        c = set_cols(s)
        Z1 = np.stack([zr(H1[m][d][:, c]) for d in DS])     # 12 x splits x n
        Z2 = np.stack([zr(H2[m][d][:, c]) for d in DS])
        rel = (Z1 * Z2).sum(-1)                              # 12 x splits
        sb = 2 * rel / (1 + rel)
        REL[f"{m}|{s}"] = {d: {"spearman_brown_median": float(np.median(sb[k])),
                               "lo": ci(sb[k])[0], "hi": ci(sb[k])[1],
                               "half_r_median": float(np.median(rel[k]))}
                           for k, d in enumerate(DS)}
        c11 = np.einsum("asn,bsn->abs", Z1, Z1)
        c22 = np.einsum("asn,bsn->abs", Z2, Z2)
        rab = (c11 + c22) / 2                                # 12 x 12 x splits
        prod = rel[:, None, :] * rel[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            dis = np.where(prod > 0, rab / np.sqrt(np.where(prod > 0, prod, 1)), np.nan)
        relm = rel.mean(1)
        with np.errstate(invalid="ignore"):
            rom = rab.mean(2) / np.sqrt(np.outer(relm, relm))
        Robs = POINT[f"{m}|{s}"]["_R"]
        pairs = list(itertools.combinations(range(len(DS)), 2))
        per_pair = {}
        for a, b in pairs:
            x = dis[a, b]
            per_pair[f"{DS[a]}~{DS[b]}"] = {
                "observed_full": float(Robs[a, b]), "half_r_ab_mean": float(rab[a, b].mean()),
                "dis_mean": float(np.nanmean(x)) if np.isfinite(x).any() else None,
                "dis_median": float(np.nanmedian(x)) if np.isfinite(x).any() else None,
                "dis_ratio_of_means": float(rom[a, b]),
                "n_undefined": int((~np.isfinite(x)).sum())}
        cats = {"wound pairs (6)": [(REFS[i], REFS[j]) for i, j, w in zip(PI, PJ, WOUNDP) if w],
                "wound different-source (5)": [(REFS[i], REFS[j]) for i, j, w in zip(PI, PJ, WOUND_DS) if w],
                "fuseg~azh": [("fuseg", "azh")],
                "colonoscopy pairs (10)": [(REFS[i], REFS[j]) for i, j, w in zip(PI, PJ, COLONP) if w],
                "control azh~azh_train_slice": [("azh", V.CONTROL)]}
        summ_c = {}
        for cname, plist in cats.items():
            ia = [DS.index(a) for a, b in plist]
            ib = [DS.index(b) for a, b in plist]
            x = dis[ia, ib]                                  # pairs x splits
            per_split = qmean(x, 0)
            summ_c[cname] = {
                "observed_full_mean": float(np.mean(Robs[ia, ib])),
                "half_r_ab_mean": float(rab[ia, ib].mean()),
                "dis_mean": float(np.nanmean(per_split)), "dis_median": float(np.nanmedian(per_split)),
                "split_lo": ci(per_split)[0], "split_hi": ci(per_split)[1],
                "dis_ratio_of_means": float(np.mean(rom[ia, ib])),
                "share_split_values_gt1": float(np.nanmean(x > 1)),
                "n_undefined_pair_splits": int((~np.isfinite(x)).sum()), "n_pair_splits": int(x.size)}
        DIS[f"{m}|{s}"] = {"summary": summ_c, "pairs": per_pair}
S["reliability"] = REL
S["reliability_all57_dice_by_image_split"] = REL_IMG
S["reliability_by_image_split_v7"] = REL_IMG_V7
S["disattenuated"] = DIS
print(f"  done {time.time() - T0:.0f}s", flush=True)

# ================================================== 3  hierarchical bootstrap
print("bootstrap", flush=True)


def v5_method_takes(pool, b, seed=0):
    """Exactly the method draws of stats_v5.boot (same RNG stream)."""
    nm = sum((V.GROUPS[g] for g in GROUP_SETS[pool]), [])
    pos = {n: k for k, n in enumerate(nm)}
    idx = [np.array([pos[n] for n in V.GROUPS[g]]) for g in GROUP_SETS[pool]]
    colmap = V.cols(nm)
    r = np.random.default_rng(seed)
    return np.stack([colmap[np.concatenate([r.choice(i, len(i)) for i in idx])] for _ in range(b)])


rng = np.random.default_rng(SEED_HIER)
UMEANS = {m: np.zeros((B_HIER, ND, NC)) for m in METRICS}
for k, d in enumerate(DS):
    nu = UCNT[d].size
    draw = rng.integers(0, nu, size=(B_HIER, nu))
    W = np.zeros((B_HIER, nu))
    np.add.at(W, (np.repeat(np.arange(B_HIER), nu), draw.ravel()), 1.0)
    for m in METRICS:
        UMEANS[m][:, k, :] = (W @ USUM[m][d]) / (W @ UCNT[d])[:, None]


def gather(Mtensor, take):
    """Mtensor (B, D, C) or (D, C); take (B, n) -> (B, D, n)."""
    if Mtensor.ndim == 2:
        return Mtensor[:, take].transpose(1, 0, 2)
    return np.take_along_axis(Mtensor, take[:, None, :].repeat(Mtensor.shape[1], 1), axis=2)


BOOT = {}
QK = ["modality_diff", "modality_diff_excl_within_source", "same dataset", "same source",
      "different source, same modality", "across modality", "wound pairs",
      "wound different-source pairs", "fuseg~azh", "colonoscopy pairs"] + \
     [f"comp|{d}|{k}" for d in COMP for k in COMP[d]]
for s in V5_POOLS + V7_BOOT:
    gt = [rng.choice(GCOLS[g], size=(B_HIER, GCOLS[g].size)) for g in GROUP_SETS[s]]
    take_h = np.concatenate(gt, axis=1)
    take_v5h = v5_method_takes(s, 1000)
    take_v5c = v5_method_takes(s, 2000)
    full_cols = np.broadcast_to(set_cols(s), (B_HIER, set_cols(s).size))
    for m in METRICS:
        res = {}
        hier = summ(corrmat(gather(UMEANS[m], take_h)))
        unit = summ(corrmat(gather(UMEANS[m], full_cols)))
        rh = summ(corrmat(gather(MEANS[m], take_v5h)))
        rc = summ(corrmat(gather(MEANS[m], take_v5c)))
        for q in QK:
            res[q] = {"est": POINT[f"{m}|{s}"][q],
                      "method_only_repro_B1000": ci(rh[q]), "method_only_repro_B2000": ci(rc[q]),
                      "units_only": ci(unit[q]), "hierarchical": ci(hier[q]),
                      "hier_sd": float(np.nanstd(hier[q])),
                      "hier_p_le0": float(np.nanmean(hier[q] <= 0))}
        BOOT[f"{m}|{s}"] = res
S["bootstrap"] = BOOT


def v5_interval(m, s, q):
    """Method-only interval reported in stats_v5.json, with its source key."""
    h = V5["hier"].get(f"{m}|{s}", {})
    hk = {"modality_diff": "modality_diff", "same dataset": "same dataset",
          "same source": "same source", "different source, same modality": "different source, same modality",
          "across modality": "across modality"}
    if q in hk:
        e = h[hk[q]]
        return [e["lo"], e["hi"]], f"hier[{m}|{s}]", "B1000"
    if q.startswith("comp|"):
        _, dom, k = q.split("|")
        key = f"{m}|{s}|{dom}"
        if key in V5["comp"]:
            e = V5["comp"][key][k]
            return [e["lo"], e["hi"]], f"comp[{key}]", "B2000"
        return None, None, None
    if q == "wound pairs":
        key = f"{m}|{s}|wound"
        if key in V5["comp"]:
            e = V5["comp"][key]["in x in"]
            return [e["lo"], e["hi"]], f"comp[{key}] in x in", "B2000"
        if m == "dice":
            e = V5["bench40"][s]["clinical wound pairs"]
            return [e["lo"], e["hi"]], f"bench40[{s}] clinical wound pairs", "B2000"
    return None, None, None


cmpchk = []
for s in V5_POOLS:
    for m in METRICS:
        for q in QK:
            iv, src, bb = v5_interval(m, s, q)
            BOOT[f"{m}|{s}"][q]["v5_method_only"] = iv
            BOOT[f"{m}|{s}"][q]["v5_source"] = src
            if iv is not None:
                rep = BOOT[f"{m}|{s}"][q]["method_only_repro_" + bb]
                cmpchk.append(max(abs(iv[0] - rep[0]), abs(iv[1] - rep[1])))
S["check_v5_intervals_reproduced"] = {"n_compared": len(cmpchk), "max_abs_diff": float(max(cmpchk))}
print("  v5 method-only intervals reproduced: max abs diff", max(cmpchk), f"{time.time() - T0:.0f}s", flush=True)

# ================================================== 4  modality test variants
print("exact tests", flush=True)
CODE = {"wound": 0, "colonoscopy": 1, "dermoscopy": 2, "ultrasound": 3}
PERM_CACHE = {}


def exact_test(R, keys, drop_pairs=()):
    """Exact relabeling test of mean same-label minus mean different-label
    agreement over the pairs of `keys` (indices into REFS), with the pairs in
    drop_pairs (set of frozensets of REFS names) removed by identity."""
    pr = [(a, b) for a, b in itertools.combinations(keys, 2)
          if frozenset((REFS[a], REFS[b])) not in drop_pairs]
    ia = np.array([keys.index(a) for a, b in pr])
    ib = np.array([keys.index(b) for a, b in pr])
    r = np.array([R[a, b] for a, b in pr])
    lab = [CODE[LAB[k]] for k in keys]
    kk = tuple(lab)
    if kk not in PERM_CACHE:
        PERM_CACHE[kk] = np.array(list(multiset_permutations(lab)))
    L = PERM_CACHE[kk]
    same = L[:, ia] == L[:, ib]
    stat = (same * r).sum(1) / same.sum(1) - ((~same) * r).sum(1) / (~same).sum(1)
    obs_same = np.array(lab)[ia] == np.array(lab)[ib]
    obs = r[obs_same].mean() - r[~obs_same].mean()
    n_equiv = int((same == obs_same).all(1).sum())
    return {"diff": float(obs), "p": float(np.mean(stat >= obs - 1e-12)),
            "n_relabelings": int(len(L)), "min_p": n_equiv / len(L), "n_equivalent_relabelings": n_equiv, "n_pairs": int(len(pr)),
            "n_same": int(obs_same.sum()), "n_across": int((~obs_same).sum())}


DROP = {frozenset(p.split("~")) for p in WITHIN_SRC_NAMES}
ALLK = list(range(NR))
EX = {}
pchk = []
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        R = POINT[f"{m}|{s}"]["_R"]
        full = exact_test(R, ALLK)
        excl = exact_test(R, ALLK, DROP)
        P = POINT[f"{m}|{s}"]
        EX[f"{m}|{s}"] = {"full_55": full, "excl_within_source_52": excl,
                          "levels_excl": {"different source, same modality (13)": P["different source, same modality"],
                                          "across modality (39)": P["across modality"],
                                          "same source (3, excluded)": P["same source"],
                                          "same dataset (control)": P["same dataset"]}}
        if s in V5_POOLS:
            v = V5["exact"][f"{m}|{s}"]["modality"]
            pchk.append(max(abs(v["diff"] - full["diff"]), abs(v["p"] - full["p"])))
S["exact"] = EX
S["check_v5_exact_reproduced"] = {"n_compared": len(pchk), "max_abs_diff": float(max(pchk))}
print("  v5 exact tests reproduced: max abs diff", max(pchk), flush=True)

LODO = {}
for m in METRICS:
    for s in ("M", "U", "M11s", "M19", "N8s"):
        R = POINT[f"{m}|{s}"]["_R"]
        LODO[f"{m}|{s}"] = {}
        for k in range(NR):
            keys = [x for x in ALLK if x != k]
            LODO[f"{m}|{s}"][REFS[k]] = {"full": exact_test(R, keys),
                                         "excl_within_source": exact_test(R, keys, DROP)}
S["leave_one_dataset_out"] = LODO


def lomo(g):
    out = {}
    for m in METRICS:
        mc = GCOLS[g]
        out[m] = {}
        for k, n in enumerate(V.GROUPS[g]):
            c = np.delete(mc, k)
            P = summ(corrmat(MEANS[m][:, c]))
            out[m][n] = {q: float(P[q]) for q in ("modality_diff", "modality_diff_excl_within_source", "wound pairs")}
    return out


S["leave_one_method_out_M"] = lomo("M")
S["leave_one_method_out_M19"] = lomo("M19")

# ================================================== 5  stratified agreement
print("stratified", flush=True)
STR = {}
for s in ("U+F", "U+K", "U+M", "U+M11s", "U+M19", "ALL"):
    gs = GROUP_SETS[s]
    for m in METRICS:
        T = MEANS[m]
        Wp = np.empty((len(DS), sum(GCOLS[g].size for g in gs)))
        off = 0
        within, var = {}, {}
        for g in gs:
            n = GCOLS[g].size
            Wp[:, off:off + n] = (rankdata(T[:, GCOLS[g]], axis=1) - 0.5) / n
            within[g] = corrmat(T[:, GCOLS[g]])
            var[g] = Wp[:, off:off + n].var(1)                 # per dataset
            off += n
        Rs = corrmat(Wp)                                       # stratified (v5 definition)
        Zp = Wp - Wp.mean(1, keepdims=True)
        Zp /= np.sqrt((Zp ** 2).sum(1, keepdims=True))
        Rp = Zp @ Zp.T                                         # Pearson on percentiles
        ns = {g: GCOLS[g].size for g in gs}
        Rw = sum(ns[g] * within[g] for g in gs) / sum(ns.values())      # size-weighted
        num = sum(ns[g] * within[g] * np.sqrt(np.outer(var[g], var[g])) for g in gs)
        den = np.sqrt(np.outer(sum(ns[g] * var[g] for g in gs), sum(ns[g] * var[g] for g in gs)))
        Rf = num / den                                         # exact Pearson identity
        Ru = sum(within[g] for g in gs) / len(gs)              # unweighted
        Rraw = POINT[f"{m}|{s}"]["_R"]                         # unstratified pooled
        out = {}
        for lab_, R in (("stratified_spearman", Rs), ("pearson_on_percentiles", Rp),
                        ("weighted_identity", Rf), ("size_weighted_average", Rw),
                        ("unweighted_average", Ru), ("pooled_unstratified", Rraw)):
            P = summ(R)
            out[lab_] = {q: float(P[q]) for q in ("wound pairs", "modality_diff", "colonoscopy pairs",
                                                  "modality_diff_excl_within_source")}
        rp_s, rp_w = Rs[PI, PJ], Rw[PI, PJ]
        out["max_abs_pair_diff_stratified_vs_weighted"] = float(np.max(np.abs(rp_s - rp_w)))
        out["max_abs_pair_diff_pearsonpct_vs_identity"] = float(np.max(np.abs(Rp[PI, PJ] - Rf[PI, PJ])))
        if s in V5["stratified"]:
            out["v5_stratified_modality_diff"] = V5["stratified"][s][m]["modality_diff"]
            out["v5_stratified_wound_in_x_in"] = V5["stratified"][s][m]["categories_wound"]["in x in"]
        out["within_group"] = {g: {q: float(summ(within[g])[q]) for q in ("wound pairs", "modality_diff")} for g in gs}
        STR[f"{m}|{s}"] = out
S["stratified"] = STR

# ================================================== 6  empty-mask-excluded Dice
NOEMPTY = {}
for s in V5_POOLS + ["ALL57"] + list(V7_SETS):
    P = POINT[f"dice_noempty|{s}"]
    NOEMPTY[s] = {"modality_diff": P["modality_diff"], "p": EX[f"dice_noempty|{s}"]["full_55"]["p"],
                  "dice_with_empty": POINT[f"dice|{s}"]["modality_diff"],
                  "excl_within_source": P["modality_diff_excl_within_source"],
                  "wound pairs": P["wound pairs"], "wound pairs with empty": POINT[f"dice|{s}"]["wound pairs"]}
    if s in V5_POOLS:
        v = V5["hier"][f"dice_noempty|{s}"]["modality_diff"]
        NOEMPTY[s]["v5_est"] = v["est"]
        NOEMPTY[s]["v5_p"] = V5["exact"][f"dice_noempty|{s}"]["modality"]["p"]
S["dice_noempty"] = NOEMPTY
S["n_empty"] = {d: int(DATA["dice"][d]["empty"].sum()) for d in DS}

# ================================================== 7  floor effect
FLOOR = {"mean_dice": {}, "share_dice_lt_0.05_median_over_methods": {}, "flag_lt_0.1": {}}
for g in GNAMES:
    FLOOR["mean_dice"][g] = {d: float(MEANS["dice"][k, GCOLS[g]].mean()) for k, d in enumerate(DS)}
    FLOOR["flag_lt_0.1"][g] = [d for d in DS if FLOOR["mean_dice"][g][d] < 0.1]
    FLOOR["share_dice_lt_0.05_median_over_methods"][g] = {
        d: float(np.median((DATA["dice"][d]["X"][:, GCOLS[g]] < 0.05).mean(0))) for d in DS}
mchk = [abs(V5["mean_dice"][g][d] - FLOOR["mean_dice"][g][d])
        for g in ("U", "F", "K", "M", "S") for d in (DS if g != "S" else REFS)]
FLOOR["check_vs_v5_mean_dice_max_abs_diff"] = float(max(mchk))
S["floor"] = FLOOR

# ================================================== 8  seeds
print("seeds", flush=True)
FULL3 = {c: mem for c, mem in V.SEEDCOLS.items() if len(mem) == 3}
P9 = [c for c in V.GROUPS["P10s"] if c in FULL3]
N7 = [c for c in V.GROUPS["N8s"] if c in FULL3]
SEED_SETS = {"P9": P9, "N7": N7, "C16": P9 + N7}
SCOL = {k: np.array([[V.NAMES_ALL.index(FULL3[c][s]) for c in v] for s in range(3)])
        for k, v in SEED_SETS.items()}                       # 3 x n, row = seed
ACOL = {k: V.cols(v) for k, v in SEED_SETS.items()}
SEEDPAIRS = [(0, 1), (0, 2), (1, 2)]
XPERM = list(itertools.permutations(range(3), 2))           # ordered seed pairs s != t
# checks: member order is seed 0, 1, 2; seed 0 of the old ten = V6 M columns;
# seed-average column = mean of the seed columns
for c in P9:
    assert FULL3[c][0] == c.replace("@seeds", "") and FULL3[c][1].endswith("_s1") and FULL3[c][2].endswith("_s2"), c
for c in N7:
    assert [x[-3:] for x in FULL3[c]] == ["_s0", "_s1", "_s2"], c
seedchk = []
for m in METRICS:
    for k in SEED_SETS:
        seedchk.append(float(np.max(np.abs(MEANS[m][:, ACOL[k]] - MEANS[m][:, SCOL[k]].mean(1)))))
v6m = V6L.cols([c.replace("@seeds", "") for c in P9])
s0eq = all(np.array_equal(V6L.load(d, m)["X"][:, v6m], DATA[m][d]["X"][:, SCOL["P9"][0]])
           for d in DS for m in METRICS)
assert s0eq


def seed_block(T3, Ta):
    """T3 [..., 3, D, n] per-seed method means, Ta [..., D, n] seed-average
    means over the same architectures. Returns a dict of arrays [...]."""
    Z = zr(T3)
    Za = zr(Ta)
    rs = np.stack([(Z[..., a, :, :] * Z[..., b, :, :]).sum(-1) for a, b in SEEDPAIRS], -2)  # [...,3,D]
    rbar = rs.mean(-2)                                                                       # [...,D]
    Rs = np.einsum("...sin,...sjn->...sij", Z, Z)                                            # [...,3,D,D]
    Ra = np.einsum("...in,...jn->...ij", Za, Za)
    Rx = np.stack([np.einsum("...in,...jn->...ij", Z[..., a, :, :], Z[..., b, :, :]) for a, b in XPERM], -3)
    o = {"rbar": rbar, "rs": rs}
    widx = [DS.index(r) for r in WOUND_REFS]
    cidx = [DS.index(r) for r in COLON_REFS]
    o["rbar_wound"] = rbar[..., widx].mean(-1)
    o["rbar_colon"] = rbar[..., cidx].mean(-1)
    o["rbar_refs11"] = rbar[..., :NR].mean(-1)
    o["rbar_control"] = rbar[..., I_CTRL]
    ss, sa = summ(Rs), summ(Ra)
    for q in ("wound pairs", "colonoscopy pairs", "modality_diff", "same dataset", "fuseg~azh"):
        o[f"single|{q}"] = qmean(ss[q], -1)
        for s in range(3):
            o[f"seed{s}|{q}"] = ss[q][..., s]
        o[f"avg|{q}"] = sa[q]
        o[f"avg_minus_single|{q}"] = sa[q] - qmean(ss[q], -1)
    rx = Rx[..., PI, PJ]                                                                     # [...,6,55]
    o["crossseed|wound pairs"] = qmean(rx[..., WOUNDP].reshape(rx.shape[:-2] + (-1,)), -1)
    o["crossseed|colonoscopy pairs"] = qmean(rx[..., COLONP].reshape(rx.shape[:-2] + (-1,)), -1)
    o["gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"] = o["rbar_wound"] - o["crossseed|wound pairs"]
    o["_Rs"], o["_Ra"] = Rs, Ra
    return o


SCAL = ["rbar_wound", "rbar_colon", "rbar_refs11", "rbar_control",
        "single|wound pairs", "avg|wound pairs", "avg_minus_single|wound pairs", "crossseed|wound pairs",
        "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed",
        "seed0|wound pairs", "seed1|wound pairs", "seed2|wound pairs",
        "single|colonoscopy pairs", "avg|colonoscopy pairs", "avg_minus_single|colonoscopy pairs",
        "crossseed|colonoscopy pairs",
        "single|modality_diff", "avg|modality_diff", "avg_minus_single|modality_diff",
        "single|same dataset", "avg|same dataset", "single|fuseg~azh", "avg|fuseg~azh"]

SEEDS = {"sets": {k: {"architectures": v, "n": len(v),
                      "columns_by_seed": [[V.NAMES_ALL[i] for i in row] for row in SCOL[k]]}
                  for k, v in SEED_SETS.items()},
         "check_seedavg_equals_mean_of_seed_means_max_abs": float(max(seedchk)),
         "check_seed0_old_architectures_equal_v6_columns": bool(s0eq)}
rng_sb = np.random.default_rng(SEED_SEEDBOOT)
rng_sh = np.random.default_rng(SEED_SEEDHIER)
SB_TAKE = {k: rng_sb.integers(0, len(v), size=(B_SEEDBOOT, len(v))) for k, v in SEED_SETS.items()}
SH_TAKE = {k: rng_sh.integers(0, len(v), size=(B_HIER, len(v))) for k, v in SEED_SETS.items()}
SEEDPOINT = {}
for m in METRICS:
    for k in SEED_SETS:
        n = len(SEED_SETS[k])
        T3 = MEANS[m][:, SCOL[k]].transpose(1, 0, 2)                      # 3 x D x n
        Ta = MEANS[m][:, ACOL[k]]
        o = seed_block(T3, Ta)
        SEEDPOINT[f"{m}|{k}"] = o
        # method bootstrap (architectures resampled)
        tk = SB_TAKE[k]
        T3b = np.stack([gather(MEANS[m], SCOL[k][s][tk]) for s in range(3)], 1)   # B x 3 x D x n
        ob = seed_block(T3b, gather(MEANS[m], ACOL[k][tk]))
        # hierarchical (architectures + units) and units-only
        th = SH_TAKE[k]
        T3h = np.stack([gather(UMEANS[m], SCOL[k][s][th]) for s in range(3)], 1)
        oh = seed_block(T3h, gather(UMEANS[m], ACOL[k][th]))
        fu = np.broadcast_to(np.arange(n), (B_HIER, n))
        T3u = np.stack([gather(UMEANS[m], SCOL[k][s][fu]) for s in range(3)], 1)
        ou = seed_block(T3u, gather(UMEANS[m], ACOL[k][fu]))
        res = {}
        for q in SCAL:
            res[q] = {"est": float(o[q]), "method_boot": ci(ob[q]), "units_only": ci(ou[q]),
                      "hierarchical": ci(oh[q]), "method_boot_p_le0": float(np.nanmean(ob[q] <= 0))}
        per_ds = {}
        for j, d in enumerate(DS):
            rb = float(o["rbar"][j])
            per_ds[d] = {"r_01": float(o["rs"][0, j]), "r_02": float(o["rs"][1, j]), "r_12": float(o["rs"][2, j]),
                         "rbar": rb, "spearman_brown_3": fl(3 * rb / (1 + 2 * rb)) if rb > -0.5 else None,
                         "rbar_method_boot": ci(ob["rbar"][:, j])}
        SEEDS[f"{m}|{k}"] = {"n": n, "summary": res, "per_dataset": per_ds,
                             "run_effect_check": {"mean_r01_over_11_refs": float(o["rs"][0, :NR].mean()),
                                                  "mean_r02_over_11_refs": float(o["rs"][1, :NR].mean()),
                                                  "mean_r12_over_11_refs": float(o["rs"][2, :NR].mean()),
                                                  "mean_r01_wound": float(o["rs"][0, [DS.index(r) for r in WOUND_REFS]].mean()),
                                                  "mean_r02_wound": float(o["rs"][1, [DS.index(r) for r in WOUND_REFS]].mean()),
                                                  "mean_r12_wound": float(o["rs"][2, [DS.index(r) for r in WOUND_REFS]].mean())}}
# independent route: scipy for one cell
d0 = DS.index("azh")
sc = spearmanr(MEANS["dice"][d0, SCOL["C16"][0]], MEANS["dice"][d0, SCOL["C16"][1]])[0]
SEEDS["check_scipy_azh_dice_C16_r01"] = {"scipy": float(sc), "engine": float(SEEDPOINT["dice|C16"]["rs"][0, d0])}
assert abs(sc - SEEDPOINT["dice|C16"]["rs"][0, d0]) < 1e-12

# named single-seed groups (as defined in perimage_v7) vs seed-averaged groups
NAMED = {"M (V6) = P10_s0 + nnunet_2d": V.GROUPS["P10_s0"] + ["nnunet_2d"],
         "M_s1 = P10_s1 + nnunet_2d": V.GROUPS["P10_s1"] + ["nnunet_2d"],
         "M_s2 = P10_s2 + nnunet_2d": V.GROUPS["P10_s2"] + ["nnunet_2d"],
         "M11s": V.GROUPS["M11s"],
         "M19_s0 = P10_s0 + N8_s0 + nnunet_2d": V.GROUPS["P10_s0"] + V.GROUPS["N8_s0"] + ["nnunet_2d"],
         "M19_s1 = P10_s1 + N8_s1 + nnunet_2d": V.GROUPS["P10_s1"] + V.GROUPS["N8_s1"] + ["nnunet_2d"],
         "M19_s2 = P10_s2 + N8_s2 + nnunet_2d": V.GROUPS["P10_s2"] + V.GROUPS["N8_s2"] + ["nnunet_2d"],
         "M19": V.GROUPS["M19"],
         "N8_s0": V.GROUPS["N8_s0"], "N8_s1": V.GROUPS["N8_s1"], "N8_s2": V.GROUPS["N8_s2"], "N8s": V.GROUPS["N8s"]}
SEEDS["named_groups"] = {}
for m in METRICS:
    for nm, members in NAMED.items():
        P = summ(corrmat(MEANS[m][:, V.cols(members)]))
        SEEDS["named_groups"][f"{m}|{nm}"] = {"n": len(members), **{q: float(P[q]) for q in
                                              ("wound pairs", "colonoscopy pairs", "modality_diff", "same dataset")}}

# ---- 8c disattenuation for seed noise and image-sampling noise
#   observed score x = t + e_seed + e_img (independent).  Split halves of the
#   same trained models keep e_seed (it is a property of the model) and differ
#   in e_img: R_img = (T + S/k)/V.  Two seeds on the same images keep e_img and
#   differ in e_seed: Spearman-Brown over k seeds R_seed = (T + I)/V.  Hence
#   the reliability of the k-seed average is R = R_img + R_seed - 1 = T/V
#   (additive rule), and r* = r_AB / sqrt(R_A R_B).  Product rule shown too.
def split_rel(m, c):
    Z1 = np.stack([zr(H1[m][d][:, c]) for d in DS])
    Z2 = np.stack([zr(H2[m][d][:, c]) for d in DS])
    rel = (Z1 * Z2).sum(-1)
    return np.median(2 * rel / (1 + rel), 1)                 # per dataset


def dis_pair(r, Ra, Rb):
    if not (np.isfinite(Ra) and np.isfinite(Rb)) or Ra <= 0 or Rb <= 0:
        return None
    return float(r / math.sqrt(Ra * Rb))


DISS = {}
for m in METRICS:
    for k in SEED_SETS:
        o = SEEDPOINT[f"{m}|{k}"]
        rbar = o["rbar"]
        Ri_avg = split_rel(m, ACOL[k])
        Ri_seed = np.stack([split_rel(m, SCOL[k][s]) for s in range(3)])   # 3 x D
        Ri_single = Ri_seed.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Rs3 = np.where(rbar > -0.5, 3 * rbar / (1 + 2 * rbar), np.nan)
        Rs1 = rbar
        Rc_avg = Ri_avg + Rs3 - 1
        Rc_single = Ri_single + Rs1 - 1
        Rp_avg = Ri_avg * Rs3
        Ra_obs = o["_Ra"]
        Rs_obs = qmean(o["_Rs"], 0)                                          # mean over seeds, D x D
        perds = {d: {"R_img_seedavg": float(Ri_avg[j]), "R_img_single_mean": float(Ri_single[j]),
                     "R_img_single_by_seed": [float(x) for x in Ri_seed[:, j]],
                     "R_seed_1": float(Rs1[j]), "R_seed_3": fl(Rs3[j]),
                     "R_combined_avg_additive": fl(Rc_avg[j]), "R_combined_single_additive": fl(Rc_single[j]),
                     "R_combined_avg_product": fl(Rp_avg[j])} for j, d in enumerate(DS)}
        cats = {"wound pairs (6)": [(DS.index(REFS[i]), DS.index(REFS[j])) for i, j, w in zip(PI, PJ, WOUNDP) if w],
                "colonoscopy pairs (10)": [(DS.index(REFS[i]), DS.index(REFS[j])) for i, j, w in zip(PI, PJ, COLONP) if w],
                "control azh~azh_train_slice": [(I_AZH, I_CTRL)]}
        out = {"per_dataset": perds, "categories": {}, "wound_pairs": {}}
        for cname, plist in cats.items():
            rows = []
            for a, b in plist:
                row = {"observed_seedavg": float(Ra_obs[a, b]), "observed_single_mean": float(Rs_obs[a, b]),
                       "dis_img_only": dis_pair(Ra_obs[a, b], Ri_avg[a], Ri_avg[b]),
                       "dis_seed_only": dis_pair(Ra_obs[a, b], Rs3[a], Rs3[b]),
                       "dis_additive": dis_pair(Ra_obs[a, b], Rc_avg[a], Rc_avg[b]),
                       "dis_product": dis_pair(Ra_obs[a, b], Rp_avg[a], Rp_avg[b]),
                       "dis_additive_single": dis_pair(Rs_obs[a, b], Rc_single[a], Rc_single[b])}
                rows.append(row)
                if cname.startswith("wound"):
                    out["wound_pairs"][f"{DS[a]}~{DS[b]}"] = row
            agg = {"n_pairs": len(rows),
                   "observed_seedavg_mean": float(np.mean([r["observed_seedavg"] for r in rows])),
                   "observed_single_mean": float(np.mean([r["observed_single_mean"] for r in rows]))}
            for key in ("dis_img_only", "dis_seed_only", "dis_additive", "dis_product", "dis_additive_single"):
                vals = np.array([np.nan if r[key] is None else r[key] for r in rows])
                ok = np.isfinite(vals)
                agg[key] = {"raw_mean": fl(vals[ok].mean()) if ok.any() else None,
                            "capped_mean": fl(np.clip(vals[ok], -1, 1).mean()) if ok.any() else None,
                            "n_defined": int(ok.sum()), "n_abs_gt1": int((np.abs(vals[ok]) > 1).sum()),
                            "min": fl(vals[ok].min()) if ok.any() else None,
                            "max": fl(vals[ok].max()) if ok.any() else None}
            out["categories"][cname] = agg
        DISS[f"{m}|{k}"] = out
SEEDS["disattenuation"] = DISS

# ---- 8d spread of method means vs seed noise
SPREAD = {}
for m in METRICS:
    for d_i, d in enumerate(DS):
        mu19 = MEANS[m][d_i, GCOLS["M19"]]
        srt = np.sort(mu19)
        X3 = MEANS[m][d_i, SCOL["C16"]]                                    # 3 x 16
        sd_seed = X3.std(0, ddof=1)
        avg16 = X3.mean(0)
        Vb = avg16.var(ddof=1)
        Wv = (sd_seed ** 2).mean()
        Tb = Vb - Wv / 3
        # all architectures with >= 2 seeds (adds manet_r18_in, 2 seeds)
        sd2 = [MEANS[m][d_i, V.cols(mem)].std(ddof=1) for c, mem in V.SEEDCOLS.items() if len(mem) >= 2]
        img_se = UMEANS[m][:, d_i, GCOLS["M19"]].std(0, ddof=1)
        SPREAD[f"{m}|{d}"] = {
            "M19_n": int(mu19.size), "M19_mean": float(mu19.mean()), "M19_min": float(mu19.min()), "M19_max": float(mu19.max()),
            "M19_range": float(mu19.max() - mu19.min()), "M19_sd": float(mu19.std(ddof=1)),
            "M19_median_adjacent_gap": float(np.median(np.diff(srt))),
            "C16_between_sd_of_seedavg": float(math.sqrt(Vb)),
            "C16_seed_sd_pooled": float(math.sqrt(Wv)), "C16_seed_sd_median": float(np.median(sd_seed)),
            "C16_seed_sd_max": float(sd_seed.max()),
            "C16_se_of_3seed_avg": float(math.sqrt(Wv / 3)),
            "archs_ge2_seeds_n": len(sd2), "archs_ge2_seeds_sd_median": float(np.median(sd2)),
            "C16_true_between_var_est": float(Tb),
            "C16_icc_single": fl(Tb / (Tb + Wv)) if Tb + Wv > 0 else None,
            "C16_icc_3seed_avg": fl(Tb / Vb) if Vb > 0 else None,
            "ratio_between_sd_to_seed_sd": float(math.sqrt(Vb) / math.sqrt(Wv)),
            "M19_image_se_median": float(np.median(img_se)),
            "M19_image_se_max": float(img_se.max())}
SEEDS["spread"] = SPREAD
S["seeds"] = SEEDS
print(f"  seeds done {time.time() - T0:.0f}s", flush=True)

for k in POINT:
    POINT[k].pop("_R")
S["point"] = POINT


# ================================================== 0  V6 reproduction check
def walk(a, b, path, acc):
    if isinstance(a, dict):
        for k in a:
            if path == "" and k in ("definitions", "runtime_s"):
                continue
            if not isinstance(b, dict) or k not in b:
                acc["missing"].append(path + "/" + k)
                continue
            walk(a[k], b[k], path + "/" + k, acc)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            acc["missing"].append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", acc)
    elif isinstance(a, bool) or isinstance(a, str) or a is None:
        acc["n_other"] += 1
        if a != b:
            acc["bad"].append((path, a, b))
    else:
        acc["n_numeric"] += 1
        b = float("nan") if b is None else b
        if isinstance(b, (str, bool)):
            acc["bad"].append((path, a, b))
            return
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                acc["bad"].append((path, a, b))
            return
        ad = abs(a - b)
        rd = ad / max(abs(a), 1e-300)
        acc["max_abs"] = max(acc["max_abs"], ad)
        if ad == 0:
            acc["n_identical"] += 1
        if ad > TOL and rd > TOL:
            acc["bad"].append((path, a, b))


SJ = json.loads(json.dumps(S))                 # same float/NaN round trip as the V6 file
acc = {"n_numeric": 0, "n_other": 0, "n_identical": 0, "max_abs": 0.0, "bad": [], "missing": []}
walk(V6J, SJ, "", acc)
S["exclusions"] = C.EXCL_NAME
S["check_reproduces_stats_v6"] = {"n_numeric": acc["n_numeric"], "n_other": acc["n_other"],
                                  "n_numeric_bit_identical": acc["n_identical"],
                                  "max_abs_diff": acc["max_abs"], "n_failed": len(acc["bad"]),
                                  "n_missing": len(acc["missing"]), "tolerance": TOL,
                                  "excluded": ["definitions (text)", "runtime_s"],
                                  "exclusions": C.EXCL_NAME, "enforced": C.OLD}
print("V6 comparison:", S["check_reproduces_stats_v6"], flush=True)
if C.OLD and (acc["bad"] or acc["missing"]):
    print("FAILED leaves (first 20):", acc["bad"][:20], acc["missing"][:20])
    sys.exit("V6 values not reproduced under the V6 exclusions; stopping before writing any output")
if not C.OLD:
    # V6 was computed under exclusions_v5.json; reproduction is enforced by the run
    # with WB_EXCL=exclusions_v5.json (results/v7_excl_v5). Here the differences are
    # the effect of the new exclusions and are only recorded.
    S["check_reproduces_stats_v6"]["note"] = ("V6 used exclusions_v5.json; values that involve AZH or the control differ by "
                                              "design. Bit-identical reproduction of V6 is checked in results/v7_excl_v5.")
    S["check_reproduces_stats_v6"]["first_differing_leaves"] = [x[0] for x in acc["bad"][:20]]

# earlier V7 attempt (results/v7_morning/stats_v7.json, same M19 definition): informative cross-check
OLD7 = os.path.join(ROOT, "results", "v7_morning", "stats_v7.json")
if os.path.exists(OLD7) and C.OLD:
    o7 = json.load(open(OLD7))["point"]
    cmp7 = {}
    for m in ("dice", "nsd", "dice_noempty"):
        for s7, s in (("M19", "M19"), ("U+M19", "U+M19"), ("ALL65", "ALL")):
            if f"{m}|{s7}" in o7:
                cmp7[f"{m}|{s}"] = max(abs(o7[f"{m}|{s7}"][q] - POINT[f"{m}|{s}"][q]) for q in POINT[f"{m}|{s}"])
    S["check_vs_earlier_v7_attempt"] = {"max_abs_diff_point": float(max(cmp7.values())), "per_set": cmp7}
    print("earlier V7 attempt, point estimates M19/U+M19/ALL: max abs diff", max(cmp7.values()))

S["runtime_s"] = round(time.time() - T0, 1)
os.makedirs(OUT, exist_ok=True)
json.dump(S, open(os.path.join(OUT, "stats_v7.json"), "w"), indent=1)
print(f"saved {C.V7OUT_REL}/stats_v7.json", f"{time.time() - T0:.0f}s", flush=True)

# ================================================================ markdown
SD = S["seeds"]
MOD = ["M", "M11s", "M19", "N8s"]
MIX = ["U+M", "U+M11s", "U+M19", "ALL57", "ALL"]
SHOWSETS = MOD + MIX
NSET = {s: S["definitions"]["method_sets"][s]["n"] for s in GROUP_SETS}
L = []
w = L.append


def bq(m, s, q, key):
    return BOOT[f"{m}|{s}"][q][key]


def cell(m, s, q):
    b = BOOT[f"{m}|{s}"][q]
    return f"{f3(b['est'])} m{fiv(b['method_only_repro_B2000'])} h{fiv(b['hierarchical'])}"


def sv(m, k, q):
    return SD[f"{m}|{k}"]["summary"][q]


def sfc(m, k, q, key="method_boot"):
    v = sv(m, k, q)
    return f"{f3(v['est'])} {fiv(v[key])}"


w("# STATS V7: modern methods at three seeds and 19 modern architectures\n")
w("Generated by `scripts/v7/stats_v7.py` from `scripts/perimage_v7.py`. All numbers in `results/v7/stats_v7.json`. "
  "Port of `scripts/v6/stats_v6.py` (same definitions, same random seeds, same bootstrap sizes).\n")
w("An earlier, independent V7 implementation (older loader API) is kept in `scripts/v7_morning` and `results/v7_morning` "
  "and is used only as a cross-check.\n")

# ---------------------------------------------------------------- summary
d19, n19 = POINT["dice|M19"], POINT["nsd|M19"]
cs_d, cs_n = SD["dice|C16"]["summary"], SD["nsd|C16"]["summary"]
w("## Summary in plain language\n")
w(f"- **Modern methods across the wound datasets.** Mean agreement over the 6 wound pairs was {f3(POINT['dice|M']['wound pairs'])} for V6 M "
  f"(seed 0 only). With every architecture averaged over its seeds it is {f3(POINT['dice|M11s']['wound pairs'])} for the same 11 methods (M11s) "
  f"and {f3(d19['wound pairs'])} for all 19 modern methods (M19, hierarchical interval {fiv(bq('dice', 'M19', 'wound pairs', 'hierarchical'))}). "
  f"NSD: {f3(POINT['nsd|M']['wound pairs'])} (M), {f3(POINT['nsd|M11s']['wound pairs'])} (M11s), {f3(n19['wound pairs'])} (M19). "
  "Agreement stays low and the intervals are wide (for Dice they include 0); the V6 value 0.126 sits inside them.")
w(f"- **Mixed pools hardly move.** U+M wound-pair agreement {f3(POINT['dice|U+M']['wound pairs'])} (V6) becomes {f3(POINT['dice|U+M19']['wound pairs'])} "
  f"with U+M19 (Dice) and {f3(POINT['nsd|U+M']['wound pairs'])} becomes {f3(POINT['nsd|U+M19']['wound pairs'])} (NSD). ALL57 {f3(POINT['dice|ALL57']['wound pairs'])} "
  f"becomes ALL {f3(POINT['dice|ALL']['wound pairs'])} (Dice).")
w(f"- **Modality difference.** Mixed pools: U+M {f3(POINT['dice|U+M']['modality_diff'])} becomes U+M19 {f3(POINT['dice|U+M19']['modality_diff'])} "
  f"(exact p {fp(EX['dice|U+M19']['full_55'])}). Modern methods alone: M {f3(POINT['dice|M']['modality_diff'])} becomes "
  f"M11s {f3(POINT['dice|M11s']['modality_diff'])} and M19 {f3(d19['modality_diff'])} (exact p {fp(EX['dice|M19']['full_55'])}, "
  f"hierarchical interval {fiv(bq('dice', 'M19', 'modality_diff', 'hierarchical'))}); the 8 new architectures alone show none "
  f"(N8s {f3(POINT['dice|N8s']['modality_diff'])}, p {fp(EX['dice|N8s']['full_55'])}). The V6 value for M depends on seed 0: the same 11 "
  f"methods trained with seed 1 give {f3(SD['named_groups']['dice|M_s1 = P10_s1 + nnunet_2d']['modality_diff'])} and with seed 2 "
  f"{f3(SD['named_groups']['dice|M_s2 = P10_s2 + nnunet_2d']['modality_diff'])}.")
w(f"- **Seed noise is large.** Two training seeds of the same 16 architectures, scored on the same wound dataset, agree only "
  f"{f3(cs_d['rbar_wound']['est'])} on average (Dice; method bootstrap {fiv(cs_d['rbar_wound']['method_boot'])}); on the control pair "
  f"azh~azh_train_slice they agree {f3(cs_d['rbar_control']['est'])}. Spread of the 19 modern method means on a wound dataset is "
  f"of the same order as the seed-to-seed spread of one architecture (section 10d).")
w(f"- **Seed averaging does not raise cross-dataset agreement.** Wound-pair agreement of the 16 architectures: single seeds "
  f"{f3(cs_d['single|wound pairs']['est'])} (mean of the 3 seeds), seed averages {f3(cs_d['avg|wound pairs']['est'])}, difference "
  f"{sfc('dice', 'C16', 'avg_minus_single|wound pairs')} (Dice). NSD: {f3(cs_n['single|wound pairs']['est'])} vs "
  f"{f3(cs_n['avg|wound pairs']['est'])}, difference {sfc('nsd', 'C16', 'avg_minus_single|wound pairs')}.")
w(f"- **But the dataset still matters beyond seed noise (Dice).** Same dataset, different seed: {f3(cs_d['rbar_wound']['est'])}. "
  f"Different wound dataset, different seed: {f3(cs_d['crossseed|wound pairs']['est'])}. Gap {sfc('dice', 'C16', 'gap|wound: same-dataset cross-seed minus cross-dataset cross-seed')} "
  f"(method bootstrap), hierarchical {fiv(cs_d['gap|wound: same-dataset cross-seed minus cross-dataset cross-seed']['hierarchical'])}. "
  f"For NSD the gap is {sfc('nsd', 'C16', 'gap|wound: same-dataset cross-seed minus cross-dataset cross-seed')} and its interval touches 0. "
  f"The gap comes from the 9 original architectures ({f3(SD['dice|P9']['summary']['gap|wound: same-dataset cross-seed minus cross-dataset cross-seed']['est'])}); "
  f"for the 7 new ones it is {f3(SD['dice|N7']['summary']['gap|wound: same-dataset cross-seed minus cross-dataset cross-seed']['est'])}, "
  "so among the new architectures cross-dataset disagreement is no larger than seed-to-seed disagreement.")
dd, dn = SD["disattenuation"]["dice|C16"]["categories"]["wound pairs (6)"], SD["disattenuation"]["nsd|C16"]["categories"]["wound pairs (6)"]
w(f"- **Corrected for both noise sources** (16 architectures, seed averages), Dice wound-pair agreement {f3(dd['observed_seedavg_mean'])} "
  f"becomes {f3(dd['dis_additive']['raw_mean'])} raw or {f3(dd['dis_additive']['capped_mean'])} with each pair capped at 1 "
  f"(additive rule; product rule {f3(dd['dis_product']['raw_mean'])} / {f3(dd['dis_product']['capped_mean'])}). NSD: "
  f"{f3(dn['observed_seedavg_mean'])} becomes {f3(dn['dis_additive']['raw_mean'])} raw, {f3(dn['dis_additive']['capped_mean'])} capped. "
  "So for Dice the low agreement is mostly real; for NSD most of the gap to 1 can be explained by noise. These corrections are fragile (section 10c).")
w("- **Are modern methods too close to rank?** Within one dataset, partly yes: on the wound datasets their mean Dice values sit within a few "
  "hundredths of each other, the typical gap between neighbours in the ranking is far smaller than the seed-to-seed spread, and a single-seed "
  "ranking is only moderately reproducible. A 3-seed average is reproducible enough to rank the clearly better and clearly worse architectures, "
  "not the middle of the pack.\n")

# ---------------------------------------------------------------- what changed
w("## What changed from V6\n")
w("- New data: seeds 1 and 2 of the 10 ImageNet-pretrained architectures (run 2) and 8 new architectures at seeds 0, 1, 2 (run 2). "
  "Three trainings diverged and are absent: manet_r18_in seed 2, manet_r50 seeds 0 and 2. So manet_r18_in@seeds averages 2 seeds and "
  "manet_r50@seeds is a single seed (seed 1).")
w("- New method sets: M11s, M19, N8s, U+M11s, U+M19, ALL (U+F+K+M19, 65 methods). All V6 sets are recomputed and match V6 exactly.")
w("- New section 10 (seeds): seed-to-seed ranking reliability, single seed vs seed average, disattenuation for seed and image noise, spread of method means.")
w("- Extended: leave-one-dataset-out for M11s, M19, N8s; leave-one-method-out for M19; stratified analysis for U+M11s, U+M19, ALL; "
  "by-image split reliability for ALL, M, M11s, M19; floor table for the new groups.")
w("- The bootstrap tables below show the method-only interval regenerated with the stats_v5 design at B 2000 (seed 0) for every set, "
  "so V6 and V7 sets are comparable. V6's tables showed the stored stats_v5 interval, which is B 1000 for modality difference and "
  "same dataset; both versions are in the JSON (`method_only_repro_B1000`, `method_only_repro_B2000`, `v5_method_only`).")
w("- Correction to the brief: the brief quotes Medetec split-half reliability 0.962 by photograph. V6 (ALL57, Dice, Spearman-Brown median over "
  f"500 photograph splits) is {f3(REL['dice|ALL57']['medetec']['spearman_brown_median'])}; by image {f3(REL_IMG['medetec'])}. "
  "The architectures present at all three seeds are 9 of the original 10 and 7 of the new 8 (16 in total), not 9, 6 and 15: only manet_r50 "
  "is missing seeds among the new 8.\n")

# ---------------------------------------------------------------- checks
w("## Checks\n")
c6 = S["check_reproduces_stats_v6"]
if C.OLD:
    w(f"- **V6 reproduced** (run under `exclusions_v5.json`, the V6 exclusions): every leaf of `results/v6/stats_v6.json` except the definition text and the runtime was recomputed here with the V7 loader "
      f"and compared: {c6['n_numeric']} numbers and {c6['n_other']} other values (labels, None, lists of names); {c6['n_numeric_bit_identical']} numbers "
      f"bit-identical, max abs diff {c6['max_abs_diff']:.1e}, failures {c6['n_failed']}, missing {c6['n_missing']} (tolerance {TOL:.0e}). The script stops before writing if any differ.")
else:
    w(f"- **Exclusions**: `{C.EXCL_NAME}`. V6 used `exclusions_v5.json`; bit-identical reproduction of V6 by this code is checked by the run with "
      f"`WB_EXCL=exclusions_v5.json` (`results/v7_excl_v5/STATS_V7.md`). Against V6, {c6['n_numeric_bit_identical']} of {c6['n_numeric']} numbers are unchanged "
      f"here and {c6['n_failed']} differ because AZH and the control lost images.")
w(f"- Point estimates vs stats_v5 ({S['check_point_estimates_vs_v5']['n_compared']} values): max abs diff {S['check_point_estimates_vs_v5']['max_abs_diff']:.2e}. "
  f"Method-only intervals vs stats_v5 ({S['check_v5_intervals_reproduced']['n_compared']}): {S['check_v5_intervals_reproduced']['max_abs_diff']:.2e}. "
  f"Exact tests vs stats_v5 ({S['check_v5_exact_reproduced']['n_compared']}): {S['check_v5_exact_reproduced']['max_abs_diff']:.2e}. "
  f"Group mean Dice vs stats_v5: {FLOOR['check_vs_v5_mean_dice_max_abs_diff']:.2e}.")
w(f"- Independent route: wound-pair mean and control for M19, U+M19, N8s recomputed with scipy.stats.spearmanr ({S['check_independent_scipy']['n_compared']} values): "
  f"max abs diff {S['check_independent_scipy']['max_abs_diff']:.1e}. Seed engine vs scipy (azh, Dice, 16 architectures, seed 0 vs 1): "
  f"{SD['check_scipy_azh_dice_C16_r01']['engine']:.6f} vs {SD['check_scipy_azh_dice_C16_r01']['scipy']:.6f}.")
w(f"- Seed columns: seed-averaged column equals the mean of the per-seed method means (max abs diff {SD['check_seedavg_equals_mean_of_seed_means_max_abs']:.1e}); "
  f"seed-0 columns of the 9 original architectures are identical to the V6 columns per image ({SD['check_seed0_old_architectures_equal_v6_columns']}).")
if "check_vs_earlier_v7_attempt" in S:
    w(f"- Earlier V7 attempt (`results/v7/stats_v7.json`, same M19 definition): point estimates for M19, U+M19 and ALL (its ALL65) differ by at most "
      f"{S['check_vs_earlier_v7_attempt']['max_abs_diff_point']:.1e}.")
w("- Operational note: during this work `scripts/v6/stats_v6.py` was run once by mistake, which rewrote `results/v6/stats_v6.json` and `STATS_V6.md`. "
  "The rewrite is identical to the original in every value (13,451 leaves bit-identical to a copy verified against the original); only "
  "`runtime_s` (8.8 to 11.5) and the md runtime line changed.\n")

# ---------------------------------------------------------------- definitions
w("## Definitions\n")
for k in ("data", "agreement", "pairs", "modality_diff", "modality_diff_excl_within_source", "seed_average"):
    w(f"- **{k}**: {S['definitions'][k]}")
w("- **method sets**: " + ", ".join(f"{s} ({v['n']})" for s, v in S["definitions"]["method_sets"].items()))
w(f"- **random seeds**: split halves {N_SPLIT} unit splits, seed {SEED_SPLIT}; by-image split seed {SEED_SPLIT + 1}; hierarchical bootstrap B {B_HIER}, "
  f"seed {SEED_HIER} (unit draws first, then method draws pool by pool in the order {', '.join(V5_POOLS + V7_BOOT)}; the V6 pools come first so their draws are unchanged); "
  f"method-only bootstrap stats_v5 design, seed 0 per pool, B 1000 and 2000; seed analysis: architecture bootstrap B {B_SEEDBOOT} seed {SEED_SEEDBOOT}, "
  f"hierarchical B {B_HIER} with architecture draws seed {SEED_SEEDHIER} and the section-3 unit draws.")
w("- **interval notation**: m[...] = method-only bootstrap (methods resampled within group, stats_v5 design, B 2000, seed 0); "
  "h[...] = hierarchical bootstrap (methods within group and units within dataset, B 1000, seed 6060); all 2.5 and 97.5 percentiles.")
w("- Units per dataset: " + ", ".join(f"{d} {S['n_units'][d]}/{S['n_images'][d]}" for d in DS) + " (units/images).\n")

# ---------------------------------------------------------------- 1 headline
w("## 1. Headline: V6 sets next to V7 sets\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | n | wound pairs (6) | colonoscopy pairs (10) | modality diff | exact p | diff excl within-source (p) | control |")
    w("|---|---|---|---|---|---|---|---|")
    for s in SHOWSETS:
        e = EX[f"{m}|{s}"]
        w(f"| {s} | {NSET[s]} | {cell(m, s, 'wound pairs')} | {cell(m, s, 'colonoscopy pairs')} | {cell(m, s, 'modality_diff')} | "
          f"{fp(e['full_55'])} | {f3(e['excl_within_source_52']['diff'])} ({fp(e['excl_within_source_52'])}) | {cell(m, s, 'same dataset')} |")
w("\nV6 sets: M, U+M, ALL57. V7 counterparts: M11s (same 11 methods, seed-averaged), M19 (adds 8 architectures), N8s (new 8 only), "
  "U+M11s, U+M19, ALL (U+F+K+M19).\n")

# ---------------------------------------------------------------- 2 levels
w("## 2. Agreement levels (point estimates, full data)\n")
LV = ["same dataset", "same source", "different source, same modality", "across modality", "same modality (16 pairs)",
      "wound pairs", "wound different-source pairs", "fuseg~azh", "colonoscopy pairs", "modality_diff", "modality_diff_excl_within_source"]
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | n | " + " | ".join(LV) + " |")
    w("|---" * (len(LV) + 2) + "|")
    for s in GROUP_SETS:
        w(f"| {s} | {NSET[s]} | " + " | ".join(f3(POINT[f'{m}|{s}'][q]) for q in LV) + " |")

# ---------------------------------------------------------------- 3 reliability
RS = ["M", "M11s", "M19", "N8s", "ALL57", "ALL"]
w("\n## 3. Split-half reliability (Spearman-Brown, median [2.5, 97.5] over 500 photograph/image unit splits)\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| dataset | " + " | ".join(RS) + " | ALL57 by image | ALL by image | M19 by image |")
    w("|---" * (len(RS) + 4) + "|")
    for d in DS:
        row = [fci(REL[f"{m}|{s}"][d]["spearman_brown_median"], REL[f"{m}|{s}"][d]["lo"], REL[f"{m}|{s}"][d]["hi"]) for s in RS]
        row += [f3(REL_IMG_V7[f"{m}|{s}"][d]) for s in ("ALL57", "ALL", "M19")]
        w(f"| {d} | " + " | ".join(row) + " |")
w("\nBy image: the same image-level split as V6 (seed 2), Spearman-Brown median. V6 reported the by-image value for ALL57 Dice only.\n")

# ---------------------------------------------------------------- 4 disattenuated (V6 method)
w("## 4. Disattenuated agreement for image sampling (V6 method)\n")
w("Per split: r_AB = mean(corr(A1,B1), corr(A2,B2)); r_AA = corr(A1,A2); dis = r_AB / sqrt(r_AA r_BB), undefined when r_AA r_BB <= 0. "
  "Reported: observed full-data mean, median over splits [2.5, 97.5] of the split distribution (spread, not a sampling interval), ratio of means. "
  "Not clipped. This corrects for image sampling only; seed noise is treated as signal here (section 10c corrects for both).\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | category | observed | median [2.5, 97.5] | ratio of means | share >1 | undefined |")
    w("|---|---|---|---|---|---|---|")
    for s in SHOWSETS:
        for c, v in DIS[f"{m}|{s}"]["summary"].items():
            if c == "wound different-source (5)":
                continue
            w(f"| {s} | {c} | {f3(v['observed_full_mean'])} | {fci(v['dis_median'], v['split_lo'], v['split_hi'])} | "
              f"{f3(v['dis_ratio_of_means'])} | {v['share_split_values_gt1']:.2f} | {v['n_undefined_pair_splits']}/{v['n_pair_splits']} |")

# ---------------------------------------------------------------- 5 bootstrap
w("\n## 5. Bootstrap intervals: method-only, units-only, hierarchical\n")
w("Method-only: stats_v5 design, B 2000, seed 0. Units-only: units resampled within dataset, all methods kept. Hierarchical: both, B 1000, seed 6060. "
  "Resampling units adds noise to every method mean, which pulls agreement down in each draw, so units-only and hierarchical intervals sit partly "
  "below the estimate (conservative for 'greater than 0').\n")
HQ = ["wound pairs", "colonoscopy pairs", "modality_diff", "modality_diff_excl_within_source", "same dataset"]
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | quantity | est | method-only B2000 | units-only | hierarchical | P(hier <= 0) |")
    w("|---|---|---|---|---|---|---|")
    for s in SHOWSETS:
        for q in HQ:
            b = BOOT[f"{m}|{s}"][q]
            w(f"| {s} | {q} | {f3(b['est'])} | {fiv(b['method_only_repro_B2000'])} | {fiv(b['units_only'])} | {fiv(b['hierarchical'])} | {b['hier_p_le0']:.3f} |")
    w(f"\n{m}, wound domain categories (in x in = 6 wound pairs; in x out = 28 wound vs non-wound; out, same modality = 10 colonoscopy pairs; out, across modality = 11):\n")
    w("| set | category | est | method-only B2000 | hierarchical |")
    w("|---|---|---|---|---|")
    for s in SHOWSETS:
        for k in COMP["wound"]:
            b = BOOT[f"{m}|{s}"][f"comp|wound|{k}"]
            w(f"| {s} | {k} | {f3(b['est'])} | {fiv(b['method_only_repro_B2000'])} | {fiv(b['hierarchical'])} |")

# ---------------------------------------------------------------- 6 exact tests
w("\n## 6. Modality difference: exact relabeling test\n")
w("Relabeling over all 13,860 distinct assignments of the 11 modality labels (as V6). Smallest attainable p = 2/13,860 = 0.00014, marked (min). "
  "52-pair version drops the 3 within-source pairs (" + ", ".join(WITHIN_SRC_NAMES) + ") under every relabeling.\n")
w("| metric | set | n | diff (55 pairs) | p | diff (52 pairs) | p | diff-source same-modality mean | across mean | same-source mean | control |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        e = EX[f"{m}|{s}"]
        lv = e["levels_excl"]
        w(f"| {m} | {s} | {NSET[s]} | {f3(e['full_55']['diff'])} | {fp(e['full_55'])} | {f3(e['excl_within_source_52']['diff'])} | "
          f"{fp(e['excl_within_source_52'])} | {f3(lv['different source, same modality (13)'])} | {f3(lv['across modality (39)'])} | "
          f"{f3(lv['same source (3, excluded)'])} | {f3(lv['same dataset (control)'])} |")
w("\n### Leave one dataset out (modality difference, exact p), modern sets\n")
w("| metric | dropped | M diff (p) | M11s diff (p) | M19 diff (p) | N8s diff (p) |")
w("|---|---|---|---|---|---|")
for m in METRICS:
    for d in REFS:
        w(f"| {m} | {d} | " + " | ".join(f"{f3(LODO[f'{m}|{s}'][d]['full']['diff'])} ({fp(LODO[f'{m}|{s}'][d]['full'])})" for s in MOD) + " |")
w("\n### Leave one method out of M19\n")
w("| metric | dropped | modality diff | excl within-source | wound pairs |")
w("|---|---|---|---|---|")
for m in METRICS:
    LM = S["leave_one_method_out_M19"][m]
    for n, r in LM.items():
        w(f"| {m} | {n} | {f3(r['modality_diff'])} | {f3(r['modality_diff_excl_within_source'])} | {f3(r['wound pairs'])} |")
    arr = np.array([[r["modality_diff"], r["wound pairs"]] for r in LM.values()])
    w(f"| {m} | range | {f3(arr[:, 0].min())} to {f3(arr[:, 0].max())} | | {f3(arr[:, 1].min())} to {f3(arr[:, 1].max())} |")

# ---------------------------------------------------------------- 7 stratified
w("\n## 7. Composition: stratified agreement equals the size-weighted within-group agreement\n")
w("| metric | set | quantity | stratified Spearman | Pearson on percentiles | exact weighted identity | size-weighted average | unweighted average | pooled unstratified |")
w("|---|---|---|---|---|---|---|---|---|")
for s in ("U+M", "U+M11s", "U+M19", "ALL"):
    for m in METRICS:
        v = STR[f"{m}|{s}"]
        for q in ("wound pairs", "modality_diff"):
            w(f"| {m} | {s} | {q} | {f3(v['stratified_spearman'][q])} | {f3(v['pearson_on_percentiles'][q])} | {f3(v['weighted_identity'][q])} | "
              f"{f3(v['size_weighted_average'][q])} | {f3(v['unweighted_average'][q])} | {f3(v['pooled_unstratified'][q])} |")
w("\nWithin-group wound-pair agreement used by the weighting:\n")
w("| metric | set | " + " | ".join(["U", "F", "K", "M", "M11s", "M19"]) + " |")
w("|---|---|---|---|---|---|---|---|")
for s in ("U+M", "U+M11s", "U+M19", "ALL"):
    for m in METRICS:
        wg = STR[f"{m}|{s}"]["within_group"]
        w(f"| {m} | {s} | " + " | ".join(f3(wg[g]["wound pairs"]) if g in wg else "" for g in ["U", "F", "K", "M", "M11s", "M19"]) + " |")
w("\nMax abs per-pair difference, stratified vs size-weighted: " +
  ", ".join(f"{k} {v['max_abs_pair_diff_stratified_vs_weighted']:.3f}" for k, v in STR.items()) +
  f". Pearson on percentiles vs exact identity: max {max(v['max_abs_pair_diff_pearsonpct_vs_identity'] for v in STR.values()):.1e}. "
  "Reason (as V6): within-group percentiles have mean 0.5 in every group, so pooled percentile correlation is a size-weighted mean of the "
  "within-group correlations. The large pooled (unstratified) values come from between-group differences in level, not from agreement within groups.\n")

# ---------------------------------------------------------------- 8 noempty
w("## 8. Empty-mask-excluded Dice\n")
w("Empty reference masks: " + ", ".join(f"{d} {n}" for d, n in S["n_empty"].items() if n) + ".\n")
w("| set | modality diff, no empty | exact p | with empties | wound pairs, no empty | with empties | excl within-source, no empty |")
w("|---|---|---|---|---|---|---|")
for s, v in NOEMPTY.items():
    w(f"| {s} | {f3(v['modality_diff'])} | {fp(EX['dice_noempty|' + s]['full_55'])} | {f3(v['dice_with_empty'])} | {f3(v['wound pairs'])} | "
      f"{f3(v['wound pairs with empty'])} | {f3(v['excl_within_source'])} |")

# ---------------------------------------------------------------- 9 floor
w("\n## 9. Floor effect\n")
w("Group mean Dice (mean over methods of per-method mean over images). Bold = below 0.1.\n")
FG = ["U", "F", "K", "M", "M11s", "M19", "N8s", "S"]
w("| dataset | " + " | ".join(FG) + " |")
w("|---" * (len(FG) + 1) + "|")
for d in DS:
    cells = []
    for g in FG:
        x = FLOOR["mean_dice"][g][d]
        cells.append(f"**{x:.3f}**" if x < 0.1 else f"{x:.3f}")
    w(f"| {d} | " + " | ".join(cells) + " |")
w("\nOn the 7 non-wound datasets every modern group is at the floor, so modern-method rankings there order near-zero scores. "
  "Colonoscopy agreement and the modality difference of the modern sets rest on these rankings.\n")

# ---------------------------------------------------------------- 10 seeds
w("## 10. Seeds: is the low wound agreement of modern methods real or seed noise?\n")
w("Architecture sets with all three seeds: " + "; ".join(f"{k} = {len(v)} ({', '.join(c.replace('@seeds', '') for c in v)})" for k, v in SEED_SETS.items()) +
  ". manet_r18_in (seeds 0, 1) and manet_r50 (seed 1) are left out of this section; nnU-Net has one model and is also left out. "
  "Seed 0 of the original 10 comes from run 1, seeds 1 and 2 from run 2; all seeds of the new 8 come from run 2.\n")
w("Intervals in this section: method bootstrap = architectures resampled with replacement, the same draw used for all three seeds and the seed "
  f"average, B {B_SEEDBOOT}, seed {SEED_SEEDBOOT}; hierarchical = architectures (seed {SEED_SEEDHIER}) and units (section-3 draws), B {B_HIER}.\n")

w("### 10a. Same dataset, different seed\n")
w("r_st = Spearman between the per-architecture mean scores of seed s and seed t on the same dataset (same images). r-bar = mean of r_01, r_02, r_12. "
  "Spearman-Brown reliability of the 3-seed average = 3 r-bar / (1 + 2 r-bar) (undefined when r-bar <= -0.5; negative values mean no reliability).\n")
for m in METRICS:
    w(f"\n#### {m}\n")
    w("| dataset | M19 mean Dice | " + " | ".join(f"{k} r01 | r02 | r12 | r-bar | SB3" for k in SEED_SETS) + " |")
    w("|---|---|" + "---|" * (5 * len(SEED_SETS)))
    for d in DS:
        row = [f3(FLOOR["mean_dice"]["M19"][d])]
        for k in SEED_SETS:
            p = SD[f"{m}|{k}"]["per_dataset"][d]
            row += [f3(p["r_01"]), f3(p["r_02"]), f3(p["r_12"]), f3(p["rbar"]), f3(p["spearman_brown_3"]) if p["spearman_brown_3"] is not None else "undef"]
        w(f"| {d} | " + " | ".join(row) + " |")
    w(f"\n{m}: mean r-bar by dataset group, with method bootstrap and hierarchical intervals:\n")
    w("| set | n | wound (4) | colonoscopy (5) | all 11 refs | control azh_train_slice |")
    w("|---|---|---|---|---|---|")
    for k in SEED_SETS:
        x = SD[f"{m}|{k}"]["summary"]
        w(f"| {k} | {len(SEED_SETS[k])} | " + " | ".join(f"{f3(x[q]['est'])} m{fiv(x[q]['method_boot'])} h{fiv(x[q]['hierarchical'])}"
                                                         for q in ("rbar_wound", "rbar_colon", "rbar_refs11", "rbar_control")) + " |")
    w(f"\n{m}, run check (mean over the 11 refs; wound only in brackets): " + "; ".join(
        f"{k} r01 {f3(SD[f'{m}|{k}']['run_effect_check']['mean_r01_over_11_refs'])} ({f3(SD[f'{m}|{k}']['run_effect_check']['mean_r01_wound'])}), "
        f"r02 {f3(SD[f'{m}|{k}']['run_effect_check']['mean_r02_over_11_refs'])} ({f3(SD[f'{m}|{k}']['run_effect_check']['mean_r02_wound'])}), "
        f"r12 {f3(SD[f'{m}|{k}']['run_effect_check']['mean_r12_over_11_refs'])} ({f3(SD[f'{m}|{k}']['run_effect_check']['mean_r12_wound'])})"
        for k in SEED_SETS) + ".")
w("\nFor the 9 original architectures, seeds 1 and 2 (same run) agree more with each other than either does with seed 0 (run 1) on Dice over all "
  "datasets; on the wound datasets the difference is small. A run effect (software or hardware differences between runs) cannot be "
  "separated from seed noise with these data.\n")

w("### 10b. Cross-dataset wound agreement: single seeds vs seed averages\n")
w("Single seed s: wound-pair mean agreement using every architecture's seed-s model. Single-seed mean: mean over s = 0, 1, 2. "
  "Seed average: using the per-image seed-averaged columns (perimage_v7 '@seeds') of the same architectures. "
  "Cross-seed cross-dataset: seed s on one dataset vs seed t on the other (s different from t; 6 ordered seed pairs x 6 wound pairs = 36 correlations), "
  "the counterpart of the same-dataset r-bar in 10a, so that training noise is present in both and only the dataset differs. "
  "Gap = wound r-bar (10a) minus cross-seed cross-dataset agreement.\n")
SQ = ["seed0|wound pairs", "seed1|wound pairs", "seed2|wound pairs", "single|wound pairs", "avg|wound pairs", "avg_minus_single|wound pairs",
      "rbar_wound", "crossseed|wound pairs", "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed",
      "single|colonoscopy pairs", "avg|colonoscopy pairs", "single|modality_diff", "avg|modality_diff", "single|same dataset", "avg|same dataset"]
SQN = {"seed0|wound pairs": "wound pairs, seed 0", "seed1|wound pairs": "wound pairs, seed 1", "seed2|wound pairs": "wound pairs, seed 2",
       "single|wound pairs": "wound pairs, single-seed mean", "avg|wound pairs": "wound pairs, seed average",
       "avg_minus_single|wound pairs": "seed average minus single-seed mean", "rbar_wound": "same wound dataset, different seed (r-bar)",
       "crossseed|wound pairs": "different wound dataset, different seed",
       "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed": "gap (same-dataset minus cross-dataset, both cross-seed)",
       "single|colonoscopy pairs": "colonoscopy pairs, single-seed mean", "avg|colonoscopy pairs": "colonoscopy pairs, seed average",
       "single|modality_diff": "modality diff, single-seed mean", "avg|modality_diff": "modality diff, seed average",
       "single|same dataset": "control, single-seed mean", "avg|same dataset": "control, seed average"}
for m in METRICS:
    w(f"\n#### {m}\n")
    w("| quantity | " + " | ".join(f"{k} (n {len(v)})" for k, v in SEED_SETS.items()) + " |")
    w("|---|---|---|---|")
    for q in SQ:
        w(f"| {SQN[q]} | " + " | ".join(f"{f3(sv(m, k, q)['est'])} m{fiv(sv(m, k, q)['method_boot'])} h{fiv(sv(m, k, q)['hierarchical'])}"
                                       for k in SEED_SETS) + " |")
w("\nNamed single-seed groups from perimage_v7 (point estimates; n differs because of the missing trainings):\n")
w("| metric | group | n | wound pairs | colonoscopy pairs | modality diff | control |")
w("|---|---|---|---|---|---|---|")
for k, v in SD["named_groups"].items():
    m, g = k.split("|", 1)
    w(f"| {m} | {g} | {v['n']} | {f3(v['wound pairs'])} | {f3(v['colonoscopy pairs'])} | {f3(v['modality_diff'])} | {f3(v['same dataset'])} |")

w("\n### 10c. Disattenuation for seed noise and image-sampling noise\n")
w("Model: observed method mean = true value + seed error + image-sampling error, independent. "
  "Split halves of the same trained models share the seed error (it is a property of the trained model) and differ in image error, so the "
  "split-half reliability R_img = (T + S/k) / V. Two seeds on the same images share the image error and differ in seed error, so the "
  "Spearman-Brown seed reliability of a k-seed average R_seed = (T + I) / V. With V = T + S/k + I, the reliability of the k-seed average is\n")
w("    R = R_img + R_seed - 1 = T / V   (additive rule)\n")
w("    r* = r_AB / sqrt(R_A R_B)\n")
w("R_img = Spearman-Brown median over the 500 unit splits for the same columns (seed-averaged columns for the seed average; mean over seeds of "
  "the single-seed values for single seeds); R_seed = Spearman-Brown of r-bar for 3 seeds, r-bar itself for one seed. The product rule "
  "r_AB / sqrt(R_img,A R_seed,A R_img,B R_seed,B) is shown as well; it corrects less. "
  "Limits: the additive rule is exact only if the image error is fully shared across seeds; any seed-specific part is counted twice and the rule "
  "then over-corrects. Both rules are written for Pearson correlations and are approximate for Spearman. With 16, 9 or 7 architectures the "
  "reliabilities themselves are noisy, and when R is small the correction explodes, so single pairs can exceed 1. "
  "Values are reported raw and with each pair capped to [-1, 1] before averaging; undefined when R <= 0 for either dataset.\n")
for m in METRICS:
    w(f"\n#### {m}: reliabilities per dataset (16 architectures)\n")
    w("| dataset | R_img seed avg | R_img single (mean) | R_seed 1 seed | R_seed 3 seeds | R additive, 3-seed avg | R additive, single | R product, 3-seed avg |")
    w("|---|---|---|---|---|---|---|---|")
    for d, p in SD["disattenuation"][f"{m}|C16"]["per_dataset"].items():
        w(f"| {d} | {f3(p['R_img_seedavg'])} | {f3(p['R_img_single_mean'])} | {f3(p['R_seed_1'])} | {f3(p['R_seed_3'])} | "
          f"{f3(p['R_combined_avg_additive'])} | {f3(p['R_combined_single_additive'])} | {f3(p['R_combined_avg_product'])} |")
    w(f"\n#### {m}: wound pairs, 16 architectures, seed averages\n")
    w("| pair | observed (seed avg) | observed (single-seed mean) | image only | seed only | additive | product | additive, from single seeds |")
    w("|---|---|---|---|---|---|---|---|")
    for p, r in SD["disattenuation"][f"{m}|C16"]["wound_pairs"].items():
        w(f"| {p} | {f3(r['observed_seedavg'])} | {f3(r['observed_single_mean'])} | {f3(r['dis_img_only'])} | {f3(r['dis_seed_only'])} | "
          f"{f3(r['dis_additive'])} | {f3(r['dis_product'])} | {f3(r['dis_additive_single'])} |")
    w(f"\n#### {m}: category means (raw / capped; n defined of n pairs; n with abs > 1)\n")
    w("| set | category | observed seed avg | observed single | image only | seed only | additive | product | additive from single seeds |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in SEED_SETS:
        for c, a in SD["disattenuation"][f"{m}|{k}"]["categories"].items():
            def g(key):
                v = a[key]
                return f"{f3(v['raw_mean'])} / {f3(v['capped_mean'])} ({v['n_defined']}/{a['n_pairs']}; {v['n_abs_gt1']})"
            w(f"| {k} | {c} | {f3(a['observed_seedavg_mean'])} | {f3(a['observed_single_mean'])} | {g('dis_img_only')} | {g('dis_seed_only')} | "
              f"{g('dis_additive')} | {g('dis_product')} | {g('dis_additive_single')} |")
w("\nConsistency check of the model: if it held, correcting single-seed agreement with single-seed reliabilities would give the same answer as "
  "correcting seed-averaged agreement with 3-seed reliabilities. The last column differs a lot from the 'additive' column, because single-seed "
  "reliabilities on Medetec and CO2Wounds are near zero and the correction blows up. Treat the corrected values as rough.\n")

w("### 10d. Spread of method means vs seed noise\n")
w("Per dataset. M19: range and SD of the 19 method means (seed-averaged), and median gap between neighbours after sorting. "
  "Seed SD: SD over the 3 seeds of one architecture's mean (ddof 1), pooled over the 16 architectures as the root mean variance; "
  "SE of a 3-seed average = pooled seed SD / sqrt(3). Between SD (16): SD of the 16 seed-averaged means. "
  "True between variance = between variance - mean seed variance / 3; ICC = share of between variance that is not seed noise "
  "(single seed: T / (T + W); 3-seed average: T / between variance). Image SE: SD of one method's mean over the 1,000 unit resamples "
  "(median over M19); it is mostly shared by all methods (hard images are hard for every method), so it moves rankings much less than "
  "its size suggests (the split-half reliabilities in section 3 are the relevant measure).\n")
for m in METRICS:
    w(f"\n#### {m}\n")
    w("| dataset | M19 mean | M19 range | M19 SD | median neighbour gap | between SD (16) | seed SD pooled | seed SD median | SE of 3-seed avg | between/seed SD | ICC single | ICC 3-seed | image SE median |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        p = SPREAD[f"{m}|{d}"]
        w(f"| {d} | {f3(p['M19_mean'])} | {f3(p['M19_range'])} | {f3(p['M19_sd'])} | {f3(p['M19_median_adjacent_gap'])} | "
          f"{f3(p['C16_between_sd_of_seedavg'])} | {f3(p['C16_seed_sd_pooled'])} | {f3(p['C16_seed_sd_median'])} | {f3(p['C16_se_of_3seed_avg'])} | "
          f"{p['ratio_between_sd_to_seed_sd']:.2f} | {f3(p['C16_icc_single'])} | {f3(p['C16_icc_3seed_avg'])} | {f3(p['M19_image_se_median'])} |")
wd = [SPREAD[f"dice|{d}"] for d in WOUND_REFS]
w(f"\nPlain reading (Dice, the 4 wound datasets): the 19 modern methods span {min(p['M19_range'] for p in wd):.3f} to {max(p['M19_range'] for p in wd):.3f} "
  f"in mean Dice (SD {min(p['M19_sd'] for p in wd):.3f} to {max(p['M19_sd'] for p in wd):.3f}). Retraining one architecture with another seed moves its mean "
  f"by a pooled SD of {min(p['C16_seed_sd_pooled'] for p in wd):.3f} to {max(p['C16_seed_sd_pooled'] for p in wd):.3f}. The median gap between neighbours in "
  f"the ranking is {min(p['M19_median_adjacent_gap'] for p in wd):.4f} to {max(p['M19_median_adjacent_gap'] for p in wd):.4f}, several times smaller than the "
  f"seed SD. Between-architecture SD is {min(p['ratio_between_sd_to_seed_sd'] for p in wd):.2f} to {max(p['ratio_between_sd_to_seed_sd'] for p in wd):.2f} "
  f"times the seed SD. So most neighbouring pairs cannot be ordered, a single seed gives a ranking of modest reliability "
  f"(ICC {min(p['C16_icc_single'] for p in wd):.2f} to {max(p['C16_icc_single'] for p in wd):.2f}), and a 3-seed average reaches "
  f"{min(p['C16_icc_3seed_avg'] for p in wd):.2f} to {max(p['C16_icc_3seed_avg'] for p in wd):.2f}. CO2Wounds is the weakest case.\n")

w("### 10e. Conclusion on the seed question\n")
g_d = cs_d["gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"]
g_n = cs_n["gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"]
w(f"1. Seed noise is a large part of the picture. One seed of the 16 architectures reproduces another seed's ranking on the same wound dataset "
  f"at only {f3(cs_d['rbar_wound']['est'])} (Dice) and {f3(cs_n['rbar_wound']['est'])} (NSD). The V6 value 0.126 for M was one draw from this noisy process: "
  f"the same 11 methods give {f3(SD['named_groups']['dice|M_s1 = P10_s1 + nnunet_2d']['wound pairs'])} with seed 1 and "
  f"{f3(SD['named_groups']['dice|M_s2 = P10_s2 + nnunet_2d']['wound pairs'])} with seed 2.")
w(f"2. Averaging 3 seeds does not raise cross-dataset agreement: {f3(cs_d['single|wound pairs']['est'])} vs {f3(cs_d['avg|wound pairs']['est'])} (Dice), "
  f"{f3(cs_n['single|wound pairs']['est'])} vs {f3(cs_n['avg|wound pairs']['est'])} (NSD); both intervals of the difference include 0.")
w(f"3. For Dice, wound datasets disagree more than seeds do: gap {f3(g_d['est'])}, method bootstrap {fiv(g_d['method_boot'])}, hierarchical {fiv(g_d['hierarchical'])}. "
  f"After correcting for seed and image noise the Dice wound-pair agreement is about {f3(dd['dis_additive']['capped_mean'])} to {f3(dd['dis_additive']['raw_mean'])} "
  f"(additive rule, capped and raw), far from 1. So the low Dice agreement of modern methods across wound datasets is mainly real, not seed noise. "
  f"This rests on the 9 original architectures; the 7 new ones alone show no gap ({f3(SD['dice|N7']['summary']['gap|wound: same-dataset cross-seed minus cross-dataset cross-seed']['est'])}).")
w(f"4. For NSD the gap is smaller ({f3(g_n['est'])}, method bootstrap {fiv(g_n['method_boot'])}) and the corrected agreement is "
  f"{f3(dn['dis_additive']['capped_mean'])} capped: NSD disagreement among modern methods is largely explainable by noise.")
w(f"5. Pre-specified reading in `results/v7/BRIEF_V7.md` (same-dataset cross-seed agreement above 0.7 with cross-dataset agreement about 0.1 to 0.3): "
  f"the first condition fails ({f3(cs_d['rbar_wound']['est'])} for Dice on the wound datasets; only the control pair, {f3(cs_d['rbar_control']['est'])}, passes). "
  "Under that rule the claim 'modern models disagree because the datasets differ, not because of training noise' is not supported as stated. "
  "The defensible statement is narrower: modern methods are close together and single-seed rankings are noisy, yet for Dice the wound datasets "
  "still order the original architectures differently by more than retraining does.\n")

# ---------------------------------------------------------------- caveats
w("## Caveats and limitations\n")
w("- Modern sets are small (19, 11, 8 methods; 16, 9, 7 in the seed analysis). Spearman over 7 to 9 items moves in steps of 0.02 to 0.04 and "
  "its sampling spread is large; all modern-set intervals are wide.")
w("- Seed averages are uneven: manet_r18_in@seeds averages 2 seeds and manet_r50@seeds is one seed, inside M19 and N8s.")
w("- Seed 0 of the original architectures comes from a different run than seeds 1 and 2; run and seed effects are confounded for those 9 (section 10a).")
w("- Modern methods score near 0 on all 7 non-wound datasets (section 9). Their colonoscopy agreement and modality difference rank floor-level "
  "scores and should not be read as evidence about colonoscopy.")
w("- The disattenuated values in section 10c rest on a simple variance model, on Spearman instead of Pearson, and on noisy reliabilities; "
  "single pairs exceed 1. Use them as a rough range, not as estimates.")
w("- Units-only and hierarchical intervals for agreement sit partly below the estimate because resampling adds noise that attenuates correlation.")
w("- The by-image split reliability for Medetec mixes the four stored variants of one photograph across halves and overstates reliability, "
  "and the photograph split is the one to use.\n")
w(f"Runtime {S['runtime_s']} s.\n")
txt = "\n".join(L)
assert "—" not in txt and "–" not in txt, "dash character in markdown"
open(os.path.join(OUT, "STATS_V7.md"), "w").write(txt)
print(f"saved {C.V7OUT_REL}/STATS_V7.md", f"{time.time() - T0:.0f}s")

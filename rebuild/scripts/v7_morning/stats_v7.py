#!/usr/bin/env python3
"""V7 rerun of the V6 agreement statistics with the phase-2 modern models,
written to results/v7/stats_v7.json and results/v7/STATS_V7.md.

Why this is a copy and not an import: scripts/v6/stats_v6.py is a script that
computes everything and writes results/v6 at import time, with the method groups
hardcoded, so importing it would overwrite V6 results. Its logic (functions zr,
corrmat, nmean, summ, ci, exact_test, comp_masks, v5_method_takes, gather and
sections 1 to 7) is copied here unchanged in substance and parameterised by the
method groups. The engine is validated by recomputing every V6 method set and
comparing every numeric leaf of results/v6/stats_v6.json (section 0 below);
the run stops if any differs by more than 1e-12.

Data: scripts/perimage_v7.py. Columns = the 59 V6 columns, the 18 seed-averaged
architecture columns "<arch>@avg", and the per-seed columns "<arch>@s<k>"
(load_seed, same images and exclusions).

Method groups: U 18, F 14, K 14, M 11 (V6: 10 seed-0 pretrained + nnU-Net),
S 2 (never pooled), M19 (18 seed-averaged architectures + nnU-Net, primary),
M17 (M19 without the MAnets), Ms0 / Ms1 / Ms2 (architectures that have seed k,
seed-k models + nnU-Net: 18 / 19 / 17 methods).
Method sets: the V6 sets U, F, K, M, U+F, U+K, U+M, ALL57, and the V7 sets
M19, U+M19, M17, U+M17, Ms0, Ms1, Ms2, U+Ms0, U+Ms1, U+Ms2, ALL65 (U+F+K+M19).

Definitions are those of stats_v6 (agreement, pair levels, modality difference,
categories = stats_v5 f_comp for the wound and colonoscopy domains).

Seeds and draw counts as V6: 500 unit splits (seed 1), hierarchical bootstrap
B 1000 (seed 6060), method-only bootstrap with the stats_v5 design (seed 0 per
pool, B 1000 and B 2000). The hierarchical RNG stream draws the unit resamples
first (identical to V6, they depend only on the number of units), then the
method resamples pool by pool: the 7 V6 pools in V6 order (identical draws),
then the V7 pools in the order listed in BOOT_POOLS.
Run from the rebuild folder:  python scripts/v7/stats_v7.py
"""
import itertools
import json
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
import perimage_v6 as V                                            # noqa: E402
import perimage_v7 as P                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "v7")
T0 = time.time()

N_SPLIT = 500
B_HIER = 1000
SEED_SPLIT = 1
SEED_HIER = 6060
METRICS = ("dice", "nsd")

REFS = list(V.REFS)
DS = REFS + [V.CONTROL]
NR = len(REFS)
I_AZH, I_CTRL = DS.index("azh"), DS.index(V.CONTROL)
LAB = [V.MODALITY[r] for r in REFS]
SRC = [V.SOURCE[r] for r in REFS]

# ------------------------------------------------------------------ columns and groups
SEEDCOLS = [f"{a}@s{s}" for s in P.SEEDS for a in P.ARCH if s in P.seeds_of(a)]
NAMES = list(P.NAMES7) + SEEDCOLS
assert NAMES[:len(V.NAMES_ALL)] == list(V.NAMES_ALL)
GROUPS = {g: list(P.GROUPS7[g]) for g in ("U", "F", "K", "M", "S", "M19", "M17")}
for s in P.SEEDS:
    GROUPS[f"Ms{s}"] = [f"{a}@s{s}" for a in P.ARCH if s in P.seeds_of(a)] + ["nnunet_2d"]
for g in ("U", "F", "K", "M", "S"):
    assert GROUPS[g] == V.GROUPS[g]
assert [len(GROUPS[g]) for g in ("M19", "M17", "Ms0", "Ms1", "Ms2")] == [19, 17, 18, 19, 17]


def colsN(names_subset):
    return np.array([NAMES.index(n) for n in names_subset])


GCOLS = {g: colsN(v) for g, v in GROUPS.items()}
for g in ("U", "F", "K", "M", "S"):
    assert np.array_equal(GCOLS[g], V.cols(V.GROUPS[g]))

V6_SETS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
           "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"],
           "ALL57": ["U", "F", "K", "M"]}
V7_SETS = {"M19": ["M19"], "U+M19": ["U", "M19"], "M17": ["M17"], "U+M17": ["U", "M17"],
           "Ms0": ["Ms0"], "Ms1": ["Ms1"], "Ms2": ["Ms2"],
           "U+Ms0": ["U", "Ms0"], "U+Ms1": ["U", "Ms1"], "U+Ms2": ["U", "Ms2"],
           "ALL65": ["U", "F", "K", "M19"]}
GROUP_SETS = {**V6_SETS, **V7_SETS}
V5_POOLS = ["U", "F", "K", "M", "U+F", "U+K", "U+M"]
BOOT_POOLS = V5_POOLS + ["M19", "U+M19", "M17", "U+M17", "Ms0", "Ms1", "Ms2", "U+Ms0", "U+Ms1", "U+Ms2"]
STRAT_SETS = ["U+F", "U+K", "U+M", "U+M19", "U+M17", "U+Ms0", "U+Ms1", "U+Ms2"]
LODO_SETS = ["M", "U", "M19", "U+M19", "U+M"]
LOMO_GROUPS = ["M", "M19"]
FLOOR_GROUPS = ["U", "F", "K", "M", "S", "M19", "M17", "Ms0", "Ms1", "Ms2"]


def set_cols(s):
    return np.concatenate([GCOLS[g] for g in GROUP_SETS[s]])


# ------------------------------------------------------------------ pairs (as stats_v6)
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


def comp_masks(domain):
    dom = np.array([LAB[i] == domain for i in range(NR)])
    ia, ib = dom[PI], dom[PJ]
    return {"in x in": ia & ib, "in x out": ia != ib,
            "out, same modality": ~ia & ~ib & SAME,
            "out, across modality": ~ia & ~ib & ~SAME}


COMP = {"wound": comp_masks("wound"), "colonoscopy": comp_masks("colonoscopy")}


# ------------------------------------------------------------- correlation (as stats_v6)
def zr(T):
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def corrmat(T):
    Z = zr(T)
    return np.einsum("...in,...jn->...ij", Z, Z)


def nmean(A, mask):
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(A[..., mask], axis=-1)


def summ(R):
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
    return f"{r['p']:.5f}" + (" (min)" if abs(r["p"] - r["min_p"]) < 1e-12 else "")


def fci(est, lo, hi):
    return f"{f3(est)} [{f3(lo)}, {f3(hi)}]"


# ------------------------------------------------------------------- data
def load_ext(d, m):
    D = P.load7(d, m)
    D6 = V.load(d, m)
    assert np.array_equal(D["X"][:, :len(V.NAMES_ALL)], D6["X"]) and D["ids"] == D6["ids"]
    extra, names = [], []
    for s in P.SEEDS:
        X, archs = P.load_seed(d, m, s)
        assert X.shape[0] == D["X"].shape[0]
        for a in archs:
            extra.append(X[:, archs.index(a)])
            names.append(f"{a}@s{s}")
    assert names == SEEDCOLS, (d, m)
    D["X"] = np.concatenate([D["X"], np.stack(extra, 1)], 1)
    D["names"] = D["names"] + names
    assert D["names"] == NAMES
    return D


print("loading", flush=True)
DATA = {m: {d: load_ext(d, m) for d in DS} for m in METRICS}
for d in DS:
    assert DATA["dice"][d]["ids"] == DATA["nsd"][d]["ids"]
    assert list(DATA["dice"][d]["units"]) == list(DATA["nsd"][d]["units"])
UNITS = {}
for d in DS:
    uniq, inv = np.unique(DATA["dice"][d]["units"], return_inverse=True)
    UNITS[d] = inv
NC = len(NAMES)
MEANS = {m: np.stack([DATA[m][d]["X"].mean(0) for d in DS]) for m in METRICS}
MEANS["dice_noempty"] = np.stack([DATA["dice"][d]["X"][~DATA["dice"][d]["empty"]].mean(0) for d in DS])
USUM = {m: {d: np.zeros((UNITS[d].max() + 1, NC)) for d in DS} for m in METRICS}
UCNT = {d: np.bincount(UNITS[d]).astype(float) for d in DS}
for m in METRICS:
    for d in DS:
        np.add.at(USUM[m][d], UNITS[d], DATA[m][d]["X"])

V5 = json.load(open(os.path.join(ROOT, "results", "stats_v5.json")))
V6J = json.load(open(os.path.join(ROOT, "results", "v6", "stats_v6.json")))
S = {"definitions": {
    "data": "scripts/perimage_v7.py (V6 images, exclusions and units); columns: V6 59, 18 seed-averaged '<arch>@avg', per-seed '<arch>@s<k>'",
    "engine": "logic copied from scripts/v6/stats_v6.py (it cannot be imported without rerunning it and overwriting results/v6); validated by reproducing every numeric leaf of results/v6/stats_v6.json for the V6 method sets",
    "agreement": "Spearman (Pearson on average ranks) of per-method mean scores between two datasets over a method set",
    "pairs": "55 pairs of the 11 REFS; same modality 16 (of which 3 within source: " + ", ".join(WITHIN_SRC_NAMES) + "; 13 different source), across modality 39; control = azh vs azh_train_slice",
    "modality_diff": "mean same-modality pair agreement (16) minus mean across-modality agreement (39)",
    "modality_diff_excl_within_source": "mean of the 13 different-source same-modality pairs minus mean of the 39 across-modality pairs; relabeling test drops the 3 within-source pairs by identity (52 pairs)",
    "categories": "stats_v5 f_comp: mean agreement over pairs in x in, in x out, out same modality, out across modality, for a domain (wound or colonoscopy)",
    "groups": {g: {"n": len(v), "members": v} for g, v in GROUPS.items()},
    "method_sets": {s: {"groups": g, "n": int(len(set_cols(s)))} for s, g in GROUP_SETS.items()},
    "n_splits": N_SPLIT, "seed_split": SEED_SPLIT, "B_hier": B_HIER, "seed_hier": SEED_HIER,
    "hier_pool_order": BOOT_POOLS,
    "method_only": "stats_v5 boot design: methods resampled within each group, seed 0 per pool, B 1000 (hier keys) and B 2000 (comp keys)"},
    "n_units": {d: int(UCNT[d].size) for d in DS}, "n_images": {d: int(UCNT[d].sum()) for d in DS}}

POINT = {}
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        R = corrmat(MEANS[m][:, set_cols(s)])
        POINT[f"{m}|{s}"] = {k: float(v) for k, v in summ(R).items()}
        POINT[f"{m}|{s}"]["_R"] = R

# independent route: wound pairs and control for M19 Dice with scipy, from load7 directly
ind = {}
for m in METRICS:
    mu = {d: P.load7(d, m)["X"][:, P.cols(P.GROUPS7["M19"])].mean(0) for d in DS}
    wp = [spearmanr(mu[a], mu[b])[0] for a, b in itertools.combinations(V.WOUND, 2)]
    ind[m] = {"wound pairs scipy": float(np.mean(wp)), "engine": POINT[f"{m}|M19"]["wound pairs"],
              "control scipy": float(spearmanr(mu["azh"], mu[V.CONTROL])[0]), "control engine": POINT[f"{m}|M19"]["same dataset"]}
    assert abs(ind[m]["wound pairs scipy"] - ind[m]["engine"]) < 1e-12
    assert abs(ind[m]["control scipy"] - ind[m]["control engine"]) < 1e-12
seeds_json = os.path.join(OUT, "seeds_v7.json")
if os.path.exists(seeds_json):
    SJ = json.load(open(seeds_json))
    for m in METRICS:
        ind[m]["seeds_v7 M19 wound pairs"] = SJ["cross_dataset_wound"][f"{m}|M19 seed-averaged"]["wound_pairs_mean"]
        assert abs(ind[m]["seeds_v7 M19 wound pairs"] - ind[m]["engine"]) < 1e-12
        for k in P.SEEDS:
            a_ = SJ["cross_dataset_wound"][f"{m}|seed {k} architectures + nnU-Net"]["wound_pairs_mean"]
            assert abs(a_ - POINT[f"{m}|Ms{k}"]["wound pairs"]) < 1e-12, (m, k)
        ind[m]["seeds_v7 Ms0/Ms1/Ms2 wound pairs equal"] = True
S["check_independent_M19"] = ind
print("independent M19 check ok", flush=True)

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

rng_i = np.random.default_rng(SEED_SPLIT + 1)
REL_IMG = {}
c57 = set_cols("ALL57")
for d in DS:
    X = DATA["dice"][d]["X"][:, c57]
    n = len(X)
    perm = np.argsort(rng_i.random((N_SPLIT, n)), axis=1)
    A = np.zeros((N_SPLIT, n), bool)
    A[np.arange(N_SPLIT)[:, None], perm[:, :n // 2]] = True
    h1 = (A.astype(float) @ X) / A.sum(1)[:, None]
    h2 = ((~A).astype(float) @ X) / (~A).sum(1)[:, None]
    r = (zr(h1) * zr(h2)).sum(-1)
    REL_IMG[d] = float(np.median(2 * r / (1 + r)))

REL, DIS = {}, {}
for m in METRICS:
    for s in GROUP_SETS:
        c = set_cols(s)
        Z1 = np.stack([zr(H1[m][d][:, c]) for d in DS])
        Z2 = np.stack([zr(H2[m][d][:, c]) for d in DS])
        rel = (Z1 * Z2).sum(-1)
        sb = 2 * rel / (1 + rel)
        REL[f"{m}|{s}"] = {d: {"spearman_brown_median": float(np.median(sb[k])),
                               "lo": ci(sb[k])[0], "hi": ci(sb[k])[1],
                               "half_r_median": float(np.median(rel[k]))}
                           for k, d in enumerate(DS)}
        c11 = np.einsum("asn,bsn->abs", Z1, Z1)
        c22 = np.einsum("asn,bsn->abs", Z2, Z2)
        rab = (c11 + c22) / 2
        prod = rel[:, None, :] * rel[None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            dis = np.where(prod > 0, rab / np.sqrt(np.where(prod > 0, prod, 1)), np.nan)
        relm = rel.mean(1)
        with np.errstate(invalid="ignore"):
            rom = rab.mean(2) / np.sqrt(np.outer(relm, relm))
        Robs = POINT[f"{m}|{s}"]["_R"]
        per_pair = {}
        for a, b in itertools.combinations(range(len(DS)), 2):
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
            x = dis[ia, ib]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_split = np.nanmean(x, 0)
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
S["disattenuated"] = DIS
print(f"  done {time.time() - T0:.0f}s", flush=True)

# ================================================== 3  hierarchical bootstrap
print("bootstrap", flush=True)


def v5_method_takes(pool, b, seed=0):
    """The method draws of stats_v5.boot (same RNG stream), for any pool."""
    nm = sum((GROUPS[g] for g in GROUP_SETS[pool]), [])
    pos = {n: k for k, n in enumerate(nm)}
    idx = [np.array([pos[n] for n in GROUPS[g]]) for g in GROUP_SETS[pool]]
    colmap = colsN(nm)
    r = np.random.default_rng(seed)
    return np.stack([colmap[np.concatenate([r.choice(i, len(i)) for i in idx])] for _ in range(b)])


rng = np.random.default_rng(SEED_HIER)
UMEANS = {m: np.zeros((B_HIER, len(DS), NC)) for m in METRICS}
for k, d in enumerate(DS):
    nu = UCNT[d].size
    draw = rng.integers(0, nu, size=(B_HIER, nu))
    W = np.zeros((B_HIER, nu))
    np.add.at(W, (np.repeat(np.arange(B_HIER), nu), draw.ravel()), 1.0)
    for m in METRICS:
        UMEANS[m][:, k, :] = (W @ USUM[m][d]) / (W @ UCNT[d])[:, None]


def gather(Mtensor, take):
    if Mtensor.ndim == 2:
        return Mtensor[:, take].transpose(1, 0, 2)
    return np.take_along_axis(Mtensor, take[:, None, :].repeat(Mtensor.shape[1], 1), axis=2)


def v5_interval(m, s, q):
    """Method-only interval reported in stats_v5.json (V6 pools only)."""
    if s not in V5_POOLS:
        return None, None, None
    h = V5["hier"].get(f"{m}|{s}", {})
    hk = {"modality_diff", "same dataset", "same source", "different source, same modality", "across modality"}
    if q in hk:
        e = h[q]
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


BOOT = {}
QK = ["modality_diff", "modality_diff_excl_within_source", "same dataset", "same source",
      "different source, same modality", "across modality", "wound pairs",
      "wound different-source pairs", "fuseg~azh", "colonoscopy pairs"] + \
     [f"comp|{d}|{k}" for d in COMP for k in COMP[d]]
for s in BOOT_POOLS:
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
            iv, src, _ = v5_interval(m, s, q)
            res[q] = {"est": POINT[f"{m}|{s}"][q],
                      "method_only_repro_B1000": ci(rh[q]), "method_only_repro_B2000": ci(rc[q]),
                      "units_only": ci(unit[q]), "hierarchical": ci(hier[q]),
                      "hier_sd": float(np.nanstd(hier[q])),
                      "hier_p_le0": float(np.nanmean(hier[q] <= 0)),
                      "v5_method_only": iv, "v5_source": src}
        BOOT[f"{m}|{s}"] = res
S["bootstrap"] = BOOT
print(f"  done {time.time() - T0:.0f}s", flush=True)

# ================================================== 4  modality test variants
print("exact tests", flush=True)
CODE = {"wound": 0, "colonoscopy": 1, "dermoscopy": 2, "ultrasound": 3}
_LCACHE = {}


def exact_test(R, keys, drop_pairs=()):
    pr = [(a, b) for a, b in itertools.combinations(keys, 2)
          if frozenset((REFS[a], REFS[b])) not in drop_pairs]
    ia = np.array([keys.index(a) for a, b in pr])
    ib = np.array([keys.index(b) for a, b in pr])
    r = np.array([R[a, b] for a, b in pr])
    lab = [CODE[LAB[k]] for k in keys]
    if tuple(lab) not in _LCACHE:
        _LCACHE[tuple(lab)] = np.array(list(multiset_permutations(lab)))
    L = _LCACHE[tuple(lab)]
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
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        R = POINT[f"{m}|{s}"]["_R"]
        P_ = POINT[f"{m}|{s}"]
        EX[f"{m}|{s}"] = {"full_55": exact_test(R, ALLK), "excl_within_source_52": exact_test(R, ALLK, DROP),
                          "levels_excl": {"different source, same modality (13)": P_["different source, same modality"],
                                          "across modality (39)": P_["across modality"],
                                          "same source (3, excluded)": P_["same source"],
                                          "same dataset (control)": P_["same dataset"]}}
S["exact"] = EX

LODO = {}
for m in METRICS:
    for s in LODO_SETS:
        R = POINT[f"{m}|{s}"]["_R"]
        LODO[f"{m}|{s}"] = {}
        for k in range(NR):
            keys = [x for x in ALLK if x != k]
            LODO[f"{m}|{s}"][REFS[k]] = {"full": exact_test(R, keys),
                                         "excl_within_source": exact_test(R, keys, DROP)}
S["leave_one_dataset_out"] = LODO

for g in LOMO_GROUPS:
    LOMO = {}
    for m in METRICS:
        mc = GCOLS[g]
        LOMO[m] = {}
        for k, n in enumerate(GROUPS[g]):
            c = np.delete(mc, k)
            P_ = summ(corrmat(MEANS[m][:, c]))
            LOMO[m][n] = {q: float(P_[q]) for q in ("modality_diff", "modality_diff_excl_within_source", "wound pairs")}
    S[f"leave_one_method_out_{g}"] = LOMO

# ================================================== 5  stratified agreement
print("stratified", flush=True)
STR = {}
for s in STRAT_SETS:
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
            var[g] = Wp[:, off:off + n].var(1)
            off += n
        Rs = corrmat(Wp)
        Zp = Wp - Wp.mean(1, keepdims=True)
        Zp /= np.sqrt((Zp ** 2).sum(1, keepdims=True))
        Rp = Zp @ Zp.T
        ns = {g: GCOLS[g].size for g in gs}
        Rw = sum(ns[g] * within[g] for g in gs) / sum(ns.values())
        num = sum(ns[g] * within[g] * np.sqrt(np.outer(var[g], var[g])) for g in gs)
        den = np.sqrt(np.outer(sum(ns[g] * var[g] for g in gs), sum(ns[g] * var[g] for g in gs)))
        Rf = num / den
        Ru = sum(within[g] for g in gs) / len(gs)
        Rraw = POINT[f"{m}|{s}"]["_R"]
        out = {}
        for lab_, R in (("stratified_spearman", Rs), ("pearson_on_percentiles", Rp),
                        ("weighted_identity", Rf), ("size_weighted_average", Rw),
                        ("unweighted_average", Ru), ("pooled_unstratified", Rraw)):
            P_ = summ(R)
            out[lab_] = {q: float(P_[q]) for q in ("wound pairs", "modality_diff", "colonoscopy pairs",
                                                   "modality_diff_excl_within_source")}
        out["max_abs_pair_diff_stratified_vs_weighted"] = float(np.max(np.abs(Rs[PI, PJ] - Rw[PI, PJ])))
        out["max_abs_pair_diff_pearsonpct_vs_identity"] = float(np.max(np.abs(Rp[PI, PJ] - Rf[PI, PJ])))
        if s in V5["stratified"]:
            out["v5_stratified_modality_diff"] = V5["stratified"][s][m]["modality_diff"]
            out["v5_stratified_wound_in_x_in"] = V5["stratified"][s][m]["categories_wound"]["in x in"]
        out["within_group"] = {g: {q: float(summ(within[g])[q]) for q in ("wound pairs", "modality_diff")} for g in gs}
        STR[f"{m}|{s}"] = out
S["stratified"] = STR

# ================================================== 6  empty-mask-excluded Dice
NOEMPTY = {}
for s in BOOT_POOLS:
    P_ = POINT[f"dice_noempty|{s}"]
    e = {"modality_diff": P_["modality_diff"], "p": EX[f"dice_noempty|{s}"]["full_55"]["p"],
         "dice_with_empty": POINT[f"dice|{s}"]["modality_diff"],
         "excl_within_source": P_["modality_diff_excl_within_source"]}
    if s in V5_POOLS:
        e["v5_est"] = V5["hier"][f"dice_noempty|{s}"]["modality_diff"]["est"]
        e["v5_p"] = V5["exact"][f"dice_noempty|{s}"]["modality"]["p"]
    NOEMPTY[s] = e
S["dice_noempty"] = NOEMPTY
S["n_empty"] = {d: int(DATA["dice"][d]["empty"].sum()) for d in DS}

# ================================================== 7  floor effect
FLOOR = {"mean_dice": {}, "share_dice_lt_0.05_median_over_methods": {}, "flag_lt_0.1": {}}
for g in FLOOR_GROUPS:
    FLOOR["mean_dice"][g] = {d: float(MEANS["dice"][k, GCOLS[g]].mean()) for k, d in enumerate(DS)}
    FLOOR["flag_lt_0.1"][g] = [d for d in DS if FLOOR["mean_dice"][g][d] < 0.1]
    FLOOR["share_dice_lt_0.05_median_over_methods"][g] = {
        d: float(np.median((DATA["dice"][d]["X"][:, GCOLS[g]] < 0.05).mean(0))) for d in DS}
S["floor"] = FLOOR

for k in POINT:
    POINT[k].pop("_R")
S["point"] = POINT

# ================================================== 0  validation against stats_v6.json
print("validating against results/v6/stats_v6.json", flush=True)
SKIP_TOP = {"definitions", "runtime_s", "check_point_estimates_vs_v5", "check_v5_intervals_reproduced",
            "check_v5_exact_reproduced"}
cmp = {"n_numeric": 0, "n_other": 0, "max_abs_diff": 0.0, "mismatches": [], "skipped": []}


def walk(a, b, path):
    if isinstance(a, dict):
        if not isinstance(b, dict):
            cmp["mismatches"].append(path + " (type)")
            return
        for k, v in a.items():
            if k not in b:
                cmp["mismatches"].append(f"{path}/{k} (missing)")
                continue
            walk(v, b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            cmp["mismatches"].append(path + " (list)")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(a, str) or a is None:
        cmp["n_other"] += 1
        if a != b:
            cmp["mismatches"].append(path)
    else:
        cmp["n_numeric"] += 1
        if b is None or isinstance(b, str):
            cmp["mismatches"].append(path)
            return
        if np.isnan(a) and np.isnan(b):
            return
        dlt = abs(a - b)
        cmp["max_abs_diff"] = max(cmp["max_abs_diff"], dlt)
        if dlt > 1e-12:
            cmp["mismatches"].append(f"{path} {a} vs {b}")


for top, val in V6J.items():
    if top in SKIP_TOP:
        cmp["skipped"].append(top)
        continue
    if top == "floor":
        for k, v in val.items():
            if k == "check_vs_v5_mean_dice_max_abs_diff":
                continue
            walk(v, S["floor"][k], f"floor/{k}")
        continue
    walk(val, S[top], top)
cmp["mismatches"] = cmp["mismatches"][:50]
S["check_reproduces_stats_v6"] = cmp
print("  numeric leaves compared", cmp["n_numeric"], "other", cmp["n_other"], "max abs diff", cmp["max_abs_diff"],
      "mismatches", len(cmp["mismatches"]), flush=True)
assert not cmp["mismatches"], cmp["mismatches"][:10]

S["runtime_s"] = round(time.time() - T0, 1)
json.dump(S, open(os.path.join(OUT, "stats_v7.json"), "w"), indent=1)
print("saved results/v7/stats_v7.json", f"{time.time() - T0:.0f}s", flush=True)

# ================================================================ markdown
L = []
w = L.append
NS = {s: v["n"] for s, v in S["definitions"]["method_sets"].items()}


def nm(s):
    return f"{s} (n {NS[s]})"


def bq(m, s, q):
    return BOOT[f"{m}|{s}"][q]


def iv2(x):
    return f"[{f3(x[0])}, {f3(x[1])}]"


w("# STATS V7: V6 agreement statistics with the phase-2 modern models\n")
w("Generated by `scripts/v7/stats_v7.py` from `scripts/perimage_v7.py`. Numbers in `results/v7/stats_v7.json`. "
  "V6 values are read from `results/v6/stats_v6.json`; for the V6 method sets this script recomputes them and they agree to the last digit (check below).\n")
w("## Definitions\n")
for k in ("agreement", "pairs", "modality_diff", "modality_diff_excl_within_source", "categories", "method_only", "engine"):
    w(f"- **{k}**: {S['definitions'][k]}")
w("- **groups**: " + "; ".join(f"{g} {len(v)}" for g, v in GROUPS.items()) +
  ". M19 = 18 seed-averaged pretrained architectures + nnU-Net (primary). M17 = M19 without manet_r18 and manet_r50. "
  "Ms0 / Ms1 / Ms2 = the architectures that have seed k, seed-k models only, + nnU-Net. M = V6 M11 (10 seed-0 architectures + nnU-Net).")
w("- **method sets**: " + ", ".join(nm(s) for s in GROUP_SETS))
w(f"- **draws**: unit splits {N_SPLIT} (seed {SEED_SPLIT}); hierarchical bootstrap B {B_HIER} (seed {SEED_HIER}; unit draws identical to V6; method draws in pool order "
  + ", ".join(BOOT_POOLS) + "); method-only bootstrap as stats_v5 (seed 0 per pool, B 1000 / 2000).")
w("- Units per dataset: " + ", ".join(f"{d} {S['n_units'][d]}/{S['n_images'][d]}" for d in DS) + " (units/images).\n")

w("## Checks\n")
w(f"- Engine vs results/v6/stats_v6.json: {cmp['n_numeric']} numeric and {cmp['n_other']} other leaves compared "
  f"(sections {', '.join(k for k in V6J if k not in SKIP_TOP)}), max abs diff {cmp['max_abs_diff']:.1e}, mismatches {len(cmp['mismatches'])}. "
  "This covers every M (M11) row: M11 reproduces V6 exactly.")
for m in METRICS:
    e = ind[m]
    w(f"- M19 {m}, independent route (scipy.stats.spearmanr on load7 means): wound pairs {e['wound pairs scipy']:.6f} vs engine {e['engine']:.6f}; "
      f"control {e['control scipy']:.6f} vs {e['control engine']:.6f}"
      + (f"; seeds_v7.py value {e['seeds_v7 M19 wound pairs']:.6f}." if "seeds_v7 M19 wound pairs" in e else "."))
w("- Per-seed groups Ms0, Ms1, Ms2: wound-pair means equal the 'seed k architectures + nnU-Net' values of seeds_v7.py (difference < 1e-12).\n")

# ---------------------------------------------------------------- plain summary
w("## What changes from V6 (plain reading)\n")
for m in METRICS:
    a6, a7 = V6J["bootstrap"][f"{m}|M"]["modality_diff"], bq(m, "M19", "modality_diff")
    e6, e7 = V6J["exact"][f"{m}|M"]["full_55"], EX[f"{m}|M19"]["full_55"]
    w(f"- {m}, modern group modality difference: M (n 11) {f3(a6['est'])}, exact p {fp(e6)}; M19 (n 19) {f3(a7['est'])}, hierarchical {iv2(a7['hierarchical'])}, exact p {fp(e7)}. "
      "Per-seed groups: " + "; ".join(f"Ms{k} (n {NS[f'Ms{k}']}) {f3(POINT[f'{m}|Ms{k}']['modality_diff'])}, p {fp(EX[f'{m}|Ms{k}']['full_55'])}" for k in P.SEEDS) +
      f". Colonoscopy-pair agreement: M {f3(V6J['point'][f'{m}|M']['colonoscopy pairs'])}, M19 {f3(POINT[f'{m}|M19']['colonoscopy pairs'])}, "
      + ", ".join(f"Ms{k} {f3(POINT[f'{m}|Ms{k}']['colonoscopy pairs'])}" for k in P.SEEDS) + ".")
    b6, b7 = V6J["bootstrap"][f"{m}|M"]["wound pairs"], bq(m, "M19", "wound pairs")
    w(f"- {m}, modern group wound-pair agreement: M {f3(b6['est'])} {iv2(b6['hierarchical'])}, M19 {f3(b7['est'])} {iv2(b7['hierarchical'])} (hierarchical); "
      + ", ".join(f"Ms{k} {f3(POINT[f'{m}|Ms{k}']['wound pairs'])}" for k in P.SEEDS) + ".")
    c6, c7 = V6J["bootstrap"][f"{m}|U+M"], BOOT[f"{m}|U+M19"]
    w(f"- {m}, mixture: U+M wound pairs {f3(c6['wound pairs']['est'])}, modality diff {f3(c6['modality_diff']['est'])}; "
      f"U+M19 wound pairs {f3(c7['wound pairs']['est'])} {iv2(c7['wound pairs']['hierarchical'])}, modality diff {f3(c7['modality_diff']['est'])} {iv2(c7['modality_diff']['hierarchical'])}, "
      f"exact p {fp(EX[f'{m}|U+M19']['full_55'])}. Stratified U+M19 wound pairs {f3(STR[f'{m}|U+M19']['stratified_spearman']['wound pairs'])} vs size-weighted within-group {f3(STR[f'{m}|U+M19']['size_weighted_average']['wound pairs'])}.")
w("")

# ---------------------------------------------------------------- headline
w("## Headline: V6 next to V7\n")
w("Intervals: method-only = stats_v5 design (B 2000 for wound pairs, B 1000 for modality diff); hierarchical = methods within group and units within dataset (B 1000). "
  "Exact p = relabeling test over the 13,860 distinct modality labelings of the 11 references (smallest attainable p 0.00014, marked (min)).\n")
w("### Wound-pair mean agreement (6 pairs)\n")
w("| metric | V6 set | V6 est | method-only | hierarchical | V7 set | V7 est | method-only | hierarchical |")
w("|---|---|---|---|---|---|---|---|---|")
for m in METRICS:
    for s6, s7 in (("M", "M19"), ("U+M", "U+M19")):
        a, b = V6J["bootstrap"][f"{m}|{s6}"]["wound pairs"], bq(m, s7, "wound pairs")
        w(f"| {m} | {nm(s6)} | {f3(a['est'])} | {iv2(a['method_only_repro_B2000'])} | {iv2(a['hierarchical'])} | "
          f"{nm(s7)} | {f3(b['est'])} | {iv2(b['method_only_repro_B2000'])} | {iv2(b['hierarchical'])} |")
w("\n### Modality difference (16 same-modality pairs minus 39 across-modality pairs)\n")
w("U, F, K are unchanged from V6 (same methods, same data); the V7 counterpart of M is M19 and of U+M is U+M19.\n")
w("| metric | V6 set | V6 est | method-only | hierarchical | exact p | V7 set | V7 est | method-only | hierarchical | exact p | excl within-source (52 pairs) diff, p |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in METRICS:
    for s6, s7 in (("U", "U"), ("F", "F"), ("K", "K"), ("M", "M19"), ("U+M", "U+M19")):
        a, b = V6J["bootstrap"][f"{m}|{s6}"]["modality_diff"], bq(m, s7, "modality_diff")
        ea, eb = V6J["exact"][f"{m}|{s6}"]["full_55"], EX[f"{m}|{s7}"]["full_55"]
        ex = EX[f"{m}|{s7}"]["excl_within_source_52"]
        w(f"| {m} | {nm(s6)} | {f3(a['est'])} | {iv2(a['method_only_repro_B1000'])} | {iv2(a['hierarchical'])} | {fp(ea)} | "
          f"{nm(s7)} | {f3(b['est'])} | {iv2(b['method_only_repro_B1000'])} | {iv2(b['hierarchical'])} | {fp(eb)} | {f3(ex['diff'])}, {fp(ex)} |")
w("\n### Reliability and disattenuated agreement, modern group\n")
w("Split-half reliability (Spearman-Brown, median [2.5, 97.5] over 500 unit splits) per dataset:\n")
w("| dataset | dice M (n 11) | dice M19 (n 19) | nsd M (n 11) | nsd M19 (n 19) |")
w("|---|---|---|---|---|")
for d in DS:
    cells = []
    for m in METRICS:
        for s, src in (("M", V6J["reliability"]), ("M19", REL)):
            r = src[f"{m}|{s}"][d]
            cells.append(fci(r["spearman_brown_median"], r["lo"], r["hi"]))
    w(f"| {d} | " + " | ".join(cells) + " |")
w("\nDisattenuated agreement (median over splits [2.5, 97.5 of the split distribution]; ratio of means):\n")
w("| metric | category | M observed | M disattenuated | M ratio of means | M19 observed | M19 disattenuated | M19 ratio of means |")
w("|---|---|---|---|---|---|---|---|")
for m in METRICS:
    for c in DIS[f"{m}|M19"]["summary"]:
        a, b = V6J["disattenuated"][f"{m}|M"]["summary"][c], DIS[f"{m}|M19"]["summary"][c]
        w(f"| {m} | {c} | {f3(a['observed_full_mean'])} | {fci(a['dis_median'], a['split_lo'], a['split_hi'])} | {f3(a['dis_ratio_of_means'])} | "
          f"{f3(b['observed_full_mean'])} | {fci(b['dis_median'], b['split_lo'], b['split_hi'])} | {f3(b['dis_ratio_of_means'])} |")
w("\n### Stratified vs size-weighted within-group agreement, U+M vs U+M19\n")
w("| metric | set | quantity | stratified Spearman | size-weighted within-group | unweighted | pooled unstratified | within U | within modern |")
w("|---|---|---|---|---|---|---|---|---|")
for m in METRICS:
    for s, src in (("U+M", V6J["stratified"]), ("U+M19", STR)):
        v = src[f"{m}|{s}"]
        g2 = GROUP_SETS[s][1]
        for q in ("wound pairs", "modality_diff"):
            w(f"| {m} | {nm(s)} | {q} | {f3(v['stratified_spearman'][q])} | {f3(v['size_weighted_average'][q])} | "
              f"{f3(v['unweighted_average'][q])} | {f3(v['pooled_unstratified'][q])} | {f3(v['within_group']['U'][q])} | {f3(v['within_group'][g2][q])} |")
w("\nMax abs per-pair difference, stratified vs size-weighted: " +
  ", ".join(f"{k} {v['max_abs_pair_diff_stratified_vs_weighted']:.3f}" for k, v in STR.items()) + ".\n")

# ---------------------------------------------------------------- categories
w("## Categories (stats_v5 f_comp, V5 Fig 1 data), wound domain\n")
w("Mean agreement over pairs: in x in = 6 wound pairs; in x out = 28 wound vs non-wound pairs; out, same modality = 10 colonoscopy pairs; "
  "out, across modality = 11 pairs among non-wound datasets of different modality. Intervals: method-only bootstrap with the stats_v5 design (B 2000, seed 0); hierarchical (B 1000).\n")
w("| metric | set | in x in | in x out | out, same modality | out, across modality |")
w("|---|---|---|---|---|---|")
for m in METRICS:
    for s in ("U", "M", "U+M", "M19", "U+M19", "M17", "U+M17", "Ms0", "Ms1", "Ms2"):
        src = V6J["bootstrap"] if s in V5_POOLS else BOOT
        cells = []
        for k in COMP["wound"]:
            b = src[f"{m}|{s}"][f"comp|wound|{k}"]
            cells.append(f"{f3(b['est'])} {iv2(b['method_only_repro_B2000'])}")
        w(f"| {m} | {nm(s)} | " + " | ".join(cells) + " |")
w("\nHierarchical intervals for the same cells:\n")
w("| metric | set | in x in | in x out | out, same modality | out, across modality |")
w("|---|---|---|---|---|---|")
for m in METRICS:
    for s in ("U", "M19", "U+M19"):
        w(f"| {m} | {nm(s)} | " + " | ".join(f"{f3(bq(m, s, 'comp|wound|' + k)['est'])} {iv2(bq(m, s, 'comp|wound|' + k)['hierarchical'])}" for k in COMP["wound"]) + " |")

# ---------------------------------------------------------------- sensitivity
w("\n## Sensitivity: modern group definitions\n")
w("| metric | set | wound pairs [method-only] [hier] | modality diff [method-only] [hier] | exact p | excl within-source diff, p | control azh~azh_train_slice |")
w("|---|---|---|---|---|---|---|")
for m in METRICS:
    for s in ("M", "M19", "M17", "Ms0", "Ms1", "Ms2", "U+M", "U+M19", "U+M17", "U+Ms0", "U+Ms1", "U+Ms2"):
        a, b = bq(m, s, "wound pairs"), bq(m, s, "modality_diff")
        e, ex = EX[f"{m}|{s}"]["full_55"], EX[f"{m}|{s}"]["excl_within_source_52"]
        w(f"| {m} | {nm(s)} | {f3(a['est'])} {iv2(a['method_only_repro_B2000'])} {iv2(a['hierarchical'])} | "
          f"{f3(b['est'])} {iv2(b['method_only_repro_B1000'])} {iv2(b['hierarchical'])} | {fp(e)} | {f3(ex['diff'])}, {fp(ex)} | {f3(POINT[f'{m}|{s}']['same dataset'])} |")
w("\nPer-pair wound agreement:\n")
WPN = [PNAME[k] for k in np.where(WOUNDP)[0]]
w("| metric | set | " + " | ".join(WPN) + " |")
w("|---|---|" + "---|" * len(WPN))
for m in METRICS:
    for s in ("M", "M19", "M17", "Ms0", "Ms1", "Ms2"):
        pp = DIS[f"{m}|{s}"]["pairs"]
        w(f"| {m} | {nm(s)} | " + " | ".join(f3(pp[p]["observed_full"]) for p in WPN) + " |")

# ---------------------------------------------------------------- full sections
w("\n## 1. Split-half reliability, all V7 sets (Spearman-Brown median over 500 unit splits)\n")
for m in METRICS:
    w(f"\n### {m}\n")
    sets = ["M19", "M17", "Ms0", "Ms1", "Ms2", "U+M19", "ALL65"]
    w("| dataset | " + " | ".join(nm(s) for s in sets) + " |")
    w("|---" * (len(sets) + 1) + "|")
    for d in DS:
        w(f"| {d} | " + " | ".join(fci(REL[f'{m}|{s}'][d]['spearman_brown_median'], REL[f'{m}|{s}'][d]['lo'], REL[f'{m}|{s}'][d]['hi']) for s in sets) + " |")

w("\n## 2. Disattenuated agreement, V7 sets\n")
w("Definitions as V6: per split r_AB = mean(corr(A1,B1), corr(A2,B2)), r_AA = corr(A1,A2), dis = r_AB / sqrt(r_AA r_BB), undefined if r_AA r_BB <= 0. "
  "Median over splits [2.5, 97.5] is split-to-split spread, not a sampling interval. Ratio of means = mean(r_AB)/sqrt(mean r_AA mean r_BB).\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | category | observed | half-level r_AB | disattenuated mean | median [2.5, 97.5] | ratio of means | share >1 | undefined |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in V7_SETS:
        for c, v in DIS[f"{m}|{s}"]["summary"].items():
            w(f"| {s} | {c} | {f3(v['observed_full_mean'])} | {f3(v['half_r_ab_mean'])} | {f3(v['dis_mean'])} | "
              f"{fci(v['dis_median'], v['split_lo'], v['split_hi'])} | {f3(v['dis_ratio_of_means'])} | "
              f"{v['share_split_values_gt1']:.2f} | {v['n_undefined_pair_splits']}/{v['n_pair_splits']} |")

w("\n## 3. Hierarchical bootstrap, V7 pools\n")
SHOW = ["modality_diff", "wound pairs", "same dataset", "modality_diff_excl_within_source"]
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | quantity | est | method-only (B 1000 / 2000) | units-only | hierarchical | P(hier <= 0) |")
    w("|---|---|---|---|---|---|---|")
    for s in BOOT_POOLS[len(V5_POOLS):]:
        for q in SHOW:
            b = bq(m, s, q)
            mo = b["method_only_repro_B2000"] if q == "wound pairs" else b["method_only_repro_B1000"]
            w(f"| {s} | {q} | {f3(b['est'])} | {iv2(mo)} | {iv2(b['units_only'])} | {iv2(b['hierarchical'])} | {b['hier_p_le0']:.3f} |")

w("\n## 4. Modality test, all sets\n")
w("| metric | set | diff (55 pairs) | p | diff (52 pairs) | p | diff-source same-modality mean | across mean | same-source mean | control |")
w("|---|---|---|---|---|---|---|---|---|---|")
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        e = EX[f"{m}|{s}"]
        lv = e["levels_excl"]
        w(f"| {m} | {nm(s)} | {f3(e['full_55']['diff'])} | {fp(e['full_55'])} | {f3(e['excl_within_source_52']['diff'])} | "
          f"{fp(e['excl_within_source_52'])} | {f3(lv['different source, same modality (13)'])} | {f3(lv['across modality (39)'])} | "
          f"{f3(lv['same source (3, excluded)'])} | {f3(lv['same dataset (control)'])} |")
w("\n### Leave one dataset out, M19 and U+M19 (modality difference, exact p)\n")
w("| metric | set | dropped | diff | p | diff excl within-source | p |")
w("|---|---|---|---|---|---|---|")
for k, v in LODO.items():
    m, s = k.split("|")
    if s not in ("M19", "U+M19"):
        continue
    for d, r in v.items():
        w(f"| {m} | {s} | {d} | {f3(r['full']['diff'])} | {fp(r['full'])} | {f3(r['excl_within_source']['diff'])} | {fp(r['excl_within_source'])} |")
for s in ("M", "M19", "U+M", "U+M19"):
    for m in METRICS:
        ps = [r["full"]["p"] for r in LODO[f"{m}|{s}"].values()]
        ds_ = [r["full"]["diff"] for r in LODO[f"{m}|{s}"].values()]
        w(f"\nLODO range {m} {s}: diff {f3(min(ds_))} to {f3(max(ds_))}, p {min(ps):.5f} to {max(ps):.5f} (11 drops).")
w("\n### Leave one method out, M19\n")
w("| metric | dropped | modality diff | excl within-source | wound pairs |")
w("|---|---|---|---|---|")
for m in METRICS:
    lo = S["leave_one_method_out_M19"][m]
    for n, r in lo.items():
        w(f"| {m} | {n} | {f3(r['modality_diff'])} | {f3(r['modality_diff_excl_within_source'])} | {f3(r['wound pairs'])} |")
    arr = np.array([[r["modality_diff"], r["wound pairs"]] for r in lo.values()])
    w(f"| {m} | range | {f3(arr[:, 0].min())} to {f3(arr[:, 0].max())} | | {f3(arr[:, 1].min())} to {f3(arr[:, 1].max())} |")

w("\n## 5. Stratified agreement, all mixtures\n")
w("| metric | set | quantity | stratified Spearman | Pearson on percentiles | exact weighted identity | size-weighted average | unweighted average | pooled unstratified |")
w("|---|---|---|---|---|---|---|---|---|")
for k, v in STR.items():
    m, s = k.split("|")
    for q in ("wound pairs", "modality_diff"):
        w(f"| {m} | {s} | {q} | {f3(v['stratified_spearman'][q])} | {f3(v['pearson_on_percentiles'][q])} | "
          f"{f3(v['weighted_identity'][q])} | {f3(v['size_weighted_average'][q])} | {f3(v['unweighted_average'][q])} | {f3(v['pooled_unstratified'][q])} |")
w(f"\nPearson on percentiles vs exact identity: max abs per-pair difference {max(v['max_abs_pair_diff_pearsonpct_vs_identity'] for v in STR.values()):.1e}.\n")

w("## 6. Empty-mask-excluded Dice modality difference\n")
w("| set | diff (no empty) | exact p | diff with empties | no-empty excl within-source |")
w("|---|---|---|---|---|")
for s, v in NOEMPTY.items():
    w(f"| {nm(s)} | {f3(v['modality_diff'])} | {fp(EX['dice_noempty|' + s]['full_55'])} | {f3(v['dice_with_empty'])} | {f3(v['excl_within_source'])} |")

w("\n## 7. Floor effect: group mean Dice (bold = below 0.1)\n")
w("| dataset | " + " | ".join(FLOOR_GROUPS) + " |")
w("|---" * (len(FLOOR_GROUPS) + 1) + "|")
for d in DS:
    cells = []
    for g in FLOOR_GROUPS:
        x = FLOOR["mean_dice"][g][d]
        cells.append(f"**{x:.3f}**" if x < 0.1 else f"{x:.3f}")
    w(f"| {d} | " + " | ".join(cells) + " |")
w("\nShare of images with Dice < 0.05 (median over the group's methods), non-wound datasets:\n")
w("| dataset | " + " | ".join(FLOOR_GROUPS) + " |")
w("|---" * (len(FLOOR_GROUPS) + 1) + "|")
for d in DS:
    if d in V.WOUND or d == V.CONTROL:
        continue
    w(f"| {d} | " + " | ".join(f"{FLOOR['share_dice_lt_0.05_median_over_methods'][g][d]:.2f}" for g in FLOOR_GROUPS) + " |")
w(f"\nRuntime {S['runtime_s']} s.\n")
open(os.path.join(OUT, "STATS_V7.md"), "w").write("\n".join(L))
print("saved results/v7/STATS_V7.md", f"{time.time() - T0:.0f}s")

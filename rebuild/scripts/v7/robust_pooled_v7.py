#!/usr/bin/env python3
"""V7 robustness of the POOLED-set results (composition effect, modality
difference) to post-processing, NSD tolerance and resolution.

robust_v7.py covers only the Kaggle-trained modern sets M19 and K11 (rescore
files results/kaggle2/rescore). The 32 local methods (18 untrained U, 14 from
scratch on FUSeg F) were rescored in the same layout by
scripts/v7/rescore_local_v7.py (results/v7/rescore_local/<set>.json), so the
pooled sets U+M19, U+F and U+K11 can now be analysed under every variant.

Writes results/v7/robust_pooled_v7.json and results/v7/ROBUST_POOLED_V7.md
(the md is generated from the json values; nothing is typed by hand). Reads
only. Run from the rebuild folder:
    nice -n 10 python scripts/v7/robust_pooled_v7.py

Data
  local rescore  results/v7/rescore_local/<set>.json: names (perimage32 order),
                 var[image][method][variant][metric], variants P0 P1 P2,
                 metrics dice nsd1 nsd2 nsd3 nsd5 nsd_diag1, unrounded, no
                 exclusions applied. Untrained (binary-mask) methods: P1 is a
                 copy of P0 by definition (checked here, exactly).
  Kaggle rescore results/kaggle2/rescore/<family>/<set>.json, same var layout,
                 values rounded to 5 decimals (robust_v7.py docstring).
  Exclusions     results/exclusions_v5.json by image index (perimage_v6.EXCL);
                 fuseg_r256 uses the fuseg exclusions (same images, same order).

Method groups
  U    18 untrained classical methods, F 14 trained from scratch on FUSeg:
       split by woundbench.combined.ALL_METHOD_FAMILY exactly as
       perimage_v6.groups() (checked equal to perimage_v6.GROUPS U and F).
  M19  perimage_v7.GROUPS['M19']: 18 ImageNet-pretrained architectures, each
       averaged per image over the seeds that exist (perimage_v7.SEEDCOLS),
       plus nnunet_2d. Same construction as robust_v7.py.
  K11  perimage_v7.GROUPS['K'] minus the 3 Kvasir pixel classifiers, which were
       not rescored (as robust_v7.py).
  Sets U, F, M19, K11 (single groups) and U+M19, U+F, U+K11 (pooled).

Quantities (stats_v7.py definitions; 11 references, 55 pairs)
  agreement           Spearman (Pearson on average ranks) of per-method mean
                      scores between two datasets over the method set
  wound pairs         mean of the 6 pairs among medetec, co2wounds, fuseg, azh
  wound vs other      mean of the 28 pairs wound x non-wound reference
                      (stats_v7 comp|wound|in x out)
  colonoscopy pairs / colonoscopy vs other: same for the colonoscopy domain
                      (10 and 30 pairs)
  modality difference mean of 16 same-modality pairs minus mean of 39
                      across-modality pairs
  stratified          rank methods within each group first, percentiles
                      (rank - 0.5) / n_g, then Pearson on the pooled
                      percentiles (stats_v7 section 7 "pearson_on_percentiles",
                      equal to the exact weighted identity)
  size-weighted within-group average: sum_g n_g R_g / N (stats_v7)

Method bootstrap: stats_v5 design as stats_v7.v5_method_takes: per draw,
methods resampled with replacement within each group, groups in set order,
one numpy default_rng(0) stream per set, B 2000, 95% percentile interval (nan
ignored). The same draws are used for every condition of a set, so changes
between conditions have paired intervals. Within-group and stratified values
in a pooled draw use the group parts of that same draw. The engine is checked
against results/v7/stats_v7.json with run-1 scores (point estimates, B 2000
method-only intervals, stratified point values) and against
results/v7/robust_v7.json for M19 and K11 (every summary robust_v7 reports).

Exact relabeling test of the modality difference: copy of stats_v7.exact_test
(stats_v7.py computes at import time, so it is not imported): all 13,860
distinct assignments of the 11 modality labels; p = share of relabelings with a
difference >= observed (1e-12 slack). Checked against stats_v7.json.

Composition criteria (per pooled set and condition; domain = wound):
  C1 pooled wound-pair agreement above the higher of the two within-group
     values, with the paired 95% interval of (pooled minus higher within) above 0
  C2 pooled wound-vs-other agreement below 0, 95% interval below 0
  C3 stratified wound-pair agreement within STRAT_TOL of the size-weighted
     within-group average (point values)
  The effect "holds" in a condition when C1, C2 and C3 all hold. The same
  criteria in the colonoscopy domain are reported as the mirror view.
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
import perimage_v6 as V6                                           # noqa: E402
import perimage_v7 as P                                            # noqa: E402

T0 = time.time()
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "v7")
RS = os.path.join(RES, "kaggle2", "rescore")
RL = os.path.join(OUT, "rescore_local")
FAMS = ["pretrained_s0", "pretrained_s1", "pretrained_s2", "modern2_s0",
        "modern2_s1", "modern2_s2", "kvasir_nets", "kvasir_unet", "nnunet"]
VARIANTS = ["P0", "P1", "P2"]
METRICS = ["dice", "nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1"]
REFS = list(V6.REFS)
CONTROL = V6.CONTROL
DS = REFS + [CONTROL]
NR = len(REFS)
LAB = [V6.MODALITY[r] for r in REFS]
B = 2000
SEED_METHOD = 0
TOL_P32 = 1e-6
STRAT_TOL = 0.05
WOUND = [r for r in REFS if V6.MODALITY[r] == "wound"]


def jload(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------- pairs (stats_v7)
PI, PJ = map(np.array, zip(*itertools.combinations(range(NR), 2)))
SAME = np.array([LAB[i] == LAB[j] for i, j in zip(PI, PJ)])
ACROSS = ~SAME
FA = np.array([{REFS[i], REFS[j]} == {"fuseg", "azh"} for i, j in zip(PI, PJ)])
I_AZH, I_CTRL = DS.index("azh"), DS.index(CONTROL)


def comp_masks(domain):
    dom = np.array([LAB[i] == domain for i in range(NR)])
    ia, ib = dom[PI], dom[PJ]
    return {"in x in": ia & ib, "in x out": ia != ib}


CW, CC = comp_masks("wound"), comp_masks("colonoscopy")
assert SAME.sum() == 16 and ACROSS.sum() == 39 and FA.sum() == 1
assert CW["in x in"].sum() == 6 and CW["in x out"].sum() == 28
assert CC["in x in"].sum() == 10 and CC["in x out"].sum() == 30

QK = ["wound pairs", "wound vs other", "colonoscopy pairs", "colonoscopy vs other",
      "modality_diff", "same modality (16)", "across modality (39)", "fuseg~azh",
      "control azh~azh_train_slice"]
SV7_KEY = {"wound pairs": "wound pairs", "wound vs other": "comp|wound|in x out",
           "colonoscopy pairs": "colonoscopy pairs", "colonoscopy vs other": "comp|colonoscopy|in x out",
           "modality_diff": "modality_diff", "same modality (16)": "same modality (16 pairs)",
           "across modality (39)": "across modality", "fuseg~azh": "fuseg~azh",
           "control azh~azh_train_slice": "same dataset"}


def nmean(A, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(A[..., mask], axis=-1)


def summ(R):
    rp = R[..., PI, PJ]
    return {"wound pairs": nmean(rp, CW["in x in"]),
            "wound vs other": nmean(rp, CW["in x out"]),
            "colonoscopy pairs": nmean(rp, CC["in x in"]),
            "colonoscopy vs other": nmean(rp, CC["in x out"]),
            "modality_diff": nmean(rp, SAME) - nmean(rp, ACROSS),
            "same modality (16)": nmean(rp, SAME),
            "across modality (39)": nmean(rp, ACROSS),
            "fuseg~azh": rp[..., np.where(FA)[0][0]],
            "control azh~azh_train_slice": R[..., I_AZH, I_CTRL]}


def zr(T):
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def corrmat(T):
    """Spearman matrix over the dataset axis; T[..., datasets, methods]."""
    Z = zr(T)
    return np.einsum("...in,...jn->...ij", Z, Z)


def pearsonmat(W):
    Z = W - W.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = Z / np.sqrt((Z ** 2).sum(-1, keepdims=True))
    return np.einsum("...in,...jn->...ij", Z, Z)


def strat_parts(T, slices):
    """Stratified (Pearson on within-group percentiles) and size-weighted
    within-group correlation matrices, plus each group's Spearman matrix."""
    pct, within = [], []
    for sl in slices:
        n = sl.stop - sl.start
        pct.append((rankdata(T[..., sl], axis=-1) - 0.5) / n)
        within.append(corrmat(T[..., sl]))
    N = sum(sl.stop - sl.start for sl in slices)
    Rs = pearsonmat(np.concatenate(pct, -1))
    Rw = sum((sl.stop - sl.start) * Rg for sl, Rg in zip(slices, within)) / N
    return Rs, Rw, within


def ci(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).any():
        return [None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def v5_takes(sizes, b=B, seed=SEED_METHOD):
    """stats_v7.v5_method_takes for groups of the given sizes, in order
    (positions in the concatenated method list)."""
    r = np.random.default_rng(seed)
    offs = np.cumsum([0] + list(sizes))
    idx = [np.arange(offs[k], offs[k + 1]) for k in range(len(sizes))]
    return np.stack([np.concatenate([r.choice(i, len(i)) for i in idx]) for _ in range(b)])


# ------------------------------------------------------------- exact test (stats_v7)
CODE = {"wound": 0, "colonoscopy": 1, "dermoscopy": 2, "ultrasound": 3}
_LAB = [CODE[x] for x in LAB]
PERMS = np.array(list(multiset_permutations(_LAB)))
assert len(PERMS) == 13860
_SAMEL = PERMS[:, PI] == PERMS[:, PJ]
_OBS_SAME = np.array(_LAB)[PI] == np.array(_LAB)[PJ]
N_EQUIV = int((_SAMEL == _OBS_SAME).all(1).sum())


def exact_test(R):
    """stats_v7.exact_test(R, ALLK) with no dropped pairs."""
    r = R[PI, PJ]
    stat = (_SAMEL * r).sum(1) / _SAMEL.sum(1) - ((~_SAMEL) * r).sum(1) / (~_SAMEL).sum(1)
    obs = r[_OBS_SAME].mean() - r[~_OBS_SAME].mean()
    return {"diff": float(obs), "p": float(np.mean(stat >= obs - 1e-12)),
            "n_relabelings": int(len(PERMS)), "min_p": N_EQUIV / len(PERMS)}


# ------------------------------------------------------------- groups
LOCAL_NAMES = V6._local_names()
GROUPS = {"U": [n for n in LOCAL_NAMES if not V6._trained(n)],
          "F": [n for n in LOCAL_NAMES if V6._trained(n)]}
assert GROUPS["U"] == V6.GROUPS["U"] and GROUPS["F"] == V6.GROUPS["F"]
assert len(GROUPS["U"]) == 18 and len(GROUPS["F"]) == 14
RESCORE_NAMES = set()
for f in FAMS:
    RESCORE_NAMES |= set(jload(os.path.join(RS, f, "medetec.json"))["names"])
GROUPS["M19"] = list(P.GROUPS["M19"])
GROUPS["K11"] = [n for n in P.GROUPS["K"] if n in RESCORE_NAMES]
K_MISSING = [n for n in P.GROUPS["K"] if n not in RESCORE_NAMES]
assert K_MISSING == ["logreg_all_kv", "gbm_all_kv", "logreg_colour_kv"], K_MISSING
assert len(GROUPS["M19"]) == 19 and len(GROUPS["K11"]) == 11
MEMBERS = {}
for g in ("M19", "K11"):
    for c in GROUPS[g]:
        MEMBERS[c] = list(P.SEEDCOLS[c]) if c in P.SEEDCOLS else [c]
        assert all(x in RESCORE_NAMES for x in MEMBERS[c]), c
for g in ("U", "F"):
    for c in GROUPS[g]:
        MEMBERS[c] = [c]

SETS = {"U": ["U"], "F": ["F"], "M19": ["M19"], "K11": ["K11"],
        "U+M19": ["U", "M19"], "U+F": ["U", "F"], "U+K11": ["U", "K11"]}
POOLED = ["U+M19", "U+F", "U+K11"]
SINGLE = ["U", "F", "M19", "K11"]
TAKES = {s: v5_takes([len(GROUPS[g]) for g in gs]) for s, gs in SETS.items()}


def slices_of(s):
    out, o = [], 0
    for g in SETS[s]:
        n = len(GROUPS[g])
        out.append(slice(o, o + n))
        o += n
    return out


CHECKS = {}

# ------------------------------------------------------------- engine check, run-1 data
print("engine check against stats_v7.json (run-1 scores)", flush=True)
SV7 = jload(os.path.join(OUT, "stats_v7.json"))
D7 = {m: {d: P.load(d, m) for d in DS} for m in ("dice", "nsd")}
ENG_SETS = {"U": ["U"], "F": ["F"], "M19": ["M19"], "U+F": ["U", "F"],
            "U+M19": ["U", "M19"], "U+K": ["U", "K"]}
eng = {"max_abs_diff_point": 0.0, "max_abs_diff_ci_B2000": 0.0, "max_abs_diff_stratified": 0.0,
       "max_abs_diff_exact_p": 0.0, "n_compared": 0}
for s, gs in ENG_SETS.items():
    names = sum((list(P.GROUPS[g]) for g in gs), [])
    tk = v5_takes([len(P.GROUPS[g]) for g in gs])
    for m in ("dice", "nsd"):
        M = np.stack([D7[m][d]["X"][:, P.cols(names)].mean(0) for d in DS])
        pt = summ(corrmat(M))
        bs = summ(corrmat(M[:, tk].transpose(1, 0, 2)))
        ref = SV7["bootstrap"][f"{m}|{s}"]
        for q in QK:
            sk = SV7_KEY[q]
            if sk not in ref:            # stats_v7 bootstraps every key except the 16-pair mean
                eng["max_abs_diff_point"] = max(eng["max_abs_diff_point"],
                                                abs(float(pt[q]) - SV7["point"][f"{m}|{s}"][sk]))
                continue
            eng["max_abs_diff_point"] = max(eng["max_abs_diff_point"], abs(float(pt[q]) - ref[sk]["est"]))
            eng["max_abs_diff_ci_B2000"] = max(eng["max_abs_diff_ci_B2000"], max(
                abs(a - b) for a, b in zip(ci(bs[q]), ref[sk]["method_only_repro_B2000"])))
            eng["n_compared"] += 1
        ex = exact_test(corrmat(M))
        r = SV7["exact"][f"{m}|{s}"]["full_55"]
        eng["max_abs_diff_exact_p"] = max(eng["max_abs_diff_exact_p"], abs(ex["p"] - r["p"]), abs(ex["diff"] - r["diff"]))
        if len(gs) > 1:
            off, sl = 0, []
            for g in gs:
                sl.append(slice(off, off + len(P.GROUPS[g])))
                off += len(P.GROUPS[g])
            Rs, Rw, _ = strat_parts(M, sl)
            st = SV7["stratified"][f"{m}|{s}"]
            for q in ("wound pairs", "modality_diff", "colonoscopy pairs"):
                eng["max_abs_diff_stratified"] = max(
                    eng["max_abs_diff_stratified"],
                    abs(float(summ(Rs)[q]) - st["pearson_on_percentiles"][q]),
                    abs(float(summ(Rw)[q]) - st["size_weighted_average"][q]))
eng["sets"] = list(ENG_SETS)
eng["what"] = ("run-1 scores via perimage_v7.load; point summaries and method-only B 2000 intervals vs "
               "stats_v7.json bootstrap[...]['method_only_repro_B2000'] (keys " + ", ".join(SV7_KEY.values()) +
               "); exact p and difference vs stats_v7.json exact[...]['full_55']; stratified "
               "pearson_on_percentiles and size_weighted_average for U+F, U+M19, U+K")
for k in ("max_abs_diff_point", "max_abs_diff_ci_B2000", "max_abs_diff_stratified", "max_abs_diff_exact_p"):
    assert eng[k] < 1e-12, (k, eng[k])
CHECKS["engine_vs_stats_v7_run1"] = eng
print("  ", {k: v for k, v in eng.items() if k.startswith("max")}, flush=True)

# ------------------------------------------------------------- data loading
print("loading rescore files", flush=True)
P32 = {d: jload(os.path.join(RES, "perimage32", d + ".json")) for d in DS}
RSETS = DS + ["fuseg_r256"]


def keep_mask(s, n):
    k = np.ones(n, bool)
    for e in V6.EXCL.get("fuseg" if s == "fuseg_r256" else s, []):
        k[e[0]] = False
    return k


VAR = {}            # set -> {method: [n_kept, 3, 6]}
NKEPT = {}
p32chk = {}
# The one cell where the local rescore P0 differs from perimage32 by more than
# 1e-6 (recorded as pass_1e-6 False in the rescore file's own check and in
# rescore_local/run.log): segformer_mitb0 on cvc_clinicdb image index 1
# (Dice 9.0e-6, NSD 2 px 4.0e-5; a re-run of the network inference).
KNOWN_P0_DEVIATIONS = {("cvc_clinicdb", "segformer_mitb0", 1)}
P0DEV = []
P32VAL = {}
p1chk = {"max_abs_P1_minus_P0_untrained": 0.0, "untrained_methods": len(GROUPS["U"]), "sets": len(RSETS)}
for s in RSETS:
    base = "fuseg" if s == "fuseg_r256" else s
    ref = P32[base]
    n = len(ref["ids"])
    keep = keep_mask(s, n)
    VAR[s] = {}
    j = jload(os.path.join(RL, s + ".json"))
    assert j["set"] == s and j["variants"] == VARIANTS and j["metrics"] == METRICS, s
    assert j["names"] == LOCAL_NAMES, s
    assert j["n"] == n and j["ids"] == ref["ids"] and j["gt_empty"] == ref["gt_empty"], s
    assert j["exclusions_applied"] is False, s
    if s != "fuseg_r256":
        assert np.allclose(j["gt_fraction"], ref["gt_fraction"], atol=1e-9), s
    kinds = dict(zip(j["names"], j["method_kind"]))
    assert all(kinds[x] == "mask" for x in GROUPS["U"]) and all(kinds[x] == "prob" for x in GROUPS["F"]), s
    Vv = np.asarray(j["var"], float)
    assert Vv.shape == (n, 32, 3, 6) and np.isfinite(Vv).all(), (s, Vv.shape)
    # untrained methods: P1 is P0 by definition
    for k, nm in enumerate(j["names"]):
        if nm in GROUPS["U"]:
            p1chk["max_abs_P1_minus_P0_untrained"] = max(
                p1chk["max_abs_P1_minus_P0_untrained"], float(np.abs(Vv[:, k, 1] - Vv[:, k, 0]).max()))
    # P0 dice and nsd2 vs perimage32, every image (before exclusions). Any cell
    # above TOL_P32 stops the script unless it is in KNOWN_P0_DEVIATIONS; a
    # known cell must then pass the materiality check below (method ranks on
    # that set unchanged when the perimage32 value is put back).
    if s != "fuseg_r256":
        assert ref["nsd_tau"] == 2.0
        Dd = np.abs(Vv[:, :, 0, 0] - np.asarray(ref["dice"], float))
        Dn = np.abs(Vv[:, :, 0, 2] - np.asarray(ref["nsd"], float))
        bad = {(s, j["names"][k], int(i)) for i, k in np.argwhere((Dd > TOL_P32) | (Dn > TOL_P32))}
        assert bad <= KNOWN_P0_DEVIATIONS, ("P0 differs from perimage32 beyond tolerance", sorted(bad - KNOWN_P0_DEVIATIONS))
        ok = np.ones(Dd.shape, bool)
        for (_, nm, i) in bad:
            ok[i, j["names"].index(nm)] = False
            P0DEV.append({"set": s, "method": nm, "image_index": i, "image_id": j["ids"][i],
                          "excluded": not keep[i],
                          "dice_rescore": float(Vv[i, j["names"].index(nm), 0, 0]),
                          "dice_perimage32": float(ref["dice"][i][j["names"].index(nm)]),
                          "nsd2_rescore": float(Vv[i, j["names"].index(nm), 0, 2]),
                          "nsd2_perimage32": float(ref["nsd"][i][j["names"].index(nm)])})
        p32chk[s] = {"max_abs_diff_dice": float(Dd.max()), "max_abs_diff_nsd2": float(Dn.max()),
                     "max_abs_diff_dice_other_cells": float(Dd[ok].max()),
                     "max_abs_diff_nsd2_other_cells": float(Dn[ok].max()),
                     "n_cells_above_tol": len(bad), "n_images": n}
        P32VAL[s] = {"dice": np.asarray(ref["dice"], float)[keep], "nsd2": np.asarray(ref["nsd"], float)[keep]}
    for k, nm in enumerate(j["names"]):
        VAR[s][nm] = Vv[keep, k]
    for f in FAMS:
        jj = jload(os.path.join(RS, f, s + ".json"))
        assert jj["family"] == f and jj["set"] == s and jj["variants"] == VARIANTS and jj["metrics"] == METRICS
        assert jj["n"] == n and jj["gt_empty"] == ref["gt_empty"], (f, s)
        if s != "fuseg_r256":
            assert np.allclose(jj["gt_fraction"], ref["gt_fraction"], atol=1e-9), (f, s)
        W = np.asarray(jj["var"], float)
        assert W.shape == (n, len(jj["names"]), 3, 6) and np.isfinite(W).all()
        for k, nm in enumerate(jj["names"]):
            assert nm not in VAR[s], nm
            VAR[s][nm] = W[keep, k]
    NKEPT[s] = int(keep.sum())
assert p1chk["max_abs_P1_minus_P0_untrained"] == 0.0, p1chk
# materiality of the known deviations: with the perimage32 value put back, the
# order of the 32 local method means (P0 Dice and NSD 2 px) on that set is
# unchanged, so every rank-based quantity here (Spearman agreement, within-group
# percentiles, bootstrap draws, exact test) is unchanged for every method set.
assert {(d["set"], d["method"], d["image_index"]) for d in P0DEV} == KNOWN_P0_DEVIATIONS, P0DEV
mat = []
for dv in P0DEV:
    s = dv["set"]
    for mi, key in ((0, "dice"), (2, "nsd2")):
        a = np.array([VAR[s][nm][:, 0, mi].mean() for nm in LOCAL_NAMES])
        b = P32VAL[s][key].mean(0)
        same_order = bool(np.array_equal(rankdata(a), rankdata(b)))
        k = LOCAL_NAMES.index(dv["method"])
        gap = float(np.min(np.abs(np.delete(a, k) - a[k])))
        mat.append({"set": s, "metric": key, "method_mean_shift": float(a[k] - b[k]),
                    "nearest_other_local_method_mean_gap": gap, "rank_order_unchanged": same_order})
        assert same_order, mat[-1]
CHECKS["local_P0_vs_perimage32"] = {"tolerance": TOL_P32, "per_set": p32chk,
                                    "max_abs_diff_dice": max(v["max_abs_diff_dice"] for v in p32chk.values()),
                                    "max_abs_diff_nsd2": max(v["max_abs_diff_nsd2"] for v in p32chk.values()),
                                    "max_abs_diff_dice_other_cells": max(v["max_abs_diff_dice_other_cells"] for v in p32chk.values()),
                                    "max_abs_diff_nsd2_other_cells": max(v["max_abs_diff_nsd2_other_cells"] for v in p32chk.values()),
                                    "n_cells_compared_per_metric": int(sum(v["n_images"] for v in p32chk.values()) * 32),
                                    "cells_above_tolerance": P0DEV, "materiality": mat}
CHECKS["untrained_P1_equals_P0"] = p1chk
CHECKS["n_kept"] = NKEPT
print("  local P0 vs perimage32: dice", CHECKS["local_P0_vs_perimage32"]["max_abs_diff_dice"],
      "nsd2", CHECKS["local_P0_vs_perimage32"]["max_abs_diff_nsd2"], "; P1==P0 untrained", flush=True)


def col_mean(s, c, vi, mi):
    return float(np.mean(np.stack([VAR[s][x][:, vi, mi] for x in MEMBERS[c]], 1), 1).mean())


def set_means(st, vi, mi, r256=False):
    """[12 datasets (DS order), N methods of set st]; fuseg row replaced by
    fuseg_r256 when r256."""
    names = sum((GROUPS[g] for g in SETS[st]), [])
    rows = []
    for d in DS:
        src = "fuseg_r256" if (r256 and d == "fuseg") else d
        rows.append([col_mean(src, c, vi, mi) for c in names])
    return np.array(rows)


# ------------------------------------------------------------- conditions
CONDS = {  # key: (family, variant index, metric index, fuseg replaced by r256)
    "dice|P0": ("main", 0, 0, False),
    "dice|P1": ("post-processing", 1, 0, False),
    "dice|P2": ("post-processing", 2, 0, False),
    "nsd2|P0": ("main", 0, 2, False),
    "nsd2|P1": ("post-processing", 1, 2, False),
    "nsd2|P2": ("post-processing", 2, 2, False),
    "nsd1|P0": ("tolerance", 0, 1, False),
    "nsd3|P0": ("tolerance", 0, 3, False),
    "nsd5|P0": ("tolerance", 0, 4, False),
    "nsd_diag1|P0": ("tolerance", 0, 5, False),
    "dice|P0|fuseg_r256": ("resolution", 0, 0, True),
    "nsd2|P0|fuseg_r256": ("resolution", 0, 2, True),
}
# reference condition for the paired change of each condition
REFC = {"dice|P1": "dice|P0", "dice|P2": "dice|P0", "nsd2|P1": "nsd2|P0", "nsd2|P2": "nsd2|P0",
        "nsd1|P0": "nsd2|P0", "nsd3|P0": "nsd2|P0", "nsd5|P0": "nsd2|P0", "nsd_diag1|P0": "nsd2|P0",
        "dice|P0|fuseg_r256": "dice|P0", "nsd2|P0|fuseg_r256": "nsd2|P0"}
DELTA_Q = ["wound pairs", "wound vs other", "modality_diff", "colonoscopy pairs"]

BOOT = {}           # (set, cond) -> {name: (point, boot array)}


def es(pt, bs):
    return {"est": float(pt), "ci": ci(bs)}


def analyse(st, cond):
    fam, vi, mi, r256 = CONDS[cond]
    M = set_means(st, vi, mi, r256)
    tk = TAKES[st]
    Mb = M[:, tk].transpose(1, 0, 2)
    R = corrmat(M)
    pt, bs = summ(R), summ(corrmat(Mb))
    out = {"pooled": {q: es(pt[q], bs[q]) for q in QK}, "exact_modality": exact_test(R),
           "n_methods": int(M.shape[1])}
    store = {f"pooled|{q}": (float(pt[q]), bs[q]) for q in QK}
    if len(SETS[st]) > 1:
        sl = slices_of(st)
        Rs, Rw, Wg = strat_parts(M, sl)
        Rsb, Rwb, Wgb = strat_parts(Mb, sl)
        ps, bss, pw, bw = summ(Rs), summ(Rsb), summ(Rw), summ(Rwb)
        out["stratified"] = {q: es(ps[q], bss[q]) for q in QK}
        out["size_weighted_within"] = {q: es(pw[q], bw[q]) for q in QK}
        out["within"] = {}
        win_pt, win_bs = {}, {}
        for g, Rg, Rgb in zip(SETS[st], Wg, Wgb):
            a, b = summ(Rg), summ(Rgb)
            out["within"][g] = {q: es(a[q], b[q]) for q in QK}
            win_pt[g], win_bs[g] = a, b
        comp = {}
        for dom, kin, kout in (("wound", "wound pairs", "wound vs other"),
                               ("colonoscopy", "colonoscopy pairs", "colonoscopy vs other")):
            wmax_pt = max(float(win_pt[g][kin]) for g in SETS[st])
            wmax_bs = np.max(np.stack([win_bs[g][kin] for g in SETS[st]]), 0)
            gap = es(pt[kin] - wmax_pt, bs[kin] - wmax_bs)
            other = out["pooled"][kout]
            s_minus_w = float(ps[kin] - pw[kin])
            s_minus_p = es(ps[kin] - pt[kin], bss[kin] - bs[kin])
            lo_w = min(float(win_pt[g][kin]) for g in SETS[st])
            c1 = gap["ci"][0] is not None and gap["ci"][0] > 0
            c2 = other["ci"][1] is not None and other["ci"][1] < 0
            c3 = abs(s_minus_w) <= STRAT_TOL
            comp[dom] = {"pooled": out["pooled"][kin], "within": {g: float(win_pt[g][kin]) for g in SETS[st]},
                         "pooled_above_both_within_point": bool(float(pt[kin]) > wmax_pt),
                         "pooled_vs_other_below0_point": bool(other["est"] < 0),
                         "pooled_minus_higher_within": gap, "pooled_vs_other": other,
                         "stratified": out["stratified"][kin],
                         "size_weighted_within": float(pw[kin]),
                         "stratified_minus_size_weighted_within": s_minus_w,
                         "stratified_inside_within_range": bool(lo_w <= float(ps[kin]) <= wmax_pt),
                         "stratified_minus_pooled": s_minus_p,
                         "C1_pooled_above_both_within_ci_above0": bool(c1),
                         "C2_pooled_vs_other_ci_below0": bool(c2),
                         "C3_stratified_within_tol_of_size_weighted": bool(c3),
                         "holds": bool(c1 and c2 and c3)}
            store[f"strat|{kin}"] = (float(ps[kin]), bss[kin])
        out["composition"] = comp
    BOOT[(st, cond)] = store
    return out


def delta(st, cond, ref, name):
    a, ab = BOOT[(st, cond)][name]
    b, bb = BOOT[(st, ref)][name]
    return {"est": a - b, "ci": ci(ab - bb)}


print("analysing", flush=True)
RESU = {st: {} for st in SETS}
for st in SETS:
    for cond in CONDS:
        RESU[st][cond] = analyse(st, cond)
    for cond, ref in REFC.items():
        d = {q: delta(st, cond, ref, f"pooled|{q}") for q in DELTA_Q}
        if len(SETS[st]) > 1:
            d["stratified wound pairs"] = delta(st, cond, ref, "strat|wound pairs")
            d["stratified colonoscopy pairs"] = delta(st, cond, ref, "strat|colonoscopy pairs")
        RESU[st][cond]["change_vs_" + ref] = d
    print(f"  {st} done {time.time() - T0:.0f}s", flush=True)

# ------------------------------------------------------------- checks on the rescore data
RV7 = jload(os.path.join(OUT, "robust_v7.json"))
rv = {"max_abs_diff_est": 0.0, "max_abs_diff_ci": 0.0, "n_compared": 0}


def cmp_rv(mine, theirs):
    rv["max_abs_diff_est"] = max(rv["max_abs_diff_est"], abs(mine["est"] - theirs["est"]))
    rv["max_abs_diff_ci"] = max(rv["max_abs_diff_ci"], max(abs(a - b) for a, b in zip(mine["ci"], theirs["ci"])))
    rv["n_compared"] += 1


RV_Q = {"wound pairs": "wound pairs", "colonoscopy pairs": "colonoscopy pairs", "modality_diff": "modality_diff",
        "same modality (16)": "same modality (16)", "across modality (39)": "across modality (39)",
        "fuseg~azh": "fuseg~azh", "control azh~azh_train_slice": "control azh~azh_train_slice"}
for g in ("M19", "K11"):
    for m in ("dice", "nsd2"):
        for v in VARIANTS:
            for q, qq in RV_Q.items():
                cmp_rv(RESU[g][f"{m}|{v}"]["pooled"][q], RV7["postprocessing"]["summaries"][f"{g}|{m}|{v}"][qq])
        for v in ("P1", "P2"):
            for q in ("wound pairs", "modality_diff"):
                cmp_rv(RESU[g][f"{m}|{v}"]["change_vs_" + f"{m}|P0"][q],
                       RV7["postprocessing"]["delta_vs_P0"][f"{g}|{m}|{v}-P0|{q}"])
        e = RV7["resolution"][f"{g}|{m}"]
        for q, qq in RV_Q.items():
            cmp_rv(RESU[g][f"{m}|P0"]["pooled"][q], e["with_fuseg"][qq])
            cmp_rv(RESU[g][f"{m}|P0|fuseg_r256"]["pooled"][q], e["with_fuseg_r256"][qq])
        cmp_rv(RESU[g][f"{m}|P0|fuseg_r256"]["change_vs_" + f"{m}|P0"]["wound pairs"],
               e["delta_r256_minus_native_wound_pairs"])
    for mi, m in enumerate(METRICS):
        cond = f"{m}|P0"
        for q, qq in RV_Q.items():
            cmp_rv(RESU[g][cond]["pooled"][q], RV7["tolerance"][f"{g}|{m}|P0"][qq])
        if m not in ("dice", "nsd2"):
            for q in ("wound pairs", "modality_diff"):
                cmp_rv(RESU[g][cond]["change_vs_nsd2|P0"][q], RV7["tolerance_delta_vs_nsd2"][f"{g}|{m}-nsd2|{q}"])
assert rv["max_abs_diff_est"] < 1e-12 and rv["max_abs_diff_ci"] < 1e-12, rv
rv["what"] = ("M19 and K11 computed here vs results/v7/robust_v7.json: post-processing summaries and changes, "
              "tolerance summaries and changes vs nsd2, resolution summaries and change")
CHECKS["M19_K11_vs_robust_v7"] = rv
print("  M19/K11 vs robust_v7:", rv["max_abs_diff_est"], rv["max_abs_diff_ci"], flush=True)

# U+M19 P0 vs stats_v7 point (0.821 and 1.101)
sv = {}
for st, sk in (("U+M19", "U+M19"), ("U+F", "U+F"), ("U", "U"), ("F", "F"), ("M19", "M19"), ("U+K11", "U+K")):
    for m, mm in (("dice", "dice"), ("nsd2", "nsd")):
        mine = RESU[st][f"{m}|P0"]
        ref = SV7["point"][f"{mm}|{sk}"]
        row_ = {}
        for q in ("wound pairs", "wound vs other", "modality_diff"):
            row_[q] = {"rescore": mine["pooled"][q]["est"], "stats_v7": ref[SV7_KEY[q]],
                       "abs_diff": abs(mine["pooled"][q]["est"] - ref[SV7_KEY[q]])}
        row_["exact_p"] = {"rescore": mine["exact_modality"]["p"],
                           "stats_v7": SV7["exact"][f"{mm}|{sk}"]["full_55"]["p"]}
        if st in POOLED:
            row_["stratified wound pairs"] = {
                "rescore": mine["stratified"]["wound pairs"]["est"],
                "stats_v7": SV7["stratified"][f"{mm}|{sk}"]["pearson_on_percentiles"]["wound pairs"]}
        sv[f"{m}|{st}" + (" (stats_v7 is U+K, 14 Kvasir methods)" if st == "U+K11" else "")] = row_
CHECKS["P0_vs_stats_v7_point"] = sv
um = RESU["U+M19"]["dice|P0"]
assert round(um["pooled"]["wound pairs"]["est"], 3) == 0.821, um["pooled"]["wound pairs"]
assert round(um["pooled"]["modality_diff"]["est"], 3) == 1.101, um["pooled"]["modality_diff"]
assert um["exact_modality"]["p"] == SV7["exact"]["dice|U+M19"]["full_55"]["p"] == 2 / 13860
assert RESU["M19"]["dice|P0"]["exact_modality"]["p"] == SV7["exact"]["dice|M19"]["full_55"]["p"]
for st in ("U", "F", "U+F"):          # local-only sets: tiny differences only (P0 vs perimage32 <= 1e-6)
    for m in ("dice", "nsd2"):
        for q in ("wound pairs", "wound vs other", "modality_diff"):
            assert sv[f"{m}|{st}"][q]["abs_diff"] < 1e-6, (st, m, q, sv[f"{m}|{st}"][q])
CHECKS["U+M19_dice_P0"] = {"wound pairs": um["pooled"]["wound pairs"]["est"],
                           "modality_diff": um["pooled"]["modality_diff"]["est"],
                           "required": {"wound pairs": 0.821, "modality_diff": 1.101},
                           "exact_p": um["exact_modality"]["p"],
                           "M19_exact_p": RESU["M19"]["dice|P0"]["exact_modality"]["p"]}

# independent scipy route for a few pooled values
ind = []
for st in POOLED:
    for cond in ("dice|P0", "nsd2|P2", "nsd5|P0", "dice|P0|fuseg_r256"):
        fam, vi, mi, r256 = CONDS[cond]
        M = set_means(st, vi, mi, r256)
        w = np.mean([spearmanr(M[DS.index(a)], M[DS.index(b)])[0] for a, b in itertools.combinations(WOUND, 2)])
        ind.append(abs(w - RESU[st][cond]["pooled"]["wound pairs"]["est"]))
        o = np.mean([spearmanr(M[DS.index(a)], M[DS.index(b)])[0] for a in WOUND for b in REFS
                     if b not in WOUND])
        ind.append(abs(o - RESU[st][cond]["pooled"]["wound vs other"]["est"]))
assert max(ind) < 1e-12
CHECKS["scipy_route_pooled_wound_pairs_and_wound_vs_other"] = {"n_compared": len(ind), "max_abs_diff": float(max(ind))}

# ------------------------------------------------------------- composition summary over conditions
COMPS = {}
for st in POOLED:
    for dom in ("wound", "colonoscopy"):
        rows = {c: RESU[st][c]["composition"][dom] for c in CONDS}
        g2 = SETS[st][1]

        def rng_(f):
            v = [f(r) for r in rows.values()]
            return [float(min(v)), float(max(v))]
        COMPS[f"{st}|{dom}"] = {
            "n_conditions": len(rows),
            "n_holds": int(sum(r["holds"] for r in rows.values())),
            "conditions_failing": {c: [k for k in ("C1_pooled_above_both_within_ci_above0",
                                                   "C2_pooled_vs_other_ci_below0",
                                                   "C3_stratified_within_tol_of_size_weighted") if not r[k]]
                                   for c, r in rows.items() if not r["holds"]},
            "n_C1": int(sum(r["C1_pooled_above_both_within_ci_above0"] for r in rows.values())),
            "n_pooled_above_both_within_point": int(sum(r["pooled_above_both_within_point"] for r in rows.values())),
            "n_pooled_vs_other_below0_point": int(sum(r["pooled_vs_other_below0_point"] for r in rows.values())),
            "conditions_pooled_not_above_both_point": [c for c, r in rows.items() if not r["pooled_above_both_within_point"]],
            "n_C2": int(sum(r["C2_pooled_vs_other_ci_below0"] for r in rows.values())),
            "n_C3": int(sum(r["C3_stratified_within_tol_of_size_weighted"] for r in rows.values())),
            "n_stratified_inside_within_range": int(sum(r["stratified_inside_within_range"] for r in rows.values())),
            "n_stratified_minus_pooled_ci_below0": int(sum(r["stratified_minus_pooled"]["ci"][1] < 0 for r in rows.values())),
            "range_pooled": rng_(lambda r: r["pooled"]["est"]),
            "range_within_U": rng_(lambda r: r["within"]["U"]),
            f"range_within_{g2}": rng_(lambda r: r["within"][g2]),
            "range_pooled_minus_higher_within": rng_(lambda r: r["pooled_minus_higher_within"]["est"]),
            "range_pooled_minus_higher_within_ci_lower": rng_(lambda r: r["pooled_minus_higher_within"]["ci"][0]),
            "range_pooled_vs_other": rng_(lambda r: r["pooled_vs_other"]["est"]),
            "range_pooled_vs_other_ci_upper": rng_(lambda r: r["pooled_vs_other"]["ci"][1]),
            "range_stratified": rng_(lambda r: r["stratified"]["est"]),
            "max_abs_stratified_minus_size_weighted": float(max(abs(r["stratified_minus_size_weighted_within"])
                                                                for r in rows.values())),
            "range_stratified_minus_pooled": rng_(lambda r: r["stratified_minus_pooled"]["est"]),
        }
MODRANGE = {}
for st in SETS:
    v = [RESU[st][c]["pooled"]["modality_diff"] for c in CONDS]
    p = [RESU[st][c]["exact_modality"]["p"] for c in CONDS]
    MODRANGE[st] = {"est_range": [min(x["est"] for x in v), max(x["est"] for x in v)],
                    "lowest_ci_lower": min(x["ci"][0] for x in v),
                    "n_ci_above0": int(sum(x["ci"][0] > 0 for x in v)),
                    "p_range": [min(p), max(p)],
                    "n_p_at_min": int(sum(abs(x - N_EQUIV / len(PERMS)) < 1e-12 for x in p)),
                    "n_p_below_0.05": int(sum(x < 0.05 for x in p)), "n_conditions": len(v)}
CHG = {}
for st in SETS:
    ch = []
    for cond, ref in REFC.items():
        for q in ("wound pairs", "stratified wound pairs"):
            d = RESU[st][cond]["change_vs_" + ref].get(q)
            if d is not None:
                ch.append((cond, q, d))
    CHG[st] = {"n_changes": len(ch),
               "n_ci_excludes_0": int(sum(not (d["ci"][0] <= 0 <= d["ci"][1]) for _, _, d in ch)),
               "excluding_0": [{"condition": c, "quantity": q, **d} for c, q, d in ch
                               if not (d["ci"][0] <= 0 <= d["ci"][1])],
               "largest_abs_change": max(({"condition": c, "quantity": q, **d} for c, q, d in ch),
                                         key=lambda x: abs(x["est"]))}

R = {"definitions": {
    "groups": {g: {"n": len(v), "members": v} for g, v in GROUPS.items()},
    "sets": SETS,
    "seed_members_M19": {c: MEMBERS[c] for c in GROUPS["M19"]},
    "K11_vs_paper_K14": f"K11 = paper K14 minus {K_MISSING} (not in the Kaggle rescore files)",
    "agreement": "Spearman (Pearson on average ranks) of per-method mean scores between two datasets over a method set",
    "summaries": {"wound pairs": "mean of the 6 wound-reference pairs",
                  "wound vs other": "mean of the 28 pairs of a wound reference with a non-wound reference (stats_v7 comp|wound|in x out)",
                  "colonoscopy pairs": "mean of the 10 colonoscopy pairs",
                  "colonoscopy vs other": "mean of the 30 pairs of a colonoscopy reference with a non-colonoscopy reference",
                  "modality_diff": "mean of the 16 same-modality pairs minus mean of the 39 across-modality pairs (11 references)"},
    "stratified": "methods ranked within each group, percentiles (rank - 0.5) / n_group, Pearson correlation of the pooled percentiles (stats_v7 section 7 pearson_on_percentiles)",
    "size_weighted_within": "sum over groups of n_group times the group's agreement matrix, divided by N",
    "method_bootstrap": f"stats_v5 design (stats_v7.v5_method_takes): per draw methods resampled with replacement within each group, groups in set order, default_rng({SEED_METHOD}) per set, B {B}; 95% percentile interval; same draws for every condition of a set (paired changes); within-group and stratified values from the group parts of the same draw",
    "exact_test": "all 13,860 distinct assignments of the 11 modality labels; p = share with difference >= observed (stats_v7.exact_test); smallest attainable p = 2/13,860",
    "variants": {"P0": "main pipeline (trained: Gaussian smoothing, threshold, small components; untrained: close, open, small components)",
                 "P1": "trained: threshold then 5 px close and open, small components; untrained: identical to P0 by definition",
                 "P2": "threshold (trained) or raw mask (untrained), small-component removal only"},
    "conditions": {k: {"family": v[0], "variant": VARIANTS[v[1]], "metric": METRICS[v[2]], "fuseg_r256": v[3]}
                   for k, v in CONDS.items()},
    "change_reference": REFC,
    "composition_criteria": {
        "C1": "pooled agreement in the domain minus the higher within-group value, paired 95% interval above 0",
        "C2": "pooled domain-vs-other agreement, 95% interval below 0",
        "C3": f"|stratified minus size-weighted within-group average| <= {STRAT_TOL} (point values)",
        "holds": "C1 and C2 and C3",
        "domains": "wound (main claim); colonoscopy (mirror view)"},
    "exclusions": "results/exclusions_v5.json by image index; fuseg_r256 uses the fuseg exclusions",
    "sources": {"local": "results/v7/rescore_local/<set>.json", "kaggle": "results/kaggle2/rescore/<family>/<set>.json"}},
    "checks": CHECKS, "results": RESU, "composition_summary": COMPS,
    "modality_diff_summary": MODRANGE, "change_summary": CHG}
R["runtime_s"] = round(time.time() - T0, 1)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


JPATH = os.path.join(OUT, "robust_pooled_v7.json")
with open(JPATH + ".tmp", "w") as f:
    json.dump(clean(R), f, indent=1)
os.replace(JPATH + ".tmp", JPATH)
print("wrote robust_pooled_v7.json", flush=True)

# ================================================================ markdown
# Everything below reads the json just written.
J = jload(JPATH)
RS_ = J["results"]
L = []


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def fe(e):
    return f"{f3(e['est'])} [{f3(e['ci'][0])}, {f3(e['ci'][1])}]"


def fp(r):
    s = np.format_float_positional(r["p"], precision=2, unique=False, fractional=False, trim="-")
    return s + (" (min)" if abs(r["p"] - r["min_p"]) < 1e-12 else "")


def yn(b):
    return "yes" if b else "no"


def row(*c):
    L.append("| " + " | ".join(str(x) for x in c) + " |")


def rg(a):
    return f"{a[0]:.3f} to {a[1]:.3f}"


def p2(x):
    return np.format_float_positional(x, precision=2, unique=False, fractional=False, trim="-")


def prange(m):
    a, b = m["p_range"]
    if m["n_p_at_min"] == m["n_conditions"]:
        return f"exact p {p2(a)}, the smallest attainable, in all {m['n_conditions']} conditions"
    return f"exact p {p2(a)} to {p2(b)}"


CL = {"dice|P0": "Dice P0", "dice|P1": "Dice P1", "dice|P2": "Dice P2", "nsd2|P0": "NSD 2 px P0",
      "nsd2|P1": "NSD 2 px P1", "nsd2|P2": "NSD 2 px P2", "nsd1|P0": "NSD 1 px", "nsd3|P0": "NSD 3 px",
      "nsd5|P0": "NSD 5 px", "nsd_diag1|P0": "NSD 1% diag", "dice|P0|fuseg_r256": "Dice (FUSeg 256 px)",
      "nsd2|P0|fuseg_r256": "NSD 2 px (FUSeg 256 px)"}
CS = J["composition_summary"]
MR = J["modality_diff_summary"]
ck = J["checks"]
NC = len(CONDS)

L.append("# ROBUST POOLED V7: does the composition effect survive analysis choices?")
L.append("")
L.append("Generated by `scripts/v7/robust_pooled_v7.py`. Every number is in `results/v7/robust_pooled_v7.json`.")
L.append("")
L.append("## Bottom line")
L.append("")
cw = {st: CS[f"{st}|wound"] for st in POOLED}
L.append("Short answer: the composition effect, judged by all three criteria (C1 to C3, defined below), holds in " +
         "; ".join(f"{cw[st]['n_holds']} of {NC} conditions for {st}" for st in POOLED) +
         ". " + ("Stratification brings the pooled value back to the size-weighted within-group average (C3) in every condition and every pooled set. "
                 if all(cw[st]["n_C3"] == NC for st in POOLED) else
                 "Stratification brings the pooled value back to the size-weighted within-group average (C3) in " +
                 "; ".join(f"{cw[st]['n_C3']} of {NC} conditions for {st}" for st in POOLED) + ". ") +
         "Details per set:")
L.append("")
for st in POOLED:
    c = cw[st]
    g2 = SETS[st][1]
    fails = c["conditions_failing"]
    hold_txt = (f"holds in all {NC} conditions" if c["n_holds"] == NC else
                f"holds in {c['n_holds']} of {NC} conditions" if c["n_holds"] > 0 else
                f"does not hold in any of the {NC} conditions")
    npa = c["n_pooled_above_both_within_point"]
    notabove = c["conditions_pooled_not_above_both_point"]
    L.append(f"- **{st}**: the composition effect in the wound domain {hold_txt}. "
             f"Pooled wound-pair agreement {rg(c['range_pooled'])}, against {rg(c['range_within_U'])} within U and "
             f"{rg(c[f'range_within_{g2}'])} within {g2}. "
             f"The pooled value is above both within-group values in {npa} of {NC} conditions" +
             (f" (not in: {', '.join(CL[x] for x in notabove)})" if 0 < len(notabove) < NC else "") + ". "
             f"Pooled minus the higher within-group value: {rg(c['range_pooled_minus_higher_within'])} "
             f"(interval above 0 in {c['n_C1']} of {NC}). "
             f"Pooled wound-vs-other agreement {rg(c['range_pooled_vs_other'])} (below 0 in {c['n_pooled_vs_other_below0_point']} of {NC}; "
             f"interval below 0 in {c['n_C2']} of {NC}). "
             f"Stratified wound-pair agreement {rg(c['range_stratified'])}, at most {c['max_abs_stratified_minus_size_weighted']:.1e} "
             f"from the size-weighted within-group average (criterion met in {c['n_C3']} of {NC}); "
             f"stratifying changes the pooled value by {rg(c['range_stratified_minus_pooled'])}.")
    if fails:
        crit = {"C1_pooled_above_both_within_ci_above0": "C1", "C2_pooled_vs_other_ci_below0": "C2",
                "C3_stratified_within_tol_of_size_weighted": "C3"}
        L.append("  Conditions where it does not hold (criteria missed): " +
                 "; ".join(f"{CL[k]} ({', '.join(crit[x] for x in v)})" for k, v in fails.items()) + ".")
cc = CS["U+K11|colonoscopy"]
L.append(f"- **U+K11, colonoscopy domain (mirror view)**: the same criteria with colonoscopy in place of wound hold in "
         f"{cc['n_holds']} of {NC} conditions. Pooled colonoscopy-pair agreement {rg(cc['range_pooled'])} against "
         f"{rg(cc['range_within_U'])} within U and {rg(cc['range_within_K11'])} within K11; pooled minus the higher "
         f"within-group value {rg(cc['range_pooled_minus_higher_within'])} (interval above 0 in {cc['n_C1']} of {NC}); "
         f"pooled colonoscopy-vs-other agreement {rg(cc['range_pooled_vs_other'])} (interval below 0 in {cc['n_C2']} of {NC}).")
L.append(f"- Modality difference, pooled sets, over the {NC} conditions: " +
         "; ".join(f"{st} {rg(MR[st]['est_range'])} ({prange(MR[st])}; "
                   f"interval above 0 in {MR[st]['n_ci_above0']} of {NC})" for st in POOLED) + ".")
L.append("- Modality difference, single groups for comparison: " + "; ".join(
    f"{st} {rg(MR[st]['est_range'])} ({prange(MR[st])}; interval above 0 in {MR[st]['n_ci_above0']} of {NC})"
    for st in SINGLE) + ".")
chg = J["change_summary"]
L.append("- Paired changes from the reference condition (wound-pair agreement, pooled and stratified) with an interval that excludes 0: " +
         "; ".join(f"{st} {chg[st]['n_ci_excludes_0']} of {chg[st]['n_changes']}" for st in SETS) + ".")
L.append("")

L.append("## What was analysed")
L.append("")
L.append("- Sources: local rescore files (`results/v7/rescore_local`, 32 local methods, unrounded) and the Kaggle rescore files (`results/kaggle2/rescore`, rounded to 5 decimals). Exclusions from `results/exclusions_v5.json`; FUSeg 256 uses the FUSeg exclusions.")
L.append(f"- Groups: U {len(GROUPS['U'])} untrained, F {len(GROUPS['F'])} trained from scratch on FUSeg (split by `ALL_METHOD_FAMILY` as `perimage_v6`), "
         f"M19 {len(GROUPS['M19'])} (18 pretrained architectures averaged per image over the seeds that exist, plus nnU-Net 2D), "
         f"K11 {len(GROUPS['K11'])} (the paper's K without {', '.join(K_MISSING)}, which were not rescored).")
L.append("- Agreement: Spearman correlation of per-method mean scores between two datasets. Wound pairs: mean of the 6 pairs of medetec, co2wounds, fuseg, azh. "
         "Wound vs other: mean of the 28 pairs of a wound dataset with one of the 7 other references. Modality difference: mean of the 16 same-modality pairs minus mean of the 39 across-modality pairs.")
L.append("- Stratified agreement: methods are ranked within their own group first, ranks become within-group percentiles, and the Pearson correlation of the pooled percentiles is taken (as in `STATS_V7.md` section 7).")
L.append(f"- Intervals: method bootstrap, methods resampled within each group, {B} draws, seed {SEED_METHOD}, stats_v5 design. One set of draws per method set, used for every condition, so the changes between conditions have paired intervals. Within-group and stratified values use the group parts of the same draw.")
L.append("- Exact p: relabeling test of the modality difference over all 13,860 distinct assignments of the 11 modality labels; (min) marks the smallest attainable value, 2/13,860.")
L.append("- Conditions (12): post-processing P0, P1, P2 for Dice and NSD at 2 px; NSD at 1, 3, 5 px and at 1% of the image diagonal (P0); FUSeg replaced by FUSeg at 256 px (P0, Dice and NSD 2 px). "
         "P0 is the main pipeline. P1 = threshold, then 5 px closing and opening and small-component removal, for trained methods; for untrained methods P1 is P0, because their main pipeline already is this one. P2 = small-component removal only.")
L.append(f"- Composition criteria per pooled set and condition. C1: pooled wound-pair agreement minus the higher within-group value, paired interval above 0. "
         f"C2: pooled wound-vs-other agreement, interval below 0. C3: stratified wound-pair agreement within {STRAT_TOL} of the size-weighted within-group average. "
         "The effect holds when C1, C2 and C3 all hold. The mirror view applies the same criteria to the colonoscopy domain.")
L.append("")

L.append("## Checks")
L.append("")
e = ck["engine_vs_stats_v7_run1"]
L.append(f"- Engine with run-1 scores vs `stats_v7.json` (sets {', '.join(e['sets'])}; Dice and NSD; {e['n_compared']} summaries): "
         f"point max abs difference {e['max_abs_diff_point']:.1e}, B 2000 method-only intervals {e['max_abs_diff_ci_B2000']:.1e}, "
         f"stratified values {e['max_abs_diff_stratified']:.1e}, exact p and difference {e['max_abs_diff_exact_p']:.1e}.")
c = ck["local_P0_vs_perimage32"]
L.append(f"- Local rescore P0 vs `results/perimage32`, every image and method of the 12 clinical sets ({c['n_cells_compared_per_metric']:,} cells per metric): "
         f"max abs difference Dice {c['max_abs_diff_dice_other_cells']:.1e}, NSD 2 px {c['max_abs_diff_nsd2_other_cells']:.1e} "
         f"in every cell except {len(c['cells_above_tolerance'])} (tolerance {c['tolerance']:.0e}).")
for dv in c["cells_above_tolerance"]:
    L.append(f"  - Exception: {dv['method']} on {dv['set']} image {dv['image_id']} (index {dv['image_index']}): Dice {dv['dice_rescore']:.6f} vs "
             f"{dv['dice_perimage32']:.6f}, NSD 2 px {dv['nsd2_rescore']:.6f} vs {dv['nsd2_perimage32']:.6f}. "
             "The rescore file records the same failure of its own 1e-6 check. It does not affect any result here: ")
    for mt in c["materiality"]:
        if mt["set"] == dv["set"]:
            L.append(f"    {mt['metric']}: the method's dataset mean moves by {mt['method_mean_shift']:.1e}, the nearest other local method mean is "
                     f"{mt['nearest_other_local_method_mean_gap']:.4f} away, and the order of the 32 local method means is "
                     f"{'unchanged' if mt['rank_order_unchanged'] else 'CHANGED'}. Every quantity here depends on the means only through their order.")
L.append(f"- Untrained methods: P1 equals P0 exactly (max abs difference {ck['untrained_P1_equals_P0']['max_abs_P1_minus_P0_untrained']:.1e}, "
         f"{ck['untrained_P1_equals_P0']['untrained_methods']} methods, {ck['untrained_P1_equals_P0']['sets']} sets, all metrics).")
c = ck["M19_K11_vs_robust_v7"]
L.append(f"- M19 and K11 computed here vs `robust_v7.json` ({c['n_compared']} summaries and changes): max abs difference estimate {c['max_abs_diff_est']:.1e}, interval {c['max_abs_diff_ci']:.1e}.")
c = ck["U+M19_dice_P0"]
L.append(f"- U+M19 Dice P0: wound-pair agreement {c['wound pairs']:.4f} (required 0.821), modality difference {c['modality_diff']:.4f} (required 1.101), "
         f"exact p {c['exact_p']:.5f}; M19 exact p {c['M19_exact_p']:.5f}. Both p values equal `stats_v7.json`.")
L.append("- P0 values from the rescore files vs `stats_v7.json` (run-1 scores):")
for k, v in ck["P0_vs_stats_v7_point"].items():
    L.append(f"  - {k}: " + "; ".join(
        f"{q} {v[q]['rescore']:.4f} vs {v[q]['stats_v7']:.4f}" for q in ("wound pairs", "wound vs other", "modality_diff")) +
        (f"; stratified wound pairs {v['stratified wound pairs']['rescore']:.4f} vs {v['stratified wound pairs']['stats_v7']:.4f}"
         if "stratified wound pairs" in v else "") +
        f"; exact p {np.format_float_positional(v['exact_p']['rescore'], precision=2, unique=False, fractional=False, trim='-')} vs "
        f"{np.format_float_positional(v['exact_p']['stats_v7'], precision=2, unique=False, fractional=False, trim='-')}")
c = ck["scipy_route_pooled_wound_pairs_and_wound_vs_other"]
L.append(f"- Independent route (scipy spearmanr) for pooled wound pairs and wound vs other, {c['n_compared']} values: max abs difference {c['max_abs_diff']:.1e}.")
L.append("- Kept images: " + ", ".join(f"{s} {n}" for s, n in ck["n_kept"].items()) + ".")
L.append("")

L.append("## 1. Composition effect under every condition (wound domain)")
L.append("")
L.append("Estimate [95% method-bootstrap interval]. Within U and within the other group are the within-group wound-pair agreements of the same condition. Gap = pooled minus the higher within-group value (paired interval).")
L.append("")
row("set", "condition", "pooled wound pairs", "within U", "within other", "gap", "pooled wound vs other",
    "stratified wound pairs", "size-weighted within", "C1", "C2", "C3", "holds")
row(*["---"] * 13)
for st in POOLED:
    g2 = SETS[st][1]
    for cnd in CONDS:
        x = RS_[st][cnd]["composition"]["wound"]
        row(st, CL[cnd], fe(x["pooled"]), f3(x["within"]["U"]), f"{g2} {f3(x['within'][g2])}",
            fe(x["pooled_minus_higher_within"]), fe(x["pooled_vs_other"]), fe(x["stratified"]),
            f3(x["size_weighted_within"]), yn(x["C1_pooled_above_both_within_ci_above0"]),
            yn(x["C2_pooled_vs_other_ci_below0"]), yn(x["C3_stratified_within_tol_of_size_weighted"]), yn(x["holds"]))
L.append("")
L.append("Minimum and maximum over the 12 conditions:")
L.append("")
row("set", "conditions where it holds", "pooled above both within (point)", "pooled wound pairs", "within U", "within other", "gap", "gap interval lower bound",
    "pooled wound vs other", "its interval upper bound", "stratified wound pairs", "max abs stratified minus size-weighted", "stratified minus pooled")
row(*["---"] * 13)
for st in POOLED:
    c = cw[st]
    g2 = SETS[st][1]
    row(st, f"{c['n_holds']} of {c['n_conditions']}", f"{c['n_pooled_above_both_within_point']} of {c['n_conditions']}",
        rg(c["range_pooled"]), rg(c["range_within_U"]),
        f"{g2} {rg(c[f'range_within_{g2}'])}", rg(c["range_pooled_minus_higher_within"]),
        rg(c["range_pooled_minus_higher_within_ci_lower"]), rg(c["range_pooled_vs_other"]),
        rg(c["range_pooled_vs_other_ci_upper"]), rg(c["range_stratified"]),
        f"{c['max_abs_stratified_minus_size_weighted']:.1e}", rg(c["range_stratified_minus_pooled"]))
L.append("")

L.append("## 2. Mirror view: colonoscopy domain")
L.append("")
L.append("Same criteria with the 10 colonoscopy pairs and the 30 colonoscopy-vs-other pairs. Relevant mainly for U+K11 (Kvasir-trained methods).")
L.append("")
row("set", "condition", "pooled colonoscopy pairs", "within U", "within other", "gap", "pooled colonoscopy vs other",
    "stratified", "C1", "C2", "C3", "holds")
row(*["---"] * 12)
for st in POOLED:
    g2 = SETS[st][1]
    for cnd in CONDS:
        x = RS_[st][cnd]["composition"]["colonoscopy"]
        row(st, CL[cnd], fe(x["pooled"]), f3(x["within"]["U"]), f"{g2} {f3(x['within'][g2])}",
            fe(x["pooled_minus_higher_within"]), fe(x["pooled_vs_other"]), fe(x["stratified"]),
            yn(x["C1_pooled_above_both_within_ci_above0"]), yn(x["C2_pooled_vs_other_ci_below0"]),
            yn(x["C3_stratified_within_tol_of_size_weighted"]), yn(x["holds"]))
L.append("")
row("set", "conditions where it holds", "pooled colonoscopy pairs", "gap", "pooled colonoscopy vs other", "stratified")
row(*["---"] * 6)
for st in POOLED:
    c = CS[f"{st}|colonoscopy"]
    row(st, f"{c['n_holds']} of {c['n_conditions']}", rg(c["range_pooled"]), rg(c["range_pooled_minus_higher_within"]),
        rg(c["range_pooled_vs_other"]), rg(c["range_stratified"]))
L.append("")


def cond_table(conds, title_change):
    row("set", "condition", "wound pairs", "wound vs other", "modality diff", "exact p", "stratified wound pairs",
        f"wound pairs minus {title_change}", f"stratified minus {title_change}")
    row(*["---"] * 9)
    for st in POOLED + SINGLE:
        for cnd in conds:
            x = RS_[st][cnd]
            ref = REFC.get(cnd)
            ch = x.get("change_vs_" + ref) if ref else None
            strat = fe(x["stratified"]["wound pairs"]) if "stratified" in x else ""
            row(st, CL[cnd], fe(x["pooled"]["wound pairs"]), fe(x["pooled"]["wound vs other"]),
                fe(x["pooled"]["modality_diff"]), fp(x["exact_modality"]), strat,
                fe(ch["wound pairs"]) if ch else "",
                fe(ch["stratified wound pairs"]) if ch and "stratified wound pairs" in ch else "")


L.append("## 3. Post-processing (P0, P1, P2)")
L.append("")
L.append("Every method in the set gets the same variant. Changes are paired (same method draws) and relative to P0 of the same metric. M19 and K11 rows equal `robust_v7.json` (checked).")
L.append("")
cond_table(["dice|P0", "dice|P1", "dice|P2", "nsd2|P0", "nsd2|P1", "nsd2|P2"], "P0")
L.append("")
L.append("## 4. NSD tolerance (P0)")
L.append("")
L.append("Changes relative to NSD at 2 px, the paper's NSD.")
L.append("")
cond_table(["nsd1|P0", "nsd2|P0", "nsd3|P0", "nsd5|P0", "nsd_diag1|P0"], "NSD 2 px")
L.append("")
L.append("## 5. Resolution: FUSeg at 256 px (P0)")
L.append("")
L.append("FUSeg replaced by the same images and masks resampled so the long side is 256 px. Changes relative to the native FUSeg of the same metric.")
L.append("")
cond_table(["dice|P0", "dice|P0|fuseg_r256", "nsd2|P0", "nsd2|P0|fuseg_r256"], "native")
L.append("")
L.append("## 6. Modality difference over all 12 conditions")
L.append("")
row("set", "estimate range", "lowest interval lower bound", "interval above 0", "exact p range", "p at the minimum", "p below 0.05")
row(*["---"] * 7)
for st in POOLED + SINGLE:
    m = MR[st]
    row(st, rg(m["est_range"]), f3(m["lowest_ci_lower"]), f"{m['n_ci_above0']} of {m['n_conditions']}",
        f"{p2(m['p_range'][0])} to {p2(m['p_range'][1])}",
        f"{m['n_p_at_min']} of {m['n_conditions']}", f"{m['n_p_below_0.05']} of {m['n_conditions']}")
L.append("")
L.append("Largest change in wound-pair agreement (pooled or stratified) from the reference condition, per set:")
L.append("")
row("set", "condition", "quantity", "change", "changes with interval excluding 0")
row(*["---"] * 5)
for st in SETS:
    lc = chg[st]["largest_abs_change"]
    row(st, CL[lc["condition"]], lc["quantity"], fe(lc), f"{chg[st]['n_ci_excludes_0']} of {chg[st]['n_changes']}")
L.append("")
ex = [(st, x) for st in SETS for x in chg[st]["excluding_0"]]
if ex:
    L.append("Changes whose interval excludes 0: " + "; ".join(
        f"{st} {CL[x['condition']]} {x['quantity']} {fe(x)}" for st, x in ex) + ".")
    L.append("")

L.append("## Limitations")
L.append("")
L.append(f"- The Kvasir set is K11, not the paper's K14: the three Kvasir pixel classifiers ({', '.join(K_MISSING)}) were not rescored on Kaggle. U+K11 therefore has 29 methods, not 32, and its values differ from U+K in `STATS_V7.md`.")
L.append("- Thresholds are not varied here. The untrained methods produce binary masks with no probability map, so a threshold sweep cannot be applied to every method in a pooled set. The threshold analysis for modern methods alone is in `ROBUST_V7.md`.")
L.append("- P1 changes only the trained methods: for untrained methods P1 is P0 by definition (their main pipeline already uses closing and opening). So in U+M19, U+F and U+K11 the P1 condition moves only one group.")
L.append("- Conditions are varied one at a time: P1 and P2 are crossed with Dice and NSD at 2 px only, the NSD tolerances use P0 only, and FUSeg at 256 px uses P0 only. Only FUSeg was resampled.")
L.append("- Intervals resample methods only. They do not include image sampling, so they understate the total uncertainty.")
L.append("- The exact relabeling p is computed for each condition separately; no correction for the 12 conditions is applied. The smallest attainable value is 2/13,860 = 0.00014, so sets at that floor cannot be ordered by p.")
L.append(f"- The composition criteria are summaries chosen for this report: C3 uses a fixed tolerance of {STRAT_TOL}, and C1 and C2 depend on 95% method-bootstrap intervals.")
L.append("- The Kaggle rescore values (M19, K11) are rounded to 5 decimals and nnU-Net in them is a re-prediction of the run-1 checkpoint, so pooled values with M19 or K11 can differ slightly from `STATS_V7.md` (see Checks); the local values are unrounded.")
L.append("- M19 is near the Dice floor on the 7 non-wound references (`ROBUST_V7.md`), so its own rankings there carry little information; in pooled sets this floor is part of what separates the groups.")
L.append("")

txt = "\n".join(L)
assert chr(0x2014) not in txt and chr(0x2013) not in txt, "dash rule"
with open(os.path.join(OUT, "ROBUST_POOLED_V7.md"), "w") as f:
    f.write(txt)
print("wrote ROBUST_POOLED_V7.md", f"{time.time() - T0:.0f}s", flush=True)

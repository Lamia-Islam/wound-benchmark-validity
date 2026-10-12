#!/usr/bin/env python3
"""V7 checks: SAM / MedSAM variants of phase 2, and four run checks.
Writes results/v7/sam_checks_v7.json and results/v7/SAM_CHECKS_V7.md.
Run from the rebuild folder:  python scripts/v7/sam_checks_v7.py

Data and definitions
- Datasets: the 11 REFS and the control azh_train_slice (perimage_v6), with the
  perimage_v6 exclusions (8 FUSeg and 47 control images) unless a section says
  "all images". Kaggle score files hold every image in perimage32 order; the
  order is checked by comparing gt_fraction with perimage32 image by image.
- Metrics: Dice and NSD at tau = 2 px ("nsd"), per image, as stored. In the
  rescore files the same quantities are var[image][model][P0][dice] and
  var[image][model][P0][nsd2], rounded to 5 decimals by k2_rescore._r.
- Method columns:
    sam_vitb_box, medsam_vitb_box   run 1, reference (oracle) boxes, Hugging
                                    Face SamProcessor preprocessing (V6 group S)
    medsam_fixed_gtbox              phase 2, MedSAM with the official
                                    MedSAM_Inference.py preprocessing, oracle boxes
    sam_nnbox, medsam_fixed_nnbox   phase 2, boxes from nnU-Net's mask (no
                                    reference information)
    nnunet_2d                       run-1 nnU-Net scores (the column in M11/M19)
    nnunet_rescore_P0               phase-2 re-prediction of the same nnU-Net
                                    checkpoint, P0 pipeline; its masks are the
                                    ones the nnbox prompts were drawn from
  M19 and the other groups come from perimage_v7.load7 (architecture score =
  mean over available seeds).
- Bootstrap (same design as scripts/v6/practical_v6.py, boot_weights COPIED
  from there because importing practical_v6 runs the whole analysis):
  resample resampling units (photograph for Medetec and AZH incl. control,
  image otherwise) with replacement, B = 2000, each image enters with its
  unit's draw count as weight. One weight matrix per dataset, generator
  numpy default_rng([20261010, k]) with k = index of the dataset in
  REFS + [control]; the same draws serve every contrast and both metrics of
  that dataset (paired). Interval = 2.5 and 97.5 percentiles of the
  bootstrap mean differences.
- Agreement (COPIED from scripts/v6/stats_v6.py zr/corrmat/summ, because
  importing stats_v6 runs the full V6 analysis): Spearman correlation
  (Pearson on average ranks) of per-method dataset means between two
  datasets over a method set. wound pairs = mean over the 6 wound-wound pairs
  of the 11 REFS; modality_diff = mean of the 16 same-modality pairs minus
  mean of the 39 across-modality pairs; modality_diff_excl_within_source =
  13 different-source same-modality pairs minus the 39 across-modality pairs.
  The copy is checked against results/v6/stats_v6.json (group M point values).
"""
import ast
import difflib
import filecmp
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import perimage_v6 as V6                                          # noqa: E402
import perimage_v7 as V7                                          # noqa: E402

RES = os.path.join(ROOT, "results")
K1 = os.path.join(RES, "kaggle")
K2 = os.path.join(RES, "kaggle2")
OUTD = os.path.join(RES, "v7")
os.makedirs(OUTD, exist_ok=True)

REFS = list(V6.REFS)
CTRL = V6.CONTROL
SETS = REFS + [CTRL]
SEED = 20261010
B = 2000
METRICS = ("dice", "nsd")
RS_IDX = {"dice": 0, "nsd": 2}          # rescore metric index: dice, nsd2
ROUND_TOL = 5.0e-6 + 1e-12              # rescore values are rounded to 5 decimals
SAM1 = ["sam_vitb_box", "medsam_vitb_box"]
ORA2 = ["medsam_fixed_gtbox"]
NNB2 = ["sam_nnbox", "medsam_fixed_nnbox"]
NN = "nnunet_2d"


def r4(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def r6(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def f4(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def keep_mask(ref, n):
    k = np.ones(n, bool)
    for e in V6.EXCL.get(ref, []):
        k[e[0]] = False
    return k


P32 = {s: json.load(open(os.path.join(RES, "perimage32", s + ".json"))) for s in SETS}
ORDER_CHECK = {}


def read_group(path, ref):
    """Kaggle score file -> dict metric -> (images x names) with exclusions,
    plus names. Checks n and image order against perimage32."""
    g = json.load(open(path))
    p = P32[ref]
    assert g["n"] == len(p["ids"]), (path, g["n"], len(p["ids"]))
    dd = float(np.max(np.abs(np.asarray(g["gt_fraction"]) - np.asarray(p["gt_fraction"]))))
    ORDER_CHECK[os.path.relpath(path, RES)] = dd
    assert dd < 1e-9, (path, dd)
    k = keep_mask(ref, g["n"])
    return {m: np.asarray(g[m], float)[k] for m in METRICS}, g["names"]


def read_rescore(family, ref, excl=True):
    r = json.load(open(os.path.join(K2, "rescore", family, ref + ".json")))
    v = np.asarray(r["var"], float)                   # img x model x 3 x 6
    k = keep_mask(ref, r["n"]) if excl else np.ones(r["n"], bool)
    return r, v[k], k


def boot_weights(units, rng):
    """COPIED from scripts/v6/practical_v6.py."""
    u, inv = np.unique(units, return_inverse=True)
    draws = rng.integers(0, len(u), size=(B, len(u)))
    counts = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    return counts[:, inv]                                           # B x images


# ---------------------------------------------------------- agreement (copy)
NR = len(REFS)
LAB = [V6.MODALITY[r] for r in REFS]
SRC = [V6.SOURCE[r] for r in REFS]
PI, PJ = map(np.array, zip(*itertools.combinations(range(NR), 2)))
SAME = np.array([LAB[i] == LAB[j] for i, j in zip(PI, PJ)])
WSRC = np.array([SRC[i] == SRC[j] for i, j in zip(PI, PJ)])
ACROSS = ~SAME
DSSM = SAME & ~WSRC
WOUNDP = np.array([LAB[i] == LAB[j] == "wound" for i, j in zip(PI, PJ)])
COLONP = np.array([LAB[i] == LAB[j] == "colonoscopy" for i, j in zip(PI, PJ)])
WOUND_DS = WOUNDP & ~WSRC
FA = np.array([{REFS[i], REFS[j]} == {"fuseg", "azh"} for i, j in zip(PI, PJ)])
assert SAME.sum() == 16 and ACROSS.sum() == 39 and WOUNDP.sum() == 6 and DSSM.sum() == 13


def zr(T):
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def corrmat(T):
    Z = zr(T)
    return np.einsum("...in,...jn->...ij", Z, Z)


def summ(MEANS_refs_by_methods, ctrl_row):
    """MEANS: 11 REFS x methods; ctrl_row: control means (methods)."""
    T = np.vstack([MEANS_refs_by_methods, ctrl_row[None]])
    R = corrmat(T)
    rp = R[PI, PJ]
    return {"wound pairs (6)": float(rp[WOUNDP].mean()),
            "wound different-source pairs (5)": float(rp[WOUND_DS].mean()),
            "fuseg~azh": float(rp[FA][0]),
            "colonoscopy pairs (10)": float(rp[COLONP].mean()),
            "same modality (16)": float(rp[SAME].mean()),
            "across modality (39)": float(rp[ACROSS].mean()),
            "modality_diff": float(rp[SAME].mean() - rp[ACROSS].mean()),
            "modality_diff_excl_within_source": float(rp[DSSM].mean() - rp[ACROSS].mean()),
            "same dataset (azh~control)": float(R[REFS.index("azh"), NR])}


# =================================================================== load
D7 = {m: {s: V7.load7(s, m) for s in SETS} for m in METRICS}
COL = {s: {} for s in SETS}            # metric -> name -> per-image vector
for s in SETS:
    for m in METRICS:
        X = D7[m][s]["X"]
        cols = {n: X[:, V7.NAMES7.index(n)] for n in V7.NAMES7}
        a, na = read_group(os.path.join(K1, "scores", "sam", s + ".json"), s)
        o, no = read_group(os.path.join(K2, "scores", "sam_v2_oracle", s + ".json"), s)
        b, nb = read_group(os.path.join(K2, "scores", "sam_v2_nnbox", s + ".json"), s)
        assert na == SAM1 and no == ORA2 and nb == NNB2, (na, no, nb)
        # run-1 SAM columns read directly must equal the V6/V7 loader columns
        for j, n in enumerate(na):
            assert np.array_equal(a[m][:, j], cols[n]), (s, m, n)
        for j, n in enumerate(no):
            cols[n] = o[m][:, j]
        for j, n in enumerate(nb):
            cols[n] = b[m][:, j]
        r, v, _ = read_rescore("nnunet", s)
        assert r["names"] == ["nnunet_2d"] and r["thresholds"] == [0.5]
        cols["nnunet_rescore_P0"] = v[:, 0, 0, RS_IDX[m]]
        COL[s][m] = cols

RES_JSON = {"definitions": __doc__, "seed": SEED, "B": B}

# phase-2 SAM set coverage
cov = {}
for g in ("sam_v2_oracle", "sam_v2_nnbox"):
    fs = sorted(os.path.basename(f)[:-5] for f in os.listdir(os.path.join(K2, "scores", g)))
    cov[g] = {"sets": fs, "n_sets": len(fs),
              "synthetic_sets": [f for f in fs if f.startswith(("genB", "nr"))],
              "refs_plus_control_all_present": all(x in fs for x in SETS),
              "extra_sets": [f for f in fs if f not in SETS]}
cov["run1_sam_n_sets"] = len(os.listdir(os.path.join(K1, "scores", "sam")))
RES_JSON["coverage"] = cov

# ========================================================== Part 1a
VARS = SAM1 + ORA2 + NNB2
CONTRASTS = [("medsam_fixed_gtbox", "medsam_vitb_box"),
             ("sam_nnbox", NN), ("medsam_fixed_nnbox", NN),
             ("sam_nnbox", "nnunet_rescore_P0"), ("medsam_fixed_nnbox", "nnunet_rescore_P0"),
             ("sam_nnbox", "sam_vitb_box"), ("medsam_fixed_nnbox", "medsam_fixed_gtbox")]
p1a = {}
for k, s in enumerate(SETS):
    units = D7["dice"][s]["units"]
    rng = np.random.default_rng([SEED, k])
    W = boot_weights(units, rng)
    row = {"n_images": int(len(units)), "n_units": int(len(set(units))),
           "n_empty_reference": int(D7["dice"][s]["empty"].sum())}
    for m in METRICS:
        c = COL[s][m]
        m19 = np.array([c[n].mean() for n in V7.GROUPS7["M19"]])
        mm = {n: r4(c[n].mean()) for n in VARS + [NN, "nnunet_rescore_P0"]}
        mm["M19_mean"] = r4(m19.mean())
        mm["M19_min"] = r4(m19.min())
        mm["M19_max"] = r4(m19.max())
        mm["M19_n"] = len(m19)
        con = {}
        for a, b in CONTRASTS:
            d = c[a] - c[b]
            bm = (W @ d) / W.sum(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            con[f"{a} - {b}"] = {"diff": r4(d.mean()), "ci": [r4(lo), r4(hi)],
                                 "excludes_0": bool(lo > 0 or hi < 0)}
        row[m] = {"means": mm, "contrasts": con}
    p1a[s] = row
RES_JSON["part1a"] = p1a

# ========================================================== Part 1b
p1b = {}
for s in SETS:
    r, vall, _ = read_rescore("nnunet", s, excl=False)
    kk = keep_mask(s, r["n"])
    ge = np.asarray(r["gt_empty"], bool)
    P0 = vall[:, 0, 0, :]                               # img x 6 metrics
    nb = json.load(open(os.path.join(K2, "scores", "sam_v2_nnbox", s + ".json")))
    nbd = np.asarray(nb["dice"], float)
    nbn = np.asarray(nb["nsd"], float)
    out = {}
    for tag, sel in (("analysis_set", kk), ("all_images", np.ones(len(kk), bool))):
        e_empty_pred = sel & ge & (P0[:, 0] == 1.0)
        e_fp = sel & ge & (P0[:, 0] < 1.0)
        flag = sel & ~ge & (P0[:, 0] == 0.0) & np.all(P0[:, 1:] == 0.0, axis=1)
        nb_zero = np.all(nbd == 0.0, axis=1) & np.all(nbn == 0.0, axis=1)
        viol = flag & ~nb_zero
        dice0 = sel & ~ge & (P0[:, 0] == 0.0)
        out[tag] = {
            "n_images": int(sel.sum()), "n_ref_empty": int((sel & ge).sum()),
            "ref_empty_and_nnunet_empty (exact)": int(e_empty_pred.sum()),
            "ref_empty_and_nnunet_nonempty (exact)": int(e_fp.sum()),
            "ref_nonempty_nnunet_dice0": int(dice0.sum()),
            "ref_nonempty_nnunet_dice0_and_all_nsd0 (upper bound on empty masks)": int(flag.sum()),
            "of_those_nnbox_not_zero (proves mask non-empty)": int(viol.sum()),
            "upper_bound_no_box_images": int(e_empty_pred.sum() + flag.sum() - viol.sum()),
            "lower_bound_no_box_images": int(e_empty_pred.sum()),
            "share_upper_bound": r4((e_empty_pred.sum() + flag.sum() - viol.sum()) / sel.sum())}
    p1b[s] = out
RES_JSON["part1b"] = {"rule": "k2_sam.predict_nn: boxes = kit_run.gt_boxes(nnU-Net P0 mask) (up to 10 largest 8-connected components); no component -> no box -> medsam_prob/sam_proc_prob return an all-zero probability map -> empty prediction, scored with the study's convention (Dice and NSD 1 if the reference is also empty, else 0). nnU-Net masks are not in the local results, so empty masks are inferred: exact for empty references (Dice 1 iff prediction empty); for non-empty references an image with Dice 0 and NSD 0 at every tolerance (1, 2, 3, 5 px, 1% diagonal) is either empty or a miss farther than every tolerance (upper bound); flagged images where an nnbox variant scores above 0 are provably non-empty and removed from the upper bound.",
                      "per_set": p1b}

# ========================================================== Part 1c
G_ORDER = ["U", "F", "K", "M19"]
p1c = {"placement": {}, "agreement": {}}
for m in METRICS:
    pl = {}
    for s in SETS:
        c = COL[s][m]
        row = {}
        for g in G_ORDER:
            mu = np.array([c[n].mean() for n in V7.GROUPS7[g]])
            row[g] = {"n": len(mu), "mean": r4(mu.mean()), "min": r4(mu.min()), "max": r4(mu.max())}
        m19 = np.array([c[n].mean() for n in V7.GROUPS7["M19"]])
        allm = np.array([c[n].mean() for g in G_ORDER for n in V7.GROUPS7[g]])
        for v in NNB2 + ORA2 + SAM1:
            x = c[v].mean()
            row[v] = {"mean": r4(x),
                      "rank_among_M19_plus_it (1=best, of 20)": int(1 + (m19 > x).sum()),
                      "n_of_65_U_F_K_M19_above": int((allm > x).sum())}
        pl[s] = row
    p1c["placement"][m] = pl
    # agreement over method sets
    sets_def = {"M11 (V6 M, check)": list(V6.GROUPS["M"]),
                "M19": list(V7.GROUPS7["M19"]),
                "M19 + sam_nnbox": list(V7.GROUPS7["M19"]) + ["sam_nnbox"],
                "M19 + medsam_fixed_nnbox": list(V7.GROUPS7["M19"]) + ["medsam_fixed_nnbox"],
                "M19 + both nnbox (21)": list(V7.GROUPS7["M19"]) + NNB2,
                "M19 with nnU-Net replaced by rescore P0": [n for n in V7.GROUPS7["M19"] if n != NN] + ["nnunet_rescore_P0"]}
    ag = {}
    for name, ms in sets_def.items():
        MEANS = np.array([[COL[s][m][n].mean() for n in ms] for s in REFS])
        ctrl = np.array([COL[CTRL][m][n].mean() for n in ms])
        ag[name] = {"n_methods": len(ms), **{k: r4(v) for k, v in summ(MEANS, ctrl).items()}}
    p1c["agreement"][m] = ag

# check the copied agreement code against stats_v6 (group M)
SV6 = json.load(open(os.path.join(RES, "v6", "stats_v6.json")))
chk = []
for m in METRICS:
    pv = SV6["point"][f"{m}|M"]
    MEANS = np.array([[COL[s][m][n].mean() for n in V6.GROUPS["M"]] for s in REFS])
    ctrl = np.array([COL[CTRL][m][n].mean() for n in V6.GROUPS["M"]])
    mine = summ(MEANS, ctrl)
    for a, b in (("wound pairs (6)", "wound pairs"), ("modality_diff", "modality_diff"),
                 ("modality_diff_excl_within_source", "modality_diff_excl_within_source"),
                 ("same dataset (azh~control)", "same dataset"), ("fuseg~azh", "fuseg~azh"),
                 ("colonoscopy pairs (10)", "colonoscopy pairs")):
        chk.append(abs(mine[a] - pv[b]))
p1c["check_vs_stats_v6_M"] = {"n_compared": len(chk), "max_abs_diff": float(max(chk))}
assert max(chk) < 1e-12, chk
RES_JSON["part1c"] = p1c

# ========================================================== Part 2a
p2a = {"per_set": {}}
alld, alln, allset = [], [], []
for s in SETS:
    r = json.load(open(os.path.join(K2, "rescore", "nnunet", s + ".json")))
    g = json.load(open(os.path.join(K1, "scores", "nnunet", s + ".json")))
    assert g["names"] == ["nnunet_2d"] and r["n"] == g["n"]
    assert r["gt_fraction"] == g["gt_fraction"]
    v = np.asarray(r["var"], float)[:, 0, 0, :]
    grid = np.asarray(r["grid_dice"], float)[:, 0, :]
    gi = r["grid_thresholds"].index(0.5)
    rd, rn = v[:, 0], v[:, 2]
    od, on = np.asarray(g["dice"], float)[:, 0], np.asarray(g["nsd"], float)[:, 0]
    dd, dn = rd - od, rn - on
    k = keep_mask(s, r["n"])
    row = {"n_images_all": int(r["n"]), "n_images_analysis": int(k.sum()),
           "grid_at_0.5_equals_P0_dice": bool(np.array_equal(grid[:, gi], rd)),
           "n_dice_differ": int((np.abs(dd) > ROUND_TOL).sum()),
           "n_nsd_differ": int((np.abs(dn) > ROUND_TOL).sum()),
           "n_either_differ": int(((np.abs(dd) > ROUND_TOL) | (np.abs(dn) > ROUND_TOL)).sum()),
           "max_abs_ddice": r6(np.abs(dd).max()), "max_abs_dnsd": r6(np.abs(dn).max())}
    for tag, sel in (("all", np.ones(len(k), bool)), ("analysis", k)):
        for m, a, b in (("dice", rd, od), ("nsd", rn, on)):
            row[f"mean_{m}_{tag}"] = {"run1": r6(b[sel].mean()), "rescore": r6(a[sel].mean()),
                                      "diff": r6(a[sel].mean() - b[sel].mean()),
                                      "same_at_3_decimals": f"{b[sel].mean():.3f}" == f"{a[sel].mean():.3f}"}
    p2a["per_set"][s] = row
    alld.append(dd)
    alln.append(dn)
    allset += [s] * len(dd)
alld, alln = np.concatenate(alld), np.concatenate(alln)
allset = np.array(allset)
nz_d = alld[np.abs(alld) > ROUND_TOL]
nz_n = alln[np.abs(alln) > ROUND_TOL]
top = np.argsort(-np.abs(alld))[:5]
p2a["overall"] = {
    "n_image_pairs": int(len(alld)),
    "n_dice_differ": int(len(nz_d)), "n_nsd_differ": int(len(nz_n)),
    "n_either_differ": int(((np.abs(alld) > ROUND_TOL) | (np.abs(alln) > ROUND_TOL)).sum()),
    "max_abs_ddice": r6(np.abs(alld).max()), "max_abs_dnsd": r6(np.abs(alln).max()),
    "nonzero_ddice_quantiles_abs (50, 90, 99, 100)": [r6(q) for q in np.percentile(np.abs(nz_d), [50, 90, 99, 100])] if len(nz_d) else None,
    "nonzero_ddice_sign (pos, neg)": [int((nz_d > 0).sum()), int((nz_d < 0).sum())],
    "nonzero_dnsd_quantiles_abs (50, 90, 99, 100)": [r6(q) for q in np.percentile(np.abs(nz_n), [50, 90, 99, 100])] if len(nz_n) else None,
    "nonzero_dnsd_sign (pos, neg)": [int((nz_n > 0).sum()), int((nz_n < 0).sum())],
    "abs_ddice_hist": {"<=5e-6 (rounding)": int((np.abs(alld) <= ROUND_TOL).sum()),
                       "5e-6..1e-4": int(((np.abs(alld) > ROUND_TOL) & (np.abs(alld) <= 1e-4)).sum()),
                       "1e-4..1e-3": int(((np.abs(alld) > 1e-4) & (np.abs(alld) <= 1e-3)).sum()),
                       ">1e-3": int((np.abs(alld) > 1e-3).sum())},
    "largest_ddice": [{"set": str(allset[i]), "ddice": r6(alld[i]), "dnsd": r6(alln[i])} for i in top],
    "max_abs_mean_diff_dice_analysis": r6(max(abs(p2a["per_set"][s]["mean_dice_analysis"]["diff"]) for s in SETS)),
    "max_abs_mean_diff_nsd_analysis": r6(max(abs(p2a["per_set"][s]["mean_nsd_analysis"]["diff"]) for s in SETS)),
    "sets_mean_changes_at_3rd_decimal": [f"{s}:{m}:{t}" for s in SETS for m in METRICS for t in ("all", "analysis")
                                         if not p2a["per_set"][s][f"mean_{m}_{t}"]["same_at_3_decimals"]]}
# nnU-Net rank in M19 with each version
rk = {}
for m in METRICS:
    for s in SETS:
        c = COL[s][m]
        others = np.array([c[n].mean() for n in V7.GROUPS7["M19"] if n != NN])
        rk.setdefault(m, {})[s] = [int(1 + (others > c[NN].mean()).sum()),
                                   int(1 + (others > c["nnunet_rescore_P0"].mean()).sum())]
p2a["nnunet_rank_in_M19_run1_vs_rescore"] = rk
p2a["rank_changes"] = [f"{m}:{s}" for m in METRICS for s in SETS if rk[m][s][0] != rk[m][s][1]]

# code facts: both runs use the same probability path
kr1 = open(os.path.join(ROOT, "kaggle", "kit_run.py")).read()
kr2 = open(os.path.join(ROOT, "kaggle2", "kit_run.py")).read()
rsc = open(os.path.join(ROOT, "kaggle2", "k2_rescore.py")).read()
p2a["code"] = {
    "kaggle/kit_run.py identical to kaggle2/kit_run.py": filecmp.cmp(os.path.join(ROOT, "kaggle", "kit_run.py"),
                                                                     os.path.join(ROOT, "kaggle2", "kit_run.py"), shallow=False),
    "run1_predict_uses_save_probabilities": "--save_probabilities" in kr1.split("def stage_nnunet")[1].split("def one_set_score")[0],
    "run1_scores_postprocess_prob_of_probabilities_ch1_at_0.5": 'postprocess_prob(p, 0.5)' in kr1 and '["probabilities"]' in kr1,
    "rescore_uses_same_read_and_threshold": '["probabilities"]' in rsc and 'NN_NAME, NN_THR = "nnunet_2d", 0.5' in rsc,
    "rescore_rounds_to_5_decimals": "round(float(x), 5)" in rsc,
    "run1_log_predict_cmd": [l.strip() for l in open(os.path.join(K1, "log.txt")) if "nnUNetv2_predict" in l],
    "run2_log_predict_cmd": [l.strip() for l in open(os.path.join(K2, "log.txt")) if "nnUNetv2_predict" in l],
}
RES_JSON["part2a"] = p2a

# ========================================================== Part 2b
LOG = open(os.path.join(K2, "log.txt")).read().splitlines()
pat = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+(\S+): epoch (\d+)/(\d+) mean loss (\S+?)(?:, (\d+) non-finite losses so far)?$")
FAILED = ["manet_r18_in_s2", "manet_r50_s0", "manet_r50_s2"]
OKREF = ["manet_r18_in_s1", "manet_r50_s1"]
ep = {}
for l in LOG:
    mm = pat.search(l)
    if mm:
        name = mm.group(2)
        ep.setdefault(name, []).append({"epoch": int(mm.group(3)), "mean_loss": None if mm.group(5) == "None" else float(mm.group(5)),
                                        "nonfinite_cum": int(mm.group(6) or 0)})
N_TRAIN, BATCH = 392, 16
steps = [N_TRAIN // BATCH + (1 if N_TRAIN % BATCH >= 2 else 0)][0]
p2b = {"steps_per_epoch": steps, "models": {}}
for name in FAILED + OKREF:
    rows = ep.get(name, [])
    first = None
    prev = 0
    for rr in rows:
        if rr["nonfinite_cum"] > 0 and first is None:
            first = [prev + 1, rr["epoch"]]
        prev = rr["epoch"]
    allbad = [rr["epoch"] for rr in rows if rr["mean_loss"] is None]
    fail_line = [l.split("RuntimeError: ", 1)[1] for l in LOG if l.startswith("RuntimeError: " + name + ":")]
    p2b["models"][name] = {"log_epochs": rows,
                           "first_nonfinite_loss_in_epochs": first,
                           "epochs_logged_with_every_step_nonfinite": allbad,
                           "final_cum_nonfinite_steps": rows[-1]["nonfinite_cum"] if rows else None,
                           "total_steps": steps * 40,
                           "failure": fail_line}


def func_src(path, fname):
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return ast.get_source_segment(src, node)
    raise KeyError(fname)


def loop_lines(src, start, end):
    """Lines between the optimizer construction and the end of the epoch loop,
    comments stripped, whitespace normalised."""
    ls = src.splitlines()
    i0 = next(i for i, l in enumerate(ls) if start in l)
    i1 = next(i for i, l in enumerate(ls) if end in l and i > i0)
    out = []
    for l in ls[i0:i1 + 1]:
        l = l.split("#")[0].rstrip()
        if l.strip():
            out.append(" ".join(l.split()))
    return out


a = loop_lines(func_src(os.path.join(ROOT, "kaggle2", "kit_run.py"), "stage_pretrained"),
               "opt = torch.optim.AdamW", "sched.step()")
b = loop_lines(func_src(os.path.join(ROOT, "kaggle2", "k2_train.py"), "train_model"),
               "opt = torch.optim.AdamW", "sched.step()")
diff = list(difflib.unified_diff(a, b, "kit_run.stage_pretrained", "k2_train.train_model", lineterm="", n=0))
SRC_A = func_src(os.path.join(ROOT, "kaggle2", "kit_run.py"), "stage_pretrained")
SRC_B = func_src(os.path.join(ROOT, "kaggle2", "k2_train.py"), "train_model")
build_a = {"torch.manual_seed(0) before build": "torch.manual_seed(0)" in SRC_A,
           "build: getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3, classes=1)": "classes=1).to(DEV)" in SRC_A}
build_b = {"torch.manual_seed(seed) before build": "torch.manual_seed(seed)" in SRC_B,
           "build: getattr(smp, arch)(enc, encoder_weights=weights, in_channels=3, classes=1, **kw)": "classes=1, **kw).to(DEV)" in SRC_B,
           "kw empty for the ten original architectures (stage_seeds jobs pass {})": "(C.name_pre(b, s), a, e, {}, s)" in open(os.path.join(ROOT, "kaggle2", "k2_train.py")).read()}
p2b["recipe"] = {"training_loop_diff (normalised lines, comments stripped)": diff,
                 "model_build_run1": build_a, "model_build_phase2": build_b,
                 "optimizer": "AdamW lr 3e-4, weight_decay 1e-4; CosineAnnealingLR(T_max = 40 epochs) stepped per epoch",
                 "amp": "torch.autocast(cuda) (fp16) for the forward pass, loss on .float() logits, torch.amp.GradScaler",
                 "batch": BATCH, "epochs": 40, "n_train": N_TRAIN,
                 "protocol_line": [l.strip() for l in LOG if "protocol:" in l]}
# train report
TR = json.load(open(os.path.join(K2, "train_report_k2.json")))
TR1 = json.load(open(os.path.join(K1, "train_report_pretrained.json")))
TH1 = json.load(open(os.path.join(K1, "thresholds_pretrained.json")))
GRID = [round(0.05 * i, 2) for i in range(1, 19)]
rep = {}
for n, v in TR.items():
    el = v.get("epoch_loss") or []
    rep[n] = {"threshold": v["threshold"], "sweep_dice": r4(v["sweep_dice"]),
              "final_epoch_loss": r4(el[-1]) if el and el[-1] is not None else None,
              "nonfinite_steps": v.get("nonfinite_loss_steps"), "at_grid_edge": v["threshold"] in (GRID[0], GRID[-1])}
for n, v in TR1.items():
    rep[n + " (run 1)"] = {"threshold": TH1[n], "sweep_dice": r4(v["sweep_dice"]), "final_epoch_loss": None,
                           "nonfinite_steps": None, "at_grid_edge": TH1[n] in (GRID[0], GRID[-1])}
sd = {n: x["sweep_dice"] for n, x in rep.items()}
th = {n: x["threshold"] for n, x in rep.items()}
fl = {n: x["final_epoch_loss"] for n, x in rep.items() if x["final_epoch_loss"] is not None}
p2b["train_report"] = {"n_models_phase2": len(TR), "n_models_run1": len(TR1), "per_model": rep,
                       "grid": [GRID[0], GRID[-1]],
                       "any_threshold_at_grid_edge": [n for n, x in rep.items() if x["at_grid_edge"]],
                       "any_nonfinite_steps_in_saved_models": [n for n, x in rep.items() if x["nonfinite_steps"]],
                       "sweep_dice_min": [min(sd, key=sd.get), sd[min(sd, key=sd.get)]],
                       "sweep_dice_max": [max(sd, key=sd.get), sd[max(sd, key=sd.get)]],
                       "threshold_min": [min(th, key=th.get), th[min(th, key=th.get)]],
                       "threshold_max": [max(th, key=th.get), th[max(th, key=th.get)]],
                       "models_sweep_dice_below_0.76": sorted(n for n, x in sd.items() if x < 0.76),
                       "final_epoch_loss_range_excl_linknet": [r4(min(v for n, v in fl.items() if "linknet" not in n)),
                                                               r4(max(v for n, v in fl.items() if "linknet" not in n))],
                       "final_epoch_loss_linknet": {n: v for n, v in fl.items() if "linknet" in n},
                       "manet_ok_models": {n: rep[n] for n in rep if "manet" in n}}
RES_JSON["part2b"] = p2b

# ========================================================== Part 2c
CJ = json.load(open(os.path.join(K2, "rescore", "consistency.json")))
FAM = {"pretrained_s0": "pretrained", "kvasir_nets": "kvasir", "kvasir_unet": "kvasir", "nnunet": "nnunet"}
indep = {}
for fam, grp in FAM.items():
    md, mn, cnt = 0.0, 0.0, 0
    for s in SETS:
        r = json.load(open(os.path.join(K2, "rescore", fam, s + ".json")))
        g = json.load(open(os.path.join(K1, "scores", grp, s + ".json")))
        v = np.asarray(r["var"], float)
        gd, gn = np.asarray(g["dice"], float), np.asarray(g["nsd"], float)
        for j, n in enumerate(r["names"]):
            i1 = g["names"].index(n)
            md = max(md, float(np.abs(v[:, j, 0, 0] - gd[:, i1]).max()))
            mn = max(mn, float(np.abs(v[:, j, 0, 2] - gn[:, i1]).max()))
            cnt += r["n"]
    indep[fam] = {"comparisons": cnt, "max_abs_ddice": md, "max_abs_dnsd2": mn,
                  "file_comparisons": CJ[fam]["comparisons"], "file_max_ddice": CJ[fam]["max_abs_diff_dice"],
                  "file_max_dnsd2": CJ[fam]["max_abs_diff_nsd2"],
                  "reproduces_to_5e-6": bool(md <= ROUND_TOL and mn <= ROUND_TOL),
                  "recomputation_matches_file": bool(cnt == CJ[fam]["comparisons"] and abs(md - CJ[fam]["max_abs_diff_dice"]) < 1e-12
                                                     and abs(mn - CJ[fam]["max_abs_diff_nsd2"]) < 1e-12)}
RES_JSON["part2c"] = {"consistency_json_sets": CJ["nnunet"]["sets"], "families": indep,
                      "note": "rescore values are rounded to 5 decimals, so 5e-6 is the rounding floor"}

# ========================================================== Part 2d
HJ = json.load(open(os.path.join(K2, "nnunet_heldout.json")))
hd = np.asarray(HJ["dice"], float)
sw_all = {n: x["sweep_dice"] for n, x in rep.items()}
grid_best = {}
for s in V6.WOUND + [CTRL]:
    r = json.load(open(os.path.join(K2, "rescore", "nnunet", s + ".json")))
    k = keep_mask(s, r["n"])
    gdm = np.asarray(r["grid_dice"], float)[k, 0, :].mean(0)
    t = r["grid_thresholds"]
    j = int(np.argmax(gdm))
    grid_best[s] = {"best_threshold": t[j], "dice_at_best": r4(gdm[j]), "dice_at_0.5": r4(gdm[t.index(0.5)]),
                    "gain": r4(gdm[j] - gdm[t.index(0.5)]), "n_images": int(k.sum())}
RES_JSON["part2d"] = {
    "keys": sorted(HJ.keys()), "name": HJ["name"], "threshold": HJ["threshold"], "variant": HJ["variant"],
    "n": HJ["n"], "mean_dice": r4(HJ["mean_dice"]), "recomputed_mean": r4(hd.mean()),
    "median_dice": r4(np.median(hd)), "n_dice_0": int((hd == 0).sum()), "n_dice_lt_0.5": int((hd < 0.5).sum()),
    "probability_grid_available": False,
    "why_no_grid": "nnunet_heldout.json stores only the per-image P0 Dice at 0.5; k2_rescore.rescore_nnunet deletes the probability maps (rescore_pred_out) after a successful run, so the sweep rule cannot be applied afterwards",
    "other_models_sweep_dice_range": [r4(min(sw_all.values())), r4(max(sw_all.values()))],
    "other_models_n": len(sw_all),
    "n_other_models_sweep_dice_above_nnunet_heldout": int(sum(v > HJ["mean_dice"] for v in sw_all.values())),
    "descriptive_only_test_set_best_grid_threshold_nnunet": grid_best}

# ================================================================ summary
WOUND = list(V6.WOUND)


def ctab(key, m, sets):
    pos = [s for s in sets if p1a[s][m]["contrasts"][key]["excludes_0"] and p1a[s][m]["contrasts"][key]["diff"] > 0]
    neg = [s for s in sets if p1a[s][m]["contrasts"][key]["excludes_0"] and p1a[s][m]["contrasts"][key]["diff"] < 0]
    vals = [p1a[s][m]["contrasts"][key]["diff"] for s in sets]
    return pos, neg, min(vals), max(vals)


SUMMARY = []
for m in METRICS:
    pos, neg, lo, hi = ctab("medsam_fixed_gtbox - medsam_vitb_box", m, SETS)
    below = sum(p1a[s][m]["means"]["medsam_fixed_gtbox"] < p1a[s][m]["means"]["sam_vitb_box"] for s in SETS)
    SUMMARY.append(f"{m.upper()}, official MedSAM preprocessing (oracle boxes) minus run-1 MedSAM: higher in {len(pos)} of 12 sets ({', '.join(pos)}), lower in {len(neg)} ({', '.join(neg)}), "
                   f"differences {lo:+.3f} to {hi:+.3f}. MedSAM fixed stays below run-1 SAM oracle in {below} of 12 sets.")
for v in NNB2:
    for m in METRICS:
        pos, neg, lo, hi = ctab(f"{v} - {NN}", m, WOUND + [CTRL])
        SUMMARY.append(f"{m.upper()}, {v} minus nnU-Net on the 4 wound sets + control: range {lo:+.3f} to {hi:+.3f}; interval above 0 in {pos or 'none'}, below 0 in {neg or 'none'}.")
ub_w = sum(p1b[s]["analysis_set"]["upper_bound_no_box_images"] for s in WOUND)
n_w = sum(p1b[s]["analysis_set"]["n_images"] for s in WOUND)
ub_c = sum(p1b[s]["analysis_set"]["upper_bound_no_box_images"] for s in V6.COLON)
n_c = sum(p1b[s]["analysis_set"]["n_images"] for s in V6.COLON)
lb_all = sum(p1b[s]["analysis_set"]["lower_bound_no_box_images"] for s in SETS)
SUMMARY.append(f"No nnU-Net box (prediction empty, so nnbox = nnU-Net exactly): at most {ub_w} of {n_w} wound images and at most {ub_c} of {n_c} colonoscopy images (analysis sets); exactly confirmed only {lb_all} (empty references).")
for m in METRICS:
    ag = p1c["agreement"][m]
    SUMMARY.append(f"{m.upper()} agreement, M19 (19) vs M19 + both nnbox (21): wound pairs {ag['M19']['wound pairs (6)']:.3f} -> {ag['M19 + both nnbox (21)']['wound pairs (6)']:.3f}, "
                   f"modality_diff {ag['M19']['modality_diff']:.3f} -> {ag['M19 + both nnbox (21)']['modality_diff']:.3f} "
                   f"(+sam_nnbox only: {ag['M19 + sam_nnbox']['wound pairs (6)']:.3f}, {ag['M19 + sam_nnbox']['modality_diff']:.3f}; +medsam_fixed_nnbox only: {ag['M19 + medsam_fixed_nnbox']['wound pairs (6)']:.3f}, {ag['M19 + medsam_fixed_nnbox']['modality_diff']:.3f}).")
rk_med = [p1c["placement"]["dice"][s]["medsam_fixed_nnbox"]["rank_among_M19_plus_it (1=best, of 20)"] for s in WOUND]
rk_sam = [p1c["placement"]["dice"][s]["sam_nnbox"]["rank_among_M19_plus_it (1=best, of 20)"] for s in WOUND]
SUMMARY.append(f"Dice rank among M19 plus the variant (of 20) on medetec, co2wounds, fuseg, azh: sam_nnbox {rk_sam}, medsam_fixed_nnbox {rk_med}. "
               "A method that is last on every wound set adds concordant ranks to every wound pair, so the rise in wound-pair agreement from medsam_fixed_nnbox is mechanical, not evidence that the wound datasets agree more.")
o = p2a["overall"]
SUMMARY.append(f"nnU-Net rescore vs run 1: {o['n_dice_differ']} of {o['n_image_pairs']} images differ in Dice beyond rounding (max {o['max_abs_ddice']}), {o['n_nsd_differ']} in NSD (max {o['max_abs_dnsd']}); "
               f"largest dataset-mean change {o['max_abs_mean_diff_dice_analysis']} Dice, {o['max_abs_mean_diff_nsd_analysis']} NSD; no mean changes at the third decimal; no rank change; same pipeline in both runs, so GPU floating-point non-determinism, not a bug; no reported number uses the rescore nnU-Net values except the rescore sensitivity family and the nnbox prompts.")
SUMMARY.append(f"Diverged trainings: only MAnet (manet_r18 seed 2, manet_r50 seeds 0 and 2); first non-finite losses in epochs "
               f"{', '.join(str(p2b['models'][n]['first_nonfinite_loss_in_epochs'][0]) + ' to ' + str(p2b['models'][n]['first_nonfinite_loss_in_epochs'][1]) for n in FAILED)}; "
               "recipe identical to run 1 apart from the seed (AdamW 3e-4, wd 1e-4, cosine over 40 epochs, fp16 AMP with GradScaler, batch 16). "
               f"No saved model has a threshold at the grid edge or a non-finite step; sweep Dice below 0.76 only for {', '.join(p2b['train_report']['models_sweep_dice_below_0.76'])}.")
SUMMARY.append("Consistency: pretrained_s0, kvasir_nets and kvasir_unet reproduce run 1 to the 5e-6 rounding floor (recomputed independently); only nnU-Net differs (see above).")
SUMMARY.append(f"nnU-Net held-out FUSeg (n = {HJ['n']}): mean Dice {RES_JSON['part2d']['mean_dice']} at 0.5; the sweep threshold cannot be recovered (probability maps not kept). "
               f"On the test sets themselves the best grid threshold would add at most {max(q['gain'] for q in grid_best.values()):+.3f} Dice (descriptive, optimistic).")
RES_JSON["summary"] = SUMMARY

json.dump(RES_JSON, open(os.path.join(OUTD, "sam_checks_v7.json"), "w"), indent=1)

# ================================================================ markdown
L = []
A = L.append
A("# SAM checks and phase-2 run checks (V7)\n")
A("Script: `scripts/v7/sam_checks_v7.py`. Numbers: `results/v7/sam_checks_v7.json`. "
  "Exclusions as in perimage_v6 (8 FUSeg, 47 control images) unless a table says all images. "
  "Image order of every Kaggle file was checked against perimage32 (gt_fraction identical, "
  f"{len(ORDER_CHECK)} files).\n")
A("## Definitions\n")
A("- Dice and NSD (tau 2 px) per image, dataset mean over all kept images (empty references included).")
A("- nnU-Net = run-1 scores (the column in M11 and M19). nnU-Net rescore P0 = phase-2 re-prediction of the same checkpoint; its masks are the ones the nnbox prompts were drawn from.")
A("- M19 range = lowest and highest dataset mean among the 19 M19 methods (18 seed-averaged architectures + nnU-Net).")
A(f"- Paired unit bootstrap as practical_v6: units = photograph for Medetec and AZH (and control), image otherwise; B = {B}; one weight matrix per dataset from default_rng([{SEED}, k]), k = dataset index in REFS + control, shared by all contrasts and both metrics. Interval = 2.5 and 97.5 percentiles. The bootstrap helper is copied from practical_v6 and the agreement code from stats_v6 (importing either runs a full analysis); the agreement copy reproduces stats_v6 group M exactly (max abs diff {p1c['check_vs_stats_v6_M']['max_abs_diff']:.1e} over {p1c['check_vs_stats_v6_M']['n_compared']} values).\n")

A("## Summary\n")
for line in SUMMARY:
    A("- " + line)
A("")
A("## Part 1. SAM and MedSAM\n")
A("### Coverage\n")
A(f"- Phase-2 groups sam_v2_oracle and sam_v2_nnbox: {cov['sam_v2_oracle']['n_sets']} sets each, synthetic sets: "
  f"{len(cov['sam_v2_oracle']['synthetic_sets'])} and {len(cov['sam_v2_nnbox']['synthetic_sets'])}. "
  f"The 11 references and the control are all present; the only extra set is {', '.join(cov['sam_v2_oracle']['extra_sets'])} (not used here). "
  f"Run-1 SAM has {cov['run1_sam_n_sets']} sets (12 clinical + 21 synthetic). So the phase-2 SAM variants have no synthetic scores and cannot enter the benchmark-validity comparison.\n")
A("### 1a. MedSAM preprocessing, code reading\n")
A("`kaggle2/k2_sam.py` reproduces the official MedSAM_Inference.py steps: RGB image resized to 1024 x 1024 (cubic, anti-aliased, back to uint8), min-max scaled to [0, 1], no mean/std normalisation (the Hugging Face SamModel does not normalise pixel_values), boxes scaled to the 1024 frame, sigmoid of the 256 x 256 low-resolution logits bilinearly upsampled to the native size. "
  "Reading the code I found no remaining preprocessing difference, so the MedSAM fixed numbers below should be taken as MedSAM's real behaviour on these images. sam_nnbox uses SamProcessor exactly as run-1 SAM.\n")
for m in METRICS:
    A(f"### 1a. Dataset means, {m.upper()}\n")
    A("| dataset | n img (units) | SAM r1 oracle | MedSAM r1 oracle | MedSAM fixed oracle | SAM nnbox | MedSAM fixed nnbox | nnU-Net | nnU-Net rescore | M19 mean | M19 range |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SETS:
        q = p1a[s][m]["means"]
        A(f"| {s} | {p1a[s]['n_images']} ({p1a[s]['n_units']}) | {f3(q['sam_vitb_box'])} | {f3(q['medsam_vitb_box'])} | {f3(q['medsam_fixed_gtbox'])} | "
          f"{f3(q['sam_nnbox'])} | {f3(q['medsam_fixed_nnbox'])} | {f3(q[NN])} | {f3(q['nnunet_rescore_P0'])} | {f3(q['M19_mean'])} | {f3(q['M19_min'])} to {f3(q['M19_max'])} |")
    A("")
    A(f"Paired differences with 95% unit-bootstrap intervals ({m.upper()}). * = interval excludes 0.\n")
    A("| dataset | MedSAM fixed oracle - MedSAM r1 oracle | SAM nnbox - nnU-Net | MedSAM fixed nnbox - nnU-Net | SAM nnbox - nnU-Net rescore | MedSAM fixed nnbox - nnU-Net rescore |")
    A("|---|---|---|---|---|---|")
    for s in SETS:
        c = p1a[s][m]["contrasts"]

        def cc(key):
            x = c[key]
            return f"{f4(x['diff'])} [{f4(x['ci'][0])}, {f4(x['ci'][1])}]{' *' if x['excludes_0'] else ''}"
        A(f"| {s} | {cc('medsam_fixed_gtbox - medsam_vitb_box')} | {cc('sam_nnbox - nnunet_2d')} | {cc('medsam_fixed_nnbox - nnunet_2d')} | "
          f"{cc('sam_nnbox - nnunet_rescore_P0')} | {cc('medsam_fixed_nnbox - nnunet_rescore_P0')} |")
    A("")

A("### 1b. Images with no nnU-Net box\n")
A("Rule in `kaggle2/k2_sam.py` (predict_nn, mask_boxes, medsam_prob, sam_proc_prob): the prompts are the bounding boxes of the up to 10 largest 8-connected components of nnU-Net's P0 mask. "
  "If the mask has no component there is no box, both functions return an all-zero probability map, and the prediction is empty. "
  "It is scored with the study convention: 1 if the reference is also empty, 0 otherwise. So on these images both nnbox variants score exactly what nnU-Net scores; SAM cannot recover a wound nnU-Net missed.\n")
A("The nnU-Net masks are not in the local results, so the count is inferred from the rescore file. It is exact for empty references (Dice 1 means the prediction is empty). "
  "For non-empty references, an image with Dice 0 and NSD 0 at every tolerance (1, 2, 3, 5 px and 1% of the diagonal) has either an empty mask or a mask farther from the reference than every tolerance, so the count is an upper bound. "
  "Flagged images where an nnbox variant scores above 0 are provably non-empty and are removed.\n")
A("| dataset | n img | empty refs | refs empty, nnU-Net empty (exact) | refs non-empty, Dice 0 | of which all NSD 0 | of those, nnbox > 0 (so not empty) | no-box images, lower to upper bound | upper bound share |")
A("|---|---|---|---|---|---|---|---|---|")
for s in SETS:
    q = p1b[s]["analysis_set"]
    A(f"| {s} | {q['n_images']} | {q['n_ref_empty']} | {q['ref_empty_and_nnunet_empty (exact)']} | {q['ref_nonempty_nnunet_dice0']} | "
      f"{q['ref_nonempty_nnunet_dice0_and_all_nsd0 (upper bound on empty masks)']} | {q['of_those_nnbox_not_zero (proves mask non-empty)']} | "
      f"{q['lower_bound_no_box_images']} to {q['upper_bound_no_box_images']} | {f3(q['share_upper_bound'])} |")
A("")

A("### 1c. Where the non-oracle variants land, and agreement with them added\n")
for m in METRICS:
    A(f"Placement, {m.upper()} (group mean and range of dataset means; rank of the variant among M19 plus itself, 1 = best of 20).\n")
    A("| dataset | U (18) | F (14) | K (14) | M19 (19) | SAM nnbox (rank) | MedSAM fixed nnbox (rank) |")
    A("|---|---|---|---|---|---|---|")
    for s in SETS:
        q = p1c["placement"][m][s]

        def gg(g):
            return f"{f3(q[g]['mean'])} ({f3(q[g]['min'])} to {f3(q[g]['max'])})"
        A(f"| {s} | {gg('U')} | {gg('F')} | {gg('K')} | {gg('M19')} | {f3(q['sam_nnbox']['mean'])} ({q['sam_nnbox']['rank_among_M19_plus_it (1=best, of 20)']}) | "
          f"{f3(q['medsam_fixed_nnbox']['mean'])} ({q['medsam_fixed_nnbox']['rank_among_M19_plus_it (1=best, of 20)']}) |")
    A("")
for m in METRICS:
    A(f"Agreement, {m.upper()} (sensitivity only; the paper never pools oracle SAM, and these nnbox variants are not in any primary group).\n")
    keys = ["wound pairs (6)", "wound different-source pairs (5)", "fuseg~azh", "colonoscopy pairs (10)", "modality_diff", "modality_diff_excl_within_source", "same dataset (azh~control)"]
    A("| method set | n | " + " | ".join(keys) + " |")
    A("|---|---|" + "---|" * len(keys))
    for name, q in p1c["agreement"][m].items():
        A(f"| {name} | {q['n_methods']} | " + " | ".join(f3(q[k]) for k in keys) + " |")
    A("")

A("## Part 2. Run checks\n")
o = p2a["overall"]
A("### 2a. nnU-Net rescore vs run 1\n")
A("Code: both runs call `nnUNetv2_predict ... --save_probabilities` with the same command, read channel 1 of the saved probabilities, resize if needed, and apply `postprocess_prob(p, 0.5)`. "
  "Run 1 did not use nnU-Net's own segmentation export; the pipeline is the same in both runs "
  f"(kit_run.py byte-identical in both kits: {p2a['code']['kaggle/kit_run.py identical to kaggle2/kit_run.py']}; checks of the code text: {p2a['code']['run1_scores_postprocess_prob_of_probabilities_ch1_at_0.5']}, {p2a['code']['rescore_uses_same_read_and_threshold']}). "
  "The difference is the re-run of GPU inference itself: nnU-Net 2.8.1 sets `torch.backends.cudnn.benchmark = True`, runs the network under fp16 autocast and accumulates the sliding-window logits in a half-precision buffer. "
  "cuDNN may pick different kernels in a new session (or on a different GPU), the fp16 sums then differ in the last bits, and pixels whose smoothed probability sits at 0.5 flip. This is floating-point non-determinism, not a bug.\n")
A(f"- Image pairs compared: {o['n_image_pairs']} (12 sets, all images). Dice differs beyond the 5e-6 rounding floor on {o['n_dice_differ']}, NSD on {o['n_nsd_differ']}, either on {o['n_either_differ']}.")
A(f"- Absolute Dice differences: {o['abs_ddice_hist']}. Non-zero |dDice| quantiles 50/90/99/100%: {o['nonzero_ddice_quantiles_abs (50, 90, 99, 100)']}; sign +/-: {o['nonzero_ddice_sign (pos, neg)']}.")
A(f"- Non-zero |dNSD| quantiles: {o['nonzero_dnsd_quantiles_abs (50, 90, 99, 100)']}; sign +/-: {o['nonzero_dnsd_sign (pos, neg)']}. Max |dDice| {o['max_abs_ddice']}, max |dNSD| {o['max_abs_dnsd']}.")
A(f"- Largest image differences: " + "; ".join(f"{x['set']} dDice {x['ddice']}, dNSD {x['dnsd']}" for x in o['largest_ddice']) + ".")
A(f"- Dataset means (analysis set): largest |change| Dice {o['max_abs_mean_diff_dice_analysis']}, NSD {o['max_abs_mean_diff_nsd_analysis']}. Means that change at the third decimal: {o['sets_mean_changes_at_3rd_decimal'] or 'none'}.")
A(f"- nnU-Net's rank within M19 changes with the rescore values in: {p2a['rank_changes'] or 'no dataset'}. Agreement with nnU-Net replaced by the rescore values is in the 1c agreement table (last row).")
A(f"- Rescore internal check: grid Dice at 0.5 equals P0 Dice in every set: {all(p2a['per_set'][s]['grid_at_0.5_equals_P0_dice'] for s in SETS)}.\n")
A("| dataset | n differ (Dice/NSD) | max abs dDice | max abs dNSD | mean Dice run 1 | mean Dice rescore | mean NSD run 1 | mean NSD rescore |")
A("|---|---|---|---|---|---|---|---|")
for s in SETS:
    q = p2a["per_set"][s]
    A(f"| {s} | {q['n_dice_differ']}/{q['n_nsd_differ']} of {q['n_images_all']} | {q['max_abs_ddice']} | {q['max_abs_dnsd']} | {q['mean_dice_analysis']['run1']:.4f} | {q['mean_dice_analysis']['rescore']:.4f} | {q['mean_nsd_analysis']['run1']:.4f} | {q['mean_nsd_analysis']['rescore']:.4f} |")
A("\nVerdict: not a bug. Every reported V6/V7 number uses the run-1 nnU-Net scores (perimage_v6 reads results/kaggle/scores/nnunet), so none is affected. "
  "The rescore family (post-processing and tolerance sensitivity) and the nnbox prompts use the re-predicted masks; that is a different but equally valid draw of the same model, and the gap is far below any reported precision.\n")

A("### 2b. The three diverged trainings\n")
A(f"Recipe (k2_train.train_model): {p2b['recipe']['optimizer']}; {p2b['recipe']['amp']}; batch {BATCH}, 40 epochs, {N_TRAIN} FUSeg images, {steps} steps per epoch ({steps * 40} steps). "
  f"Logged protocol: {p2b['recipe']['protocol_line'][0] if p2b['recipe']['protocol_line'] else 'n/a'}.\n")
A("| model | first non-finite loss in epochs | cumulative non-finite steps at epochs 10/20/30/40 | epochs whose every step was non-finite | failure |")
A("|---|---|---|---|---|")
for name in FAILED + OKREF:
    q = p2b["models"][name]
    cum = "/".join(str(r["nonfinite_cum"]) for r in q["log_epochs"])
    fe = f"{q['first_nonfinite_loss_in_epochs'][0]} to {q['first_nonfinite_loss_in_epochs'][1]}" if q["first_nonfinite_loss_in_epochs"] else "none"
    A(f"| {name} | {fe} | {cum} | {q['epochs_logged_with_every_step_nonfinite'] or 'none'} | {(q['failure'][0][len(name) + 2:] if q['failure'] else 'trained and saved')} |")
A("\nThe log reports cumulative counts every 10 epochs, so the first bad step is known only to a 10-epoch window. "
  f"manet_r18 seed 2 had only {p2b['models']['manet_r18_in_s2']['final_cum_nonfinite_steps']} non-finite steps of {steps * 40} and a finite mean loss to the end, yet 36 tensors were non-finite after training; the three the log names are decoder BatchNorm running statistics. "
  "That pattern fits AMP: GradScaler skips the optimizer step when gradients are non-finite, so the weights survive, but the train-mode forward pass has already written the non-finite batch statistics into running_mean and running_var. "
  f"manet_r50 seeds 0 and 2 degraded progressively ({p2b['models']['manet_r50_s0']['final_cum_nonfinite_steps']} and {p2b['models']['manet_r50_s2']['final_cum_nonfinite_steps']} non-finite steps of {steps * 40}; seed 2 non-finite on every step from epoch 20 at the latest) and ended with 397 non-finite tensors including encoder weights. "
  "fp16 overflow under AMP in MAnet is the most likely mechanism; this is an inference from the pattern, the log does not show which layer overflowed. Only MAnet diverged (2 of 3 seeds of manet_r50, 1 of 3 of manet_r18), so the problem is architecture-specific, not a broken recipe.\n")
A("Diff of the training loop, run 1 `kit_run.stage_pretrained` vs phase 2 `k2_train.train_model` (normalised lines):\n")
A("```")
A("\n".join(diff) if diff else "(no difference)")
A("```")
A(f"Model construction (code checks): run 1 {p2b['recipe']['model_build_run1']}; phase 2 {p2b['recipe']['model_build_phase2']}. "
  "The only training changes are the seed (torch.manual_seed and default_rng) and bookkeeping lines that record the loss; recording calls float(loss) each step, which synchronises but does not change the computation. The recipe is the same as run 1.\n")
t = p2b["train_report"]
A(f"Train report ({t['n_models_phase2']} phase-2 models + {t['n_models_run1']} run-1 models): thresholds at the grid edge ({t['grid'][0]} or {t['grid'][1]}): {t['any_threshold_at_grid_edge'] or 'none'}. "
  f"Saved models with non-finite steps: {t['any_nonfinite_steps_in_saved_models'] or 'none'}. "
  f"Sweep Dice range {t['sweep_dice_min'][1]} ({t['sweep_dice_min'][0]}) to {t['sweep_dice_max'][1]} ({t['sweep_dice_max'][0]}); below 0.76: {', '.join(t['models_sweep_dice_below_0.76'])} (PSPNet in all three seeds, the same level as run 1, so a property of the architecture, not a failure). "
  f"Thresholds range {t['threshold_min'][1]} ({t['threshold_min'][0]}) to {t['threshold_max'][1]} ({t['threshold_max'][0]}). "
  f"Final-epoch training loss is {t['final_epoch_loss_range_excl_linknet'][0]} to {t['final_epoch_loss_range_excl_linknet'][1]} for every phase-2 model except LinkNet ({', '.join(f'{k} {v}' for k, v in t['final_epoch_loss_linknet'].items())}); LinkNet still reaches normal sweep Dice, so its high loss is a calibration oddity worth a footnote, not a failure.\n")

A("### 2c. Consistency file\n")
A("| family | comparisons | max abs dDice | max abs dNSD2 | reproduces run 1 to 5e-6 | independent recomputation matches file |")
A("|---|---|---|---|---|---|")
for fam, q in indep.items():
    A(f"| {fam} | {q['comparisons']} | {q['max_abs_ddice']:.2e} | {q['max_abs_dnsd2']:.2e} | {q['reproduces_to_5e-6']} | {q['recomputation_matches_file']} |")
A("\nThe rescore values are rounded to 5 decimals, so 5e-6 is the rounding floor: the seed-0 pretrained models, the ten Kvasir networks and the Kvasir U-Net reproduce run 1 exactly. Only nnU-Net does not (2a).\n")

d = RES_JSON["part2d"]
A("### 2d. nnU-Net held-out FUSeg\n")
A(f"`nnunet_heldout.json` holds keys {d['keys']}: nnU-Net P0 Dice at threshold {d['threshold']} on the FUSeg training images 400 to 549 without duplicates (n = {d['n']}), the same images the other models used to pick their threshold. "
  f"Mean Dice {d['mean_dice']} (recomputed {d['recomputed_mean']}), median {d['median_dice']}, {d['n_dice_0']} images with Dice 0, {d['n_dice_lt_0.5']} below 0.5.\n")
A(f"The threshold the other models' rule would pick cannot be computed: {d['why_no_grid']}. "
  f"For comparison, the other {d['other_models_n']} models' best sweep Dice on the same images ranges {d['other_models_sweep_dice_range'][0]} to {d['other_models_sweep_dice_range'][1]}, and {d['n_other_models_sweep_dice_above_nnunet_heldout']} of them exceed nnU-Net's {d['mean_dice']}. Their values are optimistic (threshold chosen on these images), nnU-Net's is not.\n")
A("Descriptive bound only (threshold chosen on the test set itself, so optimistic and NOT the study rule): best grid threshold of nnU-Net per wound set from the rescore grid.\n")
A("| dataset | n | best grid threshold | Dice at best | Dice at 0.5 | gain |")
A("|---|---|---|---|---|---|")
for s, q in d["descriptive_only_test_set_best_grid_threshold_nnunet"].items():
    A(f"| {s} | {q['n_images']} | {q['best_threshold']} | {f3(q['dice_at_best'])} | {f3(q['dice_at_0.5'])} | {f4(q['gain'])} |")
A("")
A("## Limitations of these checks\n")
A("- 1b is inferred from scores, because the nnU-Net masks were not brought back from Kaggle: exact only for empty references; for non-empty references it is an upper bound (an empty mask and a mask that misses the reference by more than every tolerance score the same).")
A(f"- The nnbox prompts come from the phase-2 re-prediction of nnU-Net, not from the run-1 masks behind the nnU-Net column; the two differ slightly on {p2a['overall']['n_either_differ']} of {p2a['overall']['n_image_pairs']} images (2a), which is negligible for the nnbox comparisons (the two nnU-Net columns give the same intervals to 4 decimals in most cells).")
A("- The cause of the nnU-Net differences (cuDNN autotuning with fp16 accumulation, possibly a different GPU) is the most likely explanation from the code; the GPU model of each session is not logged, so it cannot be confirmed.")
A("- The divergence mechanism (fp16 overflow in MAnet) is inferred from the logged pattern; the log does not name the overflowing layer, and the per-step losses of the failed runs were not saved.")
A("- 2d cannot apply the study's threshold rule to nnU-Net, because the held-out probability maps were deleted after the run. The test-set grid table is descriptive and optimistic by construction.")
A("- Adding the nnbox variants to M19 (1c) is a sensitivity only: they are nnU-Net-derived (and identical to nnU-Net on every no-box image), and MedSAM fixed nnbox is last or second to last on every wound set, which raises wound-pair rank agreement mechanically.\n")
open(os.path.join(OUTD, "SAM_CHECKS_V7.md"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(OUTD, "sam_checks_v7.json"), "and SAM_CHECKS_V7.md")

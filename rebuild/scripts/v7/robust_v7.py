#!/usr/bin/env python3
"""V7 robustness of the agreement results to analysis choices, from the run-2
rescore files (results/kaggle2/rescore) and the run-2 SAM score files.

Writes results/v7/robust_v7.json and results/v7/ROBUST_V7.md. Reads only;
modifies no existing file. Run from the rebuild folder:
    python scripts/v7/robust_v7.py

Rescore files (kaggle2/k2_rescore.py, read before use):
  names, thresholds (each model's FUSeg-tuned threshold), grid_thresholds (18),
  grid_dice [image][model][grid]  = dice(postprocess_prob(p, g), gt), i.e. the
                                    P0 pipeline (Gaussian smoothing, threshold g,
                                    small-component removal) at every grid value
  var [image][model][variant][metric], variants P0 P1 P2, metrics
      dice nsd1 nsd2 nsd3 nsd5 nsd_diag1
  P0 = postprocess_prob(p, t)                     (main-results pipeline)
  P1 = postprocess((p >= t))  closing + opening, 5 px ellipse, small components
  P2 = remove_small_components((p >= t))
  Values are rounded to 5 decimals in the files.

Method sets (rescore families only; local CPU methods are not rescored):
  M19  10 original architectures seed-averaged per image over available seeds
       + 8 new architectures seed-averaged + nnunet_2d (perimage_v7.GROUPS M19,
       same members and order; seed members from perimage_v7.SEEDCOLS)
  K11  the members of perimage_v7.GROUPS K that are in the rescore files:
       unet_kv + 10 Kvasir-trained networks. The three Kvasir pixel classifiers
       (logreg_all_kv, gbm_all_kv, logreg_colour_kv) are not rescored
       (k2_rescore.py excludes them), so K11 != the paper's K14.

Conventions copied from scripts/v7/stats_v7.py (itself validated against
stats_v6): agreement = Spearman (Pearson on average ranks) of per-method mean
scores between two datasets; 11 REFS, 55 pairs, wound pairs 6, colonoscopy
pairs 10, modality difference = mean same-modality (16) minus mean
across-modality (39); control = azh vs azh_train_slice. Exclusions from
results/exclusions_v6.json, or the list WB_EXCL names (image indexes dropped). fuseg_r256 holds the same
FUSeg images in the same order (k2_common.fuseg_r256_dir), so the fuseg
exclusions are applied to it by index (n and gt_empty are checked equal).

Method bootstrap: stats_v5 design as in stats_v7.v5_method_takes: methods
resampled with replacement within the group, rng = default_rng(0), B 2000,
percentile 2.5 / 97.5 (nan ignored). The same draws are used for every
condition of a group, so intervals are paired across conditions. The engine is
checked by reproducing stats_v7.json method_only_repro_B2000 for M19 and K
(run-1 scores) exactly.

Unit bootstrap (SAM and nnU-Net comparisons): resampling units from
perimage_v6.unit_of (photograph for medetec and azh, image otherwise), B 2000,
rng = default_rng(7070), dataset mean = sum over drawn units / image count.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage_v6 as V6                                           # noqa: E402
import perimage_v7 as P                                            # noqa: E402

T0 = time.time()
RES = os.path.join(ROOT, "results")
import excl_cfg as C                                               # noqa: E402
OUT = C.V7OUT
RS = os.path.join(RES, "kaggle2", "rescore")
FAMS = ["pretrained_s0", "pretrained_s1", "pretrained_s2", "modern2_s0",
        "modern2_s1", "modern2_s2", "kvasir_nets", "kvasir_unet", "nnunet"]
VARIANTS = ["P0", "P1", "P2"]
METRICS = ["dice", "nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1"]
REFS = list(V6.REFS)
CONTROL = V6.CONTROL
DS = REFS + [CONTROL]                       # 12 rows of every means matrix
NR = len(REFS)
LAB = [V6.MODALITY[r] for r in REFS]
SRC = [V6.SOURCE[r] for r in REFS]
B = 2000
SEED_METHOD = 0
SEED_UNIT = 7070
WOUND = [r for r in REFS if V6.MODALITY[r] == "wound"]
COLON = [r for r in REFS if V6.MODALITY[r] == "colonoscopy"]
FLOOR_NOTE = "M19 mean Dice < 0.1 (stats_v7 floor rule)"


def jload(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------- pairs (stats_v7)
PI, PJ = map(np.array, zip(*itertools.combinations(range(NR), 2)))
SAME = np.array([LAB[i] == LAB[j] for i, j in zip(PI, PJ)])
ACROSS = ~SAME
WSRC = np.array([SRC[i] == SRC[j] for i, j in zip(PI, PJ)])
WOUNDP = np.array([LAB[i] == LAB[j] == "wound" for i, j in zip(PI, PJ)])
COLONP = np.array([LAB[i] == LAB[j] == "colonoscopy" for i, j in zip(PI, PJ)])
FA = np.array([{REFS[i], REFS[j]} == {"fuseg", "azh"} for i, j in zip(PI, PJ)])
assert SAME.sum() == 16 and ACROSS.sum() == 39 and WOUNDP.sum() == 6 and COLONP.sum() == 10
I_AZH, I_CTRL = DS.index("azh"), DS.index(CONTROL)


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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(A[..., mask], axis=-1)


QKEYS = ["wound pairs", "colonoscopy pairs", "modality_diff", "same modality (16)",
         "across modality (39)", "fuseg~azh", "control azh~azh_train_slice"]


def summ(R):
    rp = R[..., PI, PJ]
    return {"wound pairs": nmean(rp, WOUNDP),
            "colonoscopy pairs": nmean(rp, COLONP),
            "modality_diff": nmean(rp, SAME) - nmean(rp, ACROSS),
            "same modality (16)": nmean(rp, SAME),
            "across modality (39)": nmean(rp, ACROSS),
            "fuseg~azh": rp[..., np.where(FA)[0][0]],
            "control azh~azh_train_slice": R[..., I_AZH, I_CTRL]}


def ci(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).any():
        return [None, None]
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def method_takes(n, b=B, seed=SEED_METHOD):
    """stats_v7.v5_method_takes for a single-group pool of n methods."""
    r = np.random.default_rng(seed)
    idx = np.arange(n)
    return np.stack([r.choice(idx, n) for _ in range(b)])


BSD = {}          # key -> bootstrap arrays (paired draws), for delta intervals


def summarise(M, takes, key=None):
    """M: [12, n methods] dataset means (DS order). Point summaries plus the
    method-bootstrap interval of every summary (paired draws `takes`)."""
    pt = {k: float(v) for k, v in summ(corrmat(M)).items()}
    Mb = M[:, takes].transpose(1, 0, 2)                      # [B, 12, n]
    bs = summ(corrmat(Mb))
    out = {}
    for k in QKEYS:
        out[k] = {"est": pt[k], "ci": ci(bs[k])}
    if key is not None:
        BSD[key] = {k: (pt[k], bs[k]) for k in ("wound pairs", "modality_diff", "colonoscopy pairs")}
    return out


def delta(key, ref, q="wound pairs"):
    """Condition minus reference with the paired method-bootstrap interval."""
    a, ab = BSD[key][q]
    b, bb = BSD[ref][q]
    return {"est": a - b, "ci": ci(ab - bb)}


# ------------------------------------------------------------- run-1 reference
D7 = {d: P.load(d, "dice") for d in DS}
N7 = {d: P.load(d, "nsd") for d in DS}

GROUP_M19 = list(P.GROUPS["M19"])
RESCORE_NAMES = set()
for f in FAMS:
    RESCORE_NAMES |= set(jload(os.path.join(RS, f, "medetec.json"))["names"])
GROUP_K11 = [n for n in P.GROUPS["K"] if n in RESCORE_NAMES]
K_MISSING = [n for n in P.GROUPS["K"] if n not in RESCORE_NAMES]
assert K_MISSING == ["logreg_all_kv", "gbm_all_kv", "logreg_colour_kv"], K_MISSING
assert len(GROUP_K11) == 11 and len(GROUP_M19) == 19
GROUPS = {"M19": GROUP_M19, "K11": GROUP_K11}
TAKES = {g: method_takes(len(v)) for g, v in GROUPS.items()}

# individual models behind every group column
MEMBERS = {}
for g in GROUPS.values():
    for c in g:
        MEMBERS[c] = list(P.SEEDCOLS[c]) if c in P.SEEDCOLS else [c]
for c, m in MEMBERS.items():
    for x in m:
        assert x in RESCORE_NAMES, (c, x)


def run1_means(group, metric):
    D = D7 if metric == "dice" else N7
    idx = P.cols(GROUPS[group])
    return np.stack([D[d]["X"][:, idx].mean(0) for d in DS])


CHECKS = {}

# engine check: reproduce stats_v7 method-only B2000 intervals with run-1 data
SV7 = jload(os.path.join(RES, "v7_morning", "stats_v7.json"))  # earlier independent implementation
# The earlier implementation ran under exclusions_v5.json, so the engine check is
# made in that mode only (WB_EXCL=exclusions_v5.json, results/v7_excl_v5).
assert [c.replace("@avg", "") for c in SV7["definitions"]["groups"]["M19"]["members"]] == \
    [c.replace("_in@seeds", "").replace("@seeds", "") for c in GROUP_M19]
assert SV7["definitions"]["groups"]["K"]["members"] == list(P.GROUPS["K"])
eng = {}
for m, key in ((("dice", "dice"), ("nsd", "nsd2")) if C.OLD else ()):
    M = run1_means("M19", m)
    S = summarise(M, TAKES["M19"])
    ref = SV7["bootstrap"][f"{m}|M19"]
    for q, qq in (("wound pairs", "wound pairs"), ("colonoscopy pairs", "colonoscopy pairs"),
                  ("modality_diff", "modality_diff")):
        d_est = abs(S[q]["est"] - ref[qq]["est"])
        d_ci = max(abs(a - b) for a, b in zip(S[q]["ci"], ref[qq]["method_only_repro_B2000"]))
        eng[f"{m}|M19|{q}"] = {"est_diff": d_est, "ci_diff": d_ci}
        assert d_est < 1e-12 and d_ci < 1e-12, (m, q, S[q], ref[qq])
    # K14 with run-1 scores: K14 draws use 14 methods, so check the point only
    D = D7 if m == "dice" else N7
    idx = P.cols(P.GROUPS["K"])
    MK = np.stack([D[d]["X"][:, idx].mean(0) for d in DS])
    s = {k: float(v) for k, v in summ(corrmat(MK)).items()}
    rk = SV7["bootstrap"][f"{m}|K"]
    for q in ("wound pairs", "colonoscopy pairs", "modality_diff"):
        assert abs(s[q] - rk[q]["est"]) < 1e-12
        tk = method_takes(14)
        Sk = summarise(MK, tk)
        assert max(abs(a - b) for a, b in zip(Sk[q]["ci"], rk[q]["method_only_repro_B2000"])) < 1e-12
    eng[f"{m}|K14 point and B2000 interval"] = "equal to stats_v7.json"
CHECKS["engine_vs_stats_v7"] = eng if C.OLD else {"note": "run under exclusions_v5.json only; see results/v7_excl_v5/robust_v7.json"}
print("engine reproduces stats_v7 method-only intervals", flush=True)

# ------------------------------------------------------------- rescore loading
P32 = {d: jload(os.path.join(RES, "perimage32", d + ".json")) for d in DS}
RSETS = DS + ["fuseg_r256"]


def keep_mask(s, n):
    k = np.ones(n, bool)
    for e in V6.EXCL.get("fuseg" if s == "fuseg_r256" else s, []):
        k[e[0]] = False
    return k


RAW = {}          # set -> {"grid": {model: [n,18]}, "var": {model: [n,3,6]}}
THR = {}          # model -> threshold
GRIDT = None
align = {}
for s in RSETS:
    base = "fuseg" if s == "fuseg_r256" else s
    ref = P32[base]
    n = len(ref["ids"])
    keep = keep_mask(s, n)
    RAW[s] = {"grid": {}, "var": {}, "gt_empty": None}
    for f in FAMS:
        j = jload(os.path.join(RS, f, s + ".json"))
        assert j["family"] == f and j["set"] == s
        assert j["variants"] == VARIANTS and j["metrics"] == METRICS
        assert j["n"] == n == len(j["grid_dice"]) == len(j["var"]), (f, s)
        assert j["gt_empty"] == ref["gt_empty"], (f, s, "gt_empty")
        if s != "fuseg_r256":
            assert np.allclose(j["gt_fraction"], ref["gt_fraction"], atol=1e-9), (f, s)
        if GRIDT is None:
            GRIDT = list(j["grid_thresholds"])
        assert j["grid_thresholds"] == GRIDT
        G = np.asarray(j["grid_dice"], float)
        Vv = np.asarray(j["var"], float)
        assert G.shape == (n, len(j["names"]), 18), G.shape
        assert Vv.shape == (n, len(j["names"]), 3, 6), Vv.shape
        assert np.isfinite(G).all() and np.isfinite(Vv).all()
        for k, nm in enumerate(j["names"]):
            assert nm not in RAW[s]["var"], nm
            RAW[s]["grid"][nm] = G[keep, k]
            RAW[s]["var"][nm] = Vv[keep, k]
            t = float(j["thresholds"][k])
            assert THR.setdefault(nm, t) == t, nm
    RAW[s]["gt_empty"] = np.asarray(ref["gt_empty"], bool)[keep]
    RAW[s]["ids"] = [i for i, kk in zip(ref["ids"], keep) if kk]
    RAW[s]["n_kept"] = int(keep.sum())
assert GRIDT == [round(0.05 * i, 2) for i in range(1, 19)], GRIDT
align["note"] = ("every rescore file: same n and gt_empty as results/perimage32 for its set "
                 "(fuseg_r256 vs fuseg), gt_fraction equal to 1e-9 except fuseg_r256 (resampled masks)")
align["n_kept"] = {s: RAW[s]["n_kept"] for s in RSETS}
CHECKS["alignment"] = align

# default threshold on the grid: P0 Dice must equal grid Dice at that threshold
gchk = {"max_abs_diff": 0.0, "models_off_grid": []}
for nm, t in THR.items():
    if t not in GRIDT:
        gchk["models_off_grid"].append(nm)
        continue
    gi = GRIDT.index(t)
    for s in RSETS:
        gchk["max_abs_diff"] = max(gchk["max_abs_diff"], float(np.abs(
            RAW[s]["grid"][nm][:, gi] - RAW[s]["var"][nm][:, 0, 0]).max()))
CHECKS["grid_at_default_threshold_equals_P0_dice"] = gchk
print("grid check", gchk, flush=True)


def col_img(s, c, kind, *sel):
    """Per-image vector of group column c on set s: seed-mean over members.
    kind 'var': sel = (variant index, metric index); kind 'grid': sel = (gi,)."""
    A = [RAW[s][kind][m][(slice(None),) + sel] for m in MEMBERS[c]]
    return np.mean(np.stack(A, 1), 1)


def means(group, kind, *sel, sets=DS):
    return np.stack([np.array([col_img(s, c, kind, *sel).mean() for c in GROUPS[group]])
                     for s in sets])


# rescore P0 vs run-1 scores (dataset means of every group column)
cons = {}
for g in GROUPS:
    for m, mi, r1 in (("dice", 0, "dice"), ("nsd2", 2, "nsd")):
        A = means(g, "var", 0, mi)
        Bm = run1_means(g, r1)
        diff = np.abs(A - Bm)
        nn = GROUPS[g].index("nnunet_2d") if "nnunet_2d" in GROUPS[g] else None
        other = np.delete(diff, nn, axis=1) if nn is not None else diff
        cons[f"{g}|{m}"] = {"max_abs_diff_all_columns": float(diff.max()),
                            "max_abs_diff_excl_nnunet": float(other.max()),
                            "max_abs_diff_nnunet": None if nn is None else float(diff[:, nn].max())}
        s0 = summ(corrmat(A))
        s1 = summ(corrmat(Bm))
        cons[f"{g}|{m}"]["summaries_rescoreP0_vs_run1"] = {
            k: [float(s0[k]), float(s1[k])] for k in ("wound pairs", "colonoscopy pairs", "modality_diff")}
CHECKS["rescore_P0_vs_run1_dataset_means"] = cons

# independent scipy check of one agreement value
A = means("M19", "var", 0, 0)
wp = np.mean([spearmanr(A[DS.index(a)], A[DS.index(b)])[0]
              for a, b in itertools.combinations(WOUND, 2)])
assert abs(wp - summ(corrmat(A))["wound pairs"]) < 1e-12
CHECKS["scipy_wound_pairs_M19_P0_dice"] = float(wp)

# floor datasets
FLOOR = {s: float(means("M19", "var", 0, 0, sets=[s]).mean()) for s in DS}

R = {"definitions": {
    "groups": {g: {"n": len(v), "members": v} for g, v in GROUPS.items()},
    "K11_vs_paper_K14": f"K11 = paper K14 minus {K_MISSING} (not in rescore files)",
    "seed_members": {c: MEMBERS[c] for c in GROUP_M19},
    "agreement": "Spearman (Pearson on average ranks) of per-method mean scores between two datasets over a method set",
    "summaries": "wound pairs = mean of 6 wound-pair agreements; colonoscopy pairs = mean of 10; modality_diff = mean same-modality (16) minus mean across-modality (39) over the 11 references",
    "method_bootstrap": f"methods resampled with replacement within the group, B {B}, default_rng({SEED_METHOD}), stats_v5/v7 design; same draws for every condition of a group; 95% percentile interval",
    "grid_dice": "dice(postprocess_prob(p, g), gt): the P0 pipeline (Gaussian sigma 2 smoothing, threshold g, components < 0.05% of frame removed) at each grid threshold; Dice only",
    "variants": {"P0": "postprocess_prob(p, t): Gaussian smoothing + threshold + small-component removal (main results)",
                 "P1": "postprocess(p >= t): closing then opening with 5 px ellipse, small-component removal",
                 "P2": "remove_small_components(p >= t)"},
    "nsd": "NSD at 1, 2, 3, 5 px and at 1% of the image diagonal; nsd2 is the paper's NSD",
    "exclusions": f"results/{C.EXCL_NAME}; fuseg_r256 uses the fuseg exclusions by index",
    "floor_M19_mean_dice_P0": FLOOR},
    "checks": CHECKS}

# ================================================== 1 post-processing variants
print("1 post-processing", flush=True)
POST = {}
for g in GROUPS:
    for m, mi in (("dice", 0), ("nsd2", 2)):
        for vi, v in enumerate(VARIANTS):
            POST[f"{g}|{m}|{v}"] = summarise(means(g, "var", vi, mi), TAKES[g], key=f"post|{g}|{m}|{v}")
POST_DELTA = {f"{g}|{m}|{v}-P0|{q}": delta(f"post|{g}|{m}|{v}", f"post|{g}|{m}|P0", q)
              for g in GROUPS for m in ("dice", "nsd2") for v in ("P1", "P2")
              for q in ("wound pairs", "modality_diff")}
REORDER = {}
for g in GROUPS:
    for m, mi in (("dice", 0), ("nsd2", 2)):
        Ms = [means(g, "var", vi, mi, sets=RSETS) for vi in range(3)]
        rows = {}
        for k, s in enumerate(RSETS):
            row = {}
            for a, b in ((0, 1), (0, 2), (1, 2)):
                x, y = Ms[a][k], Ms[b][k]
                rho = spearmanr(x, y)[0]
                row[f"{VARIANTS[a]}~{VARIANTS[b]}"] = None if not np.isfinite(rho) else float(rho)
            row["mean_score"] = {VARIANTS[i]: float(Ms[i][k].mean()) for i in range(3)}
            row["max_abs_method_shift_P1_minus_P0"] = float(np.abs(Ms[1][k] - Ms[0][k]).max())
            row["max_abs_method_shift_P2_minus_P0"] = float(np.abs(Ms[2][k] - Ms[0][k]).max())
            rows[s] = row
        REORDER[f"{g}|{m}"] = rows
R["postprocessing"] = {"summaries": POST, "delta_vs_P0": POST_DELTA, "variant_rank_agreement": REORDER}

# ================================================== 2 NSD tolerance
print("2 tolerance", flush=True)
TOL = {}
for g in GROUPS:
    for mi, m in enumerate(METRICS):
        TOL[f"{g}|{m}|P0"] = summarise(means(g, "var", 0, mi), TAKES[g], key=f"tol|{g}|{m}")
TOL_DELTA = {f"{g}|{m}-nsd2|{q}": delta(f"tol|{g}|{m}", f"tol|{g}|nsd2", q)
             for g in GROUPS for m in METRICS if m != "nsd2" for q in ("wound pairs", "modality_diff")}
R["tolerance"] = TOL
R["tolerance_delta_vs_nsd2"] = TOL_DELTA

# ================================================== 3 threshold
print("3 threshold", flush=True)
THRES = {"common": {}, "oracle": {}, "default": {}}
for g in GROUPS:
    THRES["default"][g] = summarise(means(g, "var", 0, 0), TAKES[g], key=f"thr|{g}|default")
    for gi, t in enumerate(GRIDT):
        THRES["common"][f"{g}|{t:.2f}"] = summarise(means(g, "grid", gi), TAKES[g], key=f"thr|{g}|{t:.2f}")
# oracle: per individual model and dataset, best grid threshold by dataset mean Dice
ORC_IND = {}
for s in RSETS:
    ORC_IND[s] = {}
    for nm in RAW[s]["grid"]:
        curve = RAW[s]["grid"][nm].mean(0)
        gi = int(np.argmax(curve))
        ORC_IND[s][nm] = {"thr": GRIDT[gi], "dice": float(curve[gi]),
                          "default_thr": THR[nm], "default_dice": float(RAW[s]["var"][nm][:, 0, 0].mean())}


def oracle_means(group, sets=DS):
    return np.stack([np.array([np.mean([ORC_IND[s][x]["dice"] for x in MEMBERS[c]])
                               for c in GROUPS[group]]) for s in sets])


ORC = {}
for g in GROUPS:
    Mo = oracle_means(g)
    Md = means(g, "var", 0, 0)
    assert (Mo >= Md - 1e-12).all()
    THRES["oracle"][g] = summarise(Mo, TAKES[g], key=f"thr|{g}|oracle")
    indiv = sorted({x for c in GROUPS[g] for x in MEMBERS[c]})
    ORC[g] = {s: {"group_mean_default": float(Md[k].mean()), "group_mean_oracle": float(Mo[k].mean()),
                  "share_models_oracle_at_grid_min_0.05": float(np.mean(
                      [ORC_IND[s][x]["thr"] == GRIDT[0] for x in indiv])),
                  "share_models_oracle_at_grid_max_0.90": float(np.mean(
                      [ORC_IND[s][x]["thr"] == GRIDT[-1] for x in indiv])),
                  "share_models_oracle_thr_differs": float(np.mean(
                      [ORC_IND[s][x]["thr"] != ORC_IND[s][x]["default_thr"] for x in indiv])),
                  "median_oracle_thr": float(np.median([ORC_IND[s][x]["thr"] for x in indiv])),
                  "spearman_default_vs_oracle_methods": float(spearmanr(Md[k], Mo[k])[0])}
              for k, s in enumerate(DS)}
THRES["oracle_detail"] = ORC
THRES["delta_vs_default"] = {f"{g}|{c}|{q}": delta(f"thr|{g}|{c}", f"thr|{g}|default", q)
                             for g in GROUPS for c in ["oracle"] + [f"{t:.2f}" for t in GRIDT]
                             for q in ("wound pairs", "modality_diff")}
# What drives agreement under one common threshold? Probe: per wound dataset,
# Spearman between method means at common t and each method's default
# (FUSeg-tuned) threshold, seed-averaged. The default threshold reflects how
# confident a model's probabilities are, so a high rho means the ranking at
# that common threshold follows calibration. Also the group mean Dice.
PROBE = {}
for g in GROUPS:
    dthr = np.array([np.mean([THR[x] for x in MEMBERS[c]]) for c in GROUPS[g]])
    for gi, t in enumerate(GRIDT):
        Mw = means(g, "grid", gi, sets=WOUND)
        PROBE[f"{g}|{t:.2f}"] = {
            "rho_with_default_threshold": {w: float(spearmanr(Mw[k], dthr)[0]) for k, w in enumerate(WOUND)},
            "mean_rho_wound": float(np.mean([spearmanr(Mw[k], dthr)[0] for k in range(len(WOUND))])),
            "group_mean_dice": {w: float(Mw[k].mean()) for k, w in enumerate(WOUND)}}
    Mw = means(g, "var", 0, 0, sets=WOUND)
    PROBE[f"{g}|default"] = {
        "rho_with_default_threshold": {w: float(spearmanr(Mw[k], dthr)[0]) for k, w in enumerate(WOUND)},
        "mean_rho_wound": float(np.mean([spearmanr(Mw[k], dthr)[0] for k in range(len(WOUND))])),
        "group_mean_dice": {w: float(Mw[k].mean()) for k, w in enumerate(WOUND)}}
THRES["calibration_probe"] = PROBE
THRES["oracle_individual"] = ORC_IND
R["threshold"] = THRES

# ================================================== 4 resolution
print("4 resolution", flush=True)
RESO = {}
for g in GROUPS:
    for m, mi in (("dice", 0), ("nsd2", 2)):
        a = means(g, "var", 0, mi, sets=["fuseg"])[0]
        b = means(g, "var", 0, mi, sets=["fuseg_r256"])[0]
        d = b - a
        M = means(g, "var", 0, mi)
        M2 = M.copy()
        M2[DS.index("fuseg")] = b
        RESO[f"{g}|{m}"] = {
            "fuseg_mean": float(a.mean()), "fuseg_r256_mean": float(b.mean()),
            "shift_mean": float(d.mean()), "shift_min": float(d.min()), "shift_max": float(d.max()),
            "shift_by_method": {c: float(x) for c, x in zip(GROUPS[g], d)},
            "spearman_fuseg_vs_r256": float(spearmanr(a, b)[0]),
            "with_fuseg": summarise(M, TAKES[g], key=f"res|{g}|{m}|native"),
            "with_fuseg_r256": summarise(M2, TAKES[g], key=f"res|{g}|{m}|r256")}
        RESO[f"{g}|{m}"]["delta_r256_minus_native_wound_pairs"] = delta(
            f"res|{g}|{m}|r256", f"res|{g}|{m}|native")
R["resolution"] = RESO

# ================================================== 5 SAM
print("5 SAM", flush=True)
K1 = os.path.join(RES, "kaggle", "scores")
K2 = os.path.join(RES, "kaggle2", "scores")
SAMSRC = [("SAM r1 (gt box)", K1, "sam", "sam_vitb_box"),
          ("MedSAM r1 (gt box)", K1, "sam", "medsam_vitb_box"),
          ("MedSAM official (gt box)", K2, "sam_v2_oracle", "medsam_fixed_gtbox"),
          ("SAM (nnU-Net box)", K2, "sam_v2_nnbox", "sam_nnbox"),
          ("MedSAM official (nnU-Net box)", K2, "sam_v2_nnbox", "medsam_fixed_nnbox"),
          ("nnU-Net r1", K1, "nnunet", "nnunet_2d")]
SAMD = {}
for d in DS:
    ref = P32[d]
    n = len(ref["ids"])
    keep = keep_mask(d, n)
    cols = {}
    for lab, root, grp, nm in SAMSRC:
        j = jload(os.path.join(root, grp, d + ".json"))
        assert j["set"] == d and j["n"] == n and j["gt_empty"] == ref["gt_empty"], (grp, d)
        k = j["names"].index(nm)
        cols[lab] = {"dice": np.asarray(j["dice"], float)[keep, k],
                     "nsd2": np.asarray(j["nsd"], float)[keep, k]}
    cols["nnU-Net rescore P0"] = {"dice": RAW[d]["var"]["nnunet_2d"][:, 0, 0],
                                  "nsd2": RAW[d]["var"]["nnunet_2d"][:, 0, 2]}
    # run-1 nnU-Net here must equal the perimage_v7 column
    assert np.array_equal(cols["nnU-Net r1"]["dice"], D7[d]["X"][:, P.cols(["nnunet_2d"])[0]])
    SAMD[d] = cols
SAM_LABELS = [x[0] for x in SAMSRC] + ["nnU-Net rescore P0"]

rngu = np.random.default_rng(SEED_UNIT)
UW = {}
for d in DS:
    units = np.array([V6.unit_of(d, i) for i in RAW[d]["ids"]])
    _, inv = np.unique(units, return_inverse=True)
    nu = inv.max() + 1
    draw = rngu.integers(0, nu, size=(B, nu))
    W = np.zeros((B, nu))
    np.add.at(W, (np.repeat(np.arange(B), nu), draw.ravel()), 1.0)
    UW[d] = (inv, W, nu)


def unit_ci(d, x):
    inv, W, nu = UW[d]
    us = np.bincount(inv, weights=x, minlength=nu)
    uc = np.bincount(inv, minlength=nu).astype(float)
    return ci((W @ us) / (W @ uc))


DIFFS = [("MedSAM official (gt box)", "MedSAM r1 (gt box)"),
         ("SAM (nnU-Net box)", "nnU-Net rescore P0"),
         ("MedSAM official (nnU-Net box)", "nnU-Net rescore P0"),
         ("SAM (nnU-Net box)", "nnU-Net r1"),
         ("MedSAM official (nnU-Net box)", "nnU-Net r1")]
SAM = {"means": {}, "diffs": {}, "no_box": {}}
for d in DS:
    SAM["means"][d] = {lab: {m: float(SAMD[d][lab][m].mean()) for m in ("dice", "nsd2")}
                       for lab in SAM_LABELS}
    SAM["means"][d]["n_images"] = len(RAW[d]["ids"])
    SAM["diffs"][d] = {}
    for a, b in DIFFS:
        for m in ("dice", "nsd2"):
            x = SAMD[d][a][m] - SAMD[d][b][m]
            c = unit_ci(d, x)
            SAM["diffs"][d][f"{a} - {b}|{m}"] = {"est": float(x.mean()), "ci": c,
                                                 "above0": bool(c[0] > 0), "below0": bool(c[1] < 0)}
    # images with no nnU-Net box. Boxes come from the rescore nnU-Net P0 masks
    # (k2_sam.predict_nn). Empty mask -> no box -> empty prediction -> Dice/NSD 1
    # if the reference is empty, else 0. Masks are not stored locally, so:
    #  empty reference: nnU-Net P0 Dice 1 <=> mask empty (exact)
    #  non-empty reference: an empty mask has Dice 0 and NSD 0 at every
    #  tolerance; that is necessary, not sufficient (upper bound). A candidate
    #  where any nnbox score is > 0 has a box, so it is removed.
    E = RAW[d]["gt_empty"]
    Vn = RAW[d]["var"]["nnunet_2d"][:, 0, :]
    nb = [SAMD[d]["SAM (nnU-Net box)"], SAMD[d]["MedSAM official (nnU-Net box)"]]
    empty_exact = E & (Vn[:, 0] == 1.0)
    for z in nb:      # consistency: no box gives an empty prediction, Dice 1 and NSD 1
        assert (z["dice"][empty_exact] == 1.0).all() and (z["nsd2"][empty_exact] == 1.0).all(), d
    cand = ~E & (Vn == 0.0).all(1)
    proven = np.zeros_like(cand)
    for z in nb:
        proven |= (z["dice"] > 0) | (z["nsd2"] > 0)
    upper = cand & ~proven
    for z in nb:      # on remaining candidates every nnbox score must be 0
        assert (z["dice"][upper] == 0).all() and (z["nsd2"][upper] == 0).all(), d
    SAM["no_box"][d] = {"n_images": int(E.size), "empty_refs": int(E.sum()),
                        "empty_ref_and_nnunet_empty_exact": int(empty_exact.sum()),
                        "nonempty_ref_dice0_allnsd0": int(cand.sum()),
                        "of_which_nnbox_gt0_so_box_present": int((cand & proven).sum()),
                        "nonempty_ref_no_box_upper_bound": int(upper.sum()),
                        "no_box_lower": int(empty_exact.sum()),
                        "no_box_upper": int(empty_exact.sum() + upper.sum())}
R["sam"] = SAM

# ================================================== 6 nnU-Net held-out FUSeg
print("6 held-out", flush=True)
H = jload(os.path.join(RES, "kaggle2", "nnunet_heldout.json"))
hd = np.asarray(H["dice"], float)
assert len(hd) == H["n"]
rngh = np.random.default_rng(SEED_UNIT + 1)
t1 = SAMD["fuseg"]["nnU-Net r1"]["dice"]
t2 = SAMD["fuseg"]["nnU-Net rescore P0"]["dice"]
bh = hd[rngh.integers(0, hd.size, (B, hd.size))].mean(1)
bt = t1[rngh.integers(0, t1.size, (B, t1.size))].mean(1)
R["heldout"] = {"source": "results/kaggle2/nnunet_heldout.json (FUSeg training images 400-549 minus duplicates; "
                          "nnU-Net trained on images 0-392 minus duplicates, threshold fixed 0.5, P0)",
                "n": int(hd.size), "mean_stored": H["mean_dice"], "mean_recomputed_from_rounded": float(hd.mean()),
                "median": float(np.median(hd)), "n_zero": int((hd == 0).sum()),
                "ci_images": ci(bh),
                "fuseg_test_r1": {"n": int(t1.size), "mean": float(t1.mean()), "median": float(np.median(t1)),
                                  "n_zero": int((t1 == 0).sum()), "ci_images": ci(bt)},
                "fuseg_test_rescore_P0_mean": float(t2.mean()),
                "heldout_minus_test_r1": float(hd.mean() - t1.mean()),
                "diff_ci_independent_bootstrap": ci(bh - bt)}
assert abs(R["heldout"]["mean_recomputed_from_rounded"] - H["mean_dice"]) < 1e-5

R["runtime_s"] = round(time.time() - T0, 1)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(os.path.join(OUT, "robust_v7.json"), "w") as f:
    json.dump(clean(R), f, indent=1)
print("wrote robust_v7.json", flush=True)

# ================================================== markdown
L = []


def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def fe(e):
    return f"{f3(e['est'])} [{f3(e['ci'][0])}, {f3(e['ci'][1])}]"


def row(*c):
    L.append("| " + " | ".join(str(x) for x in c) + " |")


base = {g: POST[f"{g}|dice|P0"] for g in GROUPS}
L.append("# ROBUST V7: do analysis choices change the conclusions?")
L.append("")
L.append("Generated by `scripts/v7/robust_v7.py`. Every number is in `results/v7/robust_v7.json`.")
L.append("")
L.append("## Bottom line")
L.append("")
L.append("@@BOTTOM@@")
L.append("")
L.append("## What was analysed")
L.append("")
L.append("- Source: run-2 rescore files (`results/kaggle2/rescore`), built by `kaggle2/k2_rescore.py`. "
         "They hold only the trained Kaggle models, so the local CPU methods (U, F) are not covered.")
L.append(f"- M19: 10 original architectures and 8 new ones, each averaged per image over its available seeds, plus nnU-Net (19 methods, as `perimage_v7.GROUPS['M19']`).")
L.append(f"- K11: the paper's K group without the three Kvasir pixel classifiers ({', '.join(K_MISSING)}), which were not rescored. So K11 has 11 methods, not 14.")
L.append("- Agreement: Spearman correlation of per-method mean scores between two datasets. Wound pairs: mean of the 6 pairs of medetec, co2wounds, fuseg, azh. Colonoscopy pairs: mean of 10. Modality difference: mean of the 16 same-modality pairs minus mean of the 39 across-modality pairs (11 references).")
L.append(f"- Intervals: method bootstrap, methods resampled within the group, {B} draws, seed {SEED_METHOD}; the same draws for every condition. Exclusions from `results/{C.EXCL_NAME}`.")
L.append("- Post-processing variants: P0 = Gaussian smoothing of the probability map, threshold, small-component removal (main results). "
         "P1 = threshold, then closing and opening with a 5 px ellipse, then small-component removal. P2 = threshold and small-component removal only.")
L.append("- `grid_dice` (threshold analysis) is Dice of the P0 pipeline at each grid threshold (`postprocess_prob(p, g)` in `k2_rescore.score_image`), so the threshold analysis keeps P0 post-processing. Dice only; there is no NSD on the grid.")
L.append("- Colonoscopy, PH2 and TN3K are floor datasets for M19 (M19 mean Dice below 0.1): " +
         ", ".join(f"{d} {FLOOR[d]:.3f}" for d in DS if FLOOR[d] < 0.1) +
         ". Rank agreement on them is close to noise for M19, so its colonoscopy-pair numbers carry little information.")
KFLOOR = {d: float(means("K11", "var", 0, 0, sets=[d]).mean()) for d in DS}
R["definitions"]["floor_K11_mean_dice_P0"] = KFLOOR
L.append("- K11 is weak on the wound sets (K11 mean Dice " + ", ".join(f"{d} {KFLOOR[d]:.3f}" for d in WOUND) +
         "), so its wound-pair agreement is computed on low scores; on colonoscopy it is " +
         ", ".join(f"{d} {KFLOOR[d]:.3f}" for d in COLON) + ".")
L.append("")
L.append("## Checks")
L.append("")
if C.OLD:
    L.append("- Bootstrap engine: with run-1 scores it reproduces `results/v7_morning/stats_v7.json` (the earlier independent implementation) exactly (point estimates and method-only B 2000 intervals for M19 wound pairs, colonoscopy pairs and modality difference, and K14 with 14-method draws; max difference 0).")
else:
    L.append("- Bootstrap engine: checked against the earlier independent implementation in the run under the V6 exclusions (`results/v7_excl_v5/ROBUST_V7.md`).")
c = CHECKS["rescore_P0_vs_run1_dataset_means"]
L.append(f"- Rescore P0 vs run-1 dataset means: max abs difference excluding nnU-Net {max(c[k]['max_abs_diff_excl_nnunet'] for k in c):.1e}; nnU-Net {max(c[k]['max_abs_diff_nnunet'] or 0 for k in c):.1e} (re-prediction).")
L.append("- Summaries, rescore P0 vs run 1 (wound pairs / colonoscopy pairs / modality difference):")
for k, v in c.items():
    s = v["summaries_rescoreP0_vs_run1"]
    L.append(f"  - {k}: " + "; ".join(f"{q} {s[q][0]:.3f} vs {s[q][1]:.3f}" for q in s))
L.append(f"- P0 Dice equals the grid Dice at each model's own threshold: max abs difference {gchk['max_abs_diff']:.1e}; models with off-grid thresholds: {len(gchk['models_off_grid'])}.")
L.append(f"- Rescore files align with `results/perimage32` (same n and empty-reference flags; same lesion fraction except fuseg_r256). Kept images: " +
         ", ".join(f"{s} {RAW[s]['n_kept']}" for s in RSETS) + ".")
L.append("")

L.append("## 1. Post-processing")
L.append("")
L.append("Agreement summaries, estimate [95% method-bootstrap interval]. The last column is the change in wound-pair agreement from P0, with its paired interval (same method draws).")
L.append("")
row("set", "metric", "variant", "wound pairs", "colonoscopy pairs", "modality diff", "wound pairs minus P0")
row(*["---"] * 7)
for g in GROUPS:
    for m in ("dice", "nsd2"):
        for v in VARIANTS:
            e = POST[f"{g}|{m}|{v}"]
            dl = "" if v == "P0" else fe(POST_DELTA[f"{g}|{m}|{v}-P0|wound pairs"])
            row(g, m, v, fe(e["wound pairs"]), fe(e["colonoscopy pairs"]), fe(e["modality_diff"]), dl)
L.append("")
nonfloor = [d for d in RSETS if FLOOR.get("fuseg" if d == "fuseg_r256" else d, 1) >= 0.1]
mins = {}
for g in GROUPS:
    for m in ("dice", "nsd2"):
        vals = [(REORDER[f"{g}|{m}"][d][k], d) for d in nonfloor for k in ("P0~P1", "P0~P2")]
        mins[f"{g}|{m}"] = min(vals)
R["postprocessing"]["min_rank_rho_on_nonfloor_sets"] = {k: {"rho": v[0], "dataset": v[1]} for k, v in mins.items()}
L.append("Does post-processing reorder methods? Spearman between variants of the per-method means on each dataset. "
         "On the non-floor sets (" + ", ".join(nonfloor) + ") the lowest value is " +
         "; ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in mins.items()) + ".")
L.append("")
row("dataset", "M19 Dice P0~P1", "P0~P2", "M19 mean Dice P0/P1/P2", "M19 NSD P0~P1", "P0~P2",
    "K11 Dice P0~P1", "P0~P2", "K11 NSD P0~P1", "P0~P2")
row(*["---"] * 10)
for s_ in RSETS:
    a, b = REORDER["M19|dice"][s_], REORDER["K11|dice"][s_]
    an, bn = REORDER["M19|nsd2"][s_], REORDER["K11|nsd2"][s_]
    row(s_, f3(a["P0~P1"]), f3(a["P0~P2"]), "/".join(f3(a["mean_score"][v]) for v in VARIANTS),
        f3(an["P0~P1"]), f3(an["P0~P2"]), f3(b["P0~P1"]), f3(b["P0~P2"]), f3(bn["P0~P1"]), f3(bn["P0~P2"]))
L.append("")

L.append("## 2. NSD tolerance (P0)")
L.append("")
L.append("Last column: change in wound-pair agreement relative to NSD at 2 px (the paper's NSD), paired interval.")
L.append("")
row("set", "metric", "wound pairs", "colonoscopy pairs", "modality diff", "wound pairs minus nsd2")
row(*["---"] * 6)
for g in GROUPS:
    for m in METRICS:
        e = TOL[f"{g}|{m}|P0"]
        dl = "" if m == "nsd2" else fe(TOL_DELTA[f"{g}|{m}-nsd2|wound pairs"])
        row(g, m, fe(e["wound pairs"]), fe(e["colonoscopy pairs"]), fe(e["modality_diff"]), dl)
L.append("")

L.append("## 3. Threshold (Dice, P0 pipeline)")
L.append("")
L.append("Default = each model's own FUSeg-tuned threshold (main results). Oracle = for each individual model and dataset, the grid threshold with the best dataset mean Dice (an upper bound that removes calibration differences); seed-averaged columns average their members' oracle scores. Common = one grid threshold for every model.")
L.append("")
row("set", "threshold", "wound pairs", "colonoscopy pairs", "modality diff", "wound pairs minus default")
row(*["---"] * 6)
for g in GROUPS:
    e = THRES["default"][g]
    row(g, "default", fe(e["wound pairs"]), fe(e["colonoscopy pairs"]), fe(e["modality_diff"]), "")
    e = THRES["oracle"][g]
    row(g, "oracle per dataset", fe(e["wound pairs"]), fe(e["colonoscopy pairs"]), fe(e["modality_diff"]),
        fe(THRES["delta_vs_default"][f"{g}|oracle|wound pairs"]))
L.append("")
L.append("One common threshold for all models. Mean wound Dice = M19 group mean Dice averaged over the 4 wound sets. "
         "Calibration rho = mean over the 4 wound sets of the Spearman between M19 method means and each method's default (FUSeg-tuned) threshold; "
         "a high value means the ranking at that threshold follows how confident each model's probabilities are.")
L.append("")
row("threshold", "M19 wound pairs", "minus default", "M19 modality diff", "mean wound Dice", "calibration rho",
    "K11 wound pairs", "K11 modality diff")
row(*["---"] * 8)
pd_ = PROBE["M19|default"]
row("default", fe(THRES["default"]["M19"]["wound pairs"]), "", fe(THRES["default"]["M19"]["modality_diff"]),
    f3(np.mean(list(pd_["group_mean_dice"].values()))), f3(pd_["mean_rho_wound"]),
    fe(THRES["default"]["K11"]["wound pairs"]), fe(THRES["default"]["K11"]["modality_diff"]))
for t in GRIDT:
    e, k_ = THRES["common"][f"M19|{t:.2f}"], THRES["common"][f"K11|{t:.2f}"]
    pr = PROBE[f"M19|{t:.2f}"]
    row(f"{t:.2f}", fe(e["wound pairs"]), fe(THRES["delta_vs_default"][f"M19|{t:.2f}|wound pairs"]),
        fe(e["modality_diff"]), f3(np.mean(list(pr["group_mean_dice"].values()))), f3(pr["mean_rho_wound"]),
        fe(k_["wound pairs"]), fe(k_["modality_diff"]))
L.append("")
L.append("Oracle effect per dataset (M19): group mean Dice at default and oracle thresholds, share of individual models whose oracle threshold differs from their default, share at the grid minimum 0.05 (the oracle is then limited by the grid), median oracle threshold, and Spearman of method means default vs oracle.")
L.append("")
row("dataset", "default", "oracle", "share thr differs", "share at 0.05", "median oracle thr", "rank rho default~oracle")
row(*["---"] * 7)
for s_ in DS:
    o = ORC["M19"][s_]
    row(s_, f3(o["group_mean_default"]), f3(o["group_mean_oracle"]), f"{o['share_models_oracle_thr_differs']:.2f}",
        f"{o['share_models_oracle_at_grid_min_0.05']:.2f}", f"{o['median_oracle_thr']:.2f}",
        f3(o["spearman_default_vs_oracle_methods"]))
L.append("")

L.append("## 4. Resolution: fuseg vs fuseg_r256 (P0)")
L.append("")
L.append("fuseg_r256 = the same FUSeg test images and masks resampled so the long side is 256 px (image INTER_AREA, mask INTER_NEAREST; `k2_common.fuseg_r256_dir`). Shift = r256 minus native, per method.")
L.append("")
row("set", "metric", "mean native", "mean r256", "shift mean [min, max]", "rank rho native~r256",
    "wound pairs with fuseg", "wound pairs with fuseg_r256", "change [paired]", "fuseg~azh native / r256")
row(*["---"] * 10)
for g in GROUPS:
    for m in ("dice", "nsd2"):
        e = RESO[f"{g}|{m}"]
        row(g, m, f3(e["fuseg_mean"]), f3(e["fuseg_r256_mean"]),
            f"{e['shift_mean']:+.3f} [{e['shift_min']:+.3f}, {e['shift_max']:+.3f}]",
            f3(e["spearman_fuseg_vs_r256"]), fe(e["with_fuseg"]["wound pairs"]),
            fe(e["with_fuseg_r256"]["wound pairs"]), fe(e["delta_r256_minus_native_wound_pairs"]),
            f"{f3(e['with_fuseg']['fuseg~azh']['est'])} / {f3(e['with_fuseg_r256']['fuseg~azh']['est'])}")
L.append("")

L.append("## 5. SAM and MedSAM")
L.append("")
L.append("Mean Dice per dataset (kept images). gt box = one box per reference component (oracle prompt). "
         "nnU-Net box = boxes of nnU-Net's rescore P0 mask (up to 10 largest components).")
L.append("")
short = {"SAM r1 (gt box)": "SAM r1 gt", "MedSAM r1 (gt box)": "MedSAM r1 gt",
         "MedSAM official (gt box)": "MedSAM off. gt", "SAM (nnU-Net box)": "SAM nnbox",
         "MedSAM official (nnU-Net box)": "MedSAM off. nnbox", "nnU-Net r1": "nnU-Net r1",
         "nnU-Net rescore P0": "nnU-Net rescore"}
row("dataset", "n", *[short[x] for x in SAM_LABELS])
row(*["---"] * (2 + len(SAM_LABELS)))
for d in DS:
    row(d, SAM["means"][d]["n_images"], *[f3(SAM["means"][d][x]["dice"]) for x in SAM_LABELS])
L.append("")
L.append(f"Paired differences in mean Dice, [95% unit-bootstrap interval, {B} draws, seed {SEED_UNIT}]. nnbox variants are compared with the nnU-Net masks their boxes came from (rescore P0).")
L.append("")
row("dataset", "MedSAM official - MedSAM r1 (gt box)", "SAM nnbox - nnU-Net", "MedSAM off. nnbox - nnU-Net")
row(*["---"] * 4)
for d in DS:
    x = SAM["diffs"][d]
    row(d, *[fe(x[k + "|dice"]) for k in ("MedSAM official (gt box) - MedSAM r1 (gt box)",
                                          "SAM (nnU-Net box) - nnU-Net rescore P0",
                                          "MedSAM official (nnU-Net box) - nnU-Net rescore P0")])
L.append("")
L.append("Same, NSD (2 px):")
L.append("")
row("dataset", "MedSAM official - MedSAM r1 (gt box)", "SAM nnbox - nnU-Net", "MedSAM off. nnbox - nnU-Net")
row(*["---"] * 4)
for d in DS:
    x = SAM["diffs"][d]
    row(d, *[fe(x[k + "|nsd2"]) for k in ("MedSAM official (gt box) - MedSAM r1 (gt box)",
                                          "SAM (nnU-Net box) - nnU-Net rescore P0",
                                          "MedSAM official (nnU-Net box) - nnU-Net rescore P0")])
L.append("")
L.append("Images with no nnU-Net box. Rule (`k2_sam.predict_nn`): no component in the nnU-Net mask means no prompt and an empty prediction, scored 1 if the reference is empty and 0 otherwise (Dice and NSD). "
         "The nnU-Net masks are not stored locally, so the count is exact only for empty references (nnU-Net Dice 1 there means an empty mask). "
         "For non-empty references, Dice 0 with NSD 0 at every tolerance is necessary for an empty mask but not sufficient, so it gives an upper bound; images where an nnbox variant scores above 0 had a box and are removed.")
L.append("")
row("dataset", "n", "empty refs", "no box, empty ref (exact)", "non-empty ref candidates", "candidates with nnbox > 0", "no box, lower to upper")
row(*["---"] * 7)
for d in DS:
    z = SAM["no_box"][d]
    row(d, z["n_images"], z["empty_refs"], z["empty_ref_and_nnunet_empty_exact"], z["nonempty_ref_dice0_allnsd0"],
        z["of_which_nnbox_gt0_so_box_present"], f"{z['no_box_lower']} to {z['no_box_upper']}")
L.append("")

L.append("## 6. nnU-Net on held-out FUSeg training images")
L.append("")
h = R["heldout"]
L.append(f"- Held-out set: FUSeg training images 400 to 549 minus duplicates (n {h['n']}); nnU-Net was trained on images 0 to 392 minus duplicates, threshold fixed at 0.5, P0 post-processing. These images were the threshold-tuning set of the other trained models.")
L.append(f"- Held-out mean Dice {h['mean_stored']:.4f} (95% image bootstrap [{h['ci_images'][0]:.3f}, {h['ci_images'][1]:.3f}]), median {h['median']:.3f}, {h['n_zero']} images at Dice 0.")
t = h["fuseg_test_r1"]
L.append(f"- FUSeg test (kept {t['n']} images, run-1 scores used in the paper): mean Dice {t['mean']:.4f} ([{t['ci_images'][0]:.3f}, {t['ci_images'][1]:.3f}]), median {t['median']:.3f}, {t['n_zero']} at 0. Rescore P0: {h['fuseg_test_rescore_P0_mean']:.4f}.")
L.append(f"- Held-out minus test: {h['heldout_minus_test_r1']:+.4f} (independent bootstrap [{h['diff_ci_independent_bootstrap'][0]:+.3f}, {h['diff_ci_independent_bootstrap'][1]:+.3f}]).")
L.append("")
L.append("## Limitations")
L.append("")
L.append("- Only trained Kaggle models are rescored. The pooled sets of the paper that include untrained (U) or FUSeg-from-scratch (F) methods, and K14's three pixel classifiers, are not covered here.")
L.append("- The threshold grid is 0.05 to 0.90 in steps of 0.05, and grid Dice exists only for the P0 pipeline and only for Dice.")
L.append("- The oracle threshold is chosen on the very images it is scored on, so it is an optimistic upper bound, not a deployable setting.")
L.append("- The oracle is limited by the grid: on medetec and co2wounds " +
         f"{ORC['M19']['medetec']['share_models_oracle_at_grid_min_0.05']:.0%} and {ORC['M19']['co2wounds']['share_models_oracle_at_grid_min_0.05']:.0%}" +
         " of the individual M19 models have their best threshold at the lowest grid value (0.05), so their true optimum may lie lower and is not covered.")
L.append("- nnU-Net in the rescore files is a re-prediction of the run-1 checkpoint (per-image Dice differs by up to 3.4e-3). The rescore numbers use it; the main paper uses run 1.")
L.append("- The no-box counts for non-empty references are bounds, because the nnU-Net masks were not brought back from Kaggle.")
L.append("- Rescore files store values rounded to 5 decimals.")
L.append("")

# bottom line, from the numbers above. Each qualitative statement is guarded
# by an assert, so the wording cannot drift from the numbers.
def wp(e):
    return e["wound pairs"]["est"]


def inc0(d):
    return d["ci"][0] <= 0 <= d["ci"][1]


# A. post-processing and tolerance: no change in wound-pair agreement distinguishable from 0
assert all(inc0(v) for k, v in POST_DELTA.items() if k.endswith("wound pairs"))
assert all(inc0(v) for k, v in TOL_DELTA.items() if k.endswith("wound pairs"))
# B. modality difference interval above 0 in every condition, both groups
md_all = []
for k, v in POST.items():
    md_all.append(("post " + k, v["modality_diff"]))
for k, v in TOL.items():
    md_all.append(("tol " + k, v["modality_diff"]))
for g in GROUPS:
    md_all.append((f"thr default {g}", THRES["default"][g]["modality_diff"]))
    md_all.append((f"thr oracle {g}", THRES["oracle"][g]["modality_diff"]))
for k, v in THRES["common"].items():
    md_all.append(("thr common " + k, v["modality_diff"]))
for k, v in RESO.items():
    md_all.append(("res native " + k, v["with_fuseg"]["modality_diff"]))
    md_all.append(("res r256 " + k, v["with_fuseg_r256"]["modality_diff"]))
md_lo = min(md_all, key=lambda x: x[1]["ci"][0])
assert md_lo[1]["ci"][0] > 0, md_lo
md_est = {g: [e["est"] for n, e in md_all if f"{g}" in n] for g in GROUPS}
R["modality_diff_all_conditions"] = {"n_conditions": len(md_all), "lowest_lower_bound": {"condition": md_lo[0], **md_lo[1]},
                                     "est_range": {g: [min(v), max(v)] for g, v in md_est.items()}}
# C. oracle lowers wound-pair agreement, change not distinguishable from 0
for g in GROUPS:
    d = THRES["delta_vs_default"][f"{g}|oracle|wound pairs"]
    assert d["est"] < 0 and inc0(d), (g, d)
# D. common thresholds where the M19 / K11 change is above 0
up = {g: [t for t in GRIDT if THRES["delta_vs_default"][f"{g}|{t:.2f}|wound pairs"]["ci"][0] > 0] for g in GROUPS}
R["common_threshold_wound_pairs_delta_ci_above0"] = up
assert up["M19"] == [t for t in GRIDT if t >= 0.45], up
assert up["K11"] == [], up
# E. resolution
for k, v in RESO.items():
    assert inc0(v["delta_r256_minus_native_wound_pairs"]), (k, v["delta_r256_minus_native_wound_pairs"])
# ranges of M19 wound-pair agreement over the non-threshold choices
dice_conds = [wp(POST[f"M19|dice|{v}"]) for v in VARIANTS] + [wp(RESO["M19|dice"]["with_fuseg_r256"]),
                                                              wp(THRES["oracle"]["M19"])]
nsd_conds = [wp(POST[f"M19|nsd2|{v}"]) for v in VARIANTS] + [wp(TOL[f"M19|{m}|P0"]) for m in METRICS[1:]] + \
    [wp(RESO["M19|nsd2"]["with_fuseg_r256"])]
c50 = THRES["common"]["M19|0.50"]["wound pairs"]
d50 = THRES["delta_vs_default"]["M19|0.50|wound pairs"]
pr = {t: PROBE[f"M19|{t}"]["mean_rho_wound"] for t in ("default", "0.50", "0.90")}
cwmax = max(GRIDT, key=lambda t: wp(THRES["common"][f"M19|{t:.2f}"]))
# SAM and held-out
md = [SAM["diffs"][d]["MedSAM official (gt box) - MedSAM r1 (gt box)|dice"] for d in DS]
md_sig = sum(1 for x in md if x["above0"] or x["below0"])
sam_up = {lab: {m: [d for d in DS if SAM["diffs"][d][f"{lab} - nnU-Net rescore P0|{m}"]["above0"]] for m in ("dice", "nsd2")}
          for lab in ("SAM (nnU-Net box)", "MedSAM official (nnU-Net box)")}
R["nnbox_beats_nnunet_ci_above0"] = sam_up
nbw = [SAM["no_box"][d] for d in WOUND]
nbc = [SAM["no_box"][d] for d in COLON]
h = R["heldout"]

bl = []
bl.append("Verdict: post-processing, NSD tolerance and resolution do not change either conclusion. The threshold protocol does change the size of the first one: "
          "a single common threshold for all models raises wound-pair agreement among modern methods, while removing calibration differences (per-dataset oracle threshold) lowers it.")
bl.append("")
bl.append(f"- Wound-pair agreement among modern methods (M19) stays in the range of the main results under every post-processing variant, NSD tolerance, the resampled FUSeg, and the oracle threshold: Dice {min(dice_conds):.3f} to {max(dice_conds):.3f}, NSD {min(nsd_conds):.3f} to {max(nsd_conds):.3f} (main results: Dice {wp(POST['M19|dice|P0']):.3f}, NSD {wp(POST['M19|nsd2|P0']):.3f}). "
          "No change from the main-result setting has a paired 95% interval that excludes 0.")
bl.append(f"- Exception, one common threshold: M19 wound-pair Dice agreement rises with the threshold, and the rise over the default is above 0 for every common threshold from {up['M19'][0]:.2f} to {up['M19'][-1]:.2f}. "
          f"At 0.50 it is {fe(c50)} (change {fe(d50)}); the maximum is {wp(THRES['common'][f'M19|{cwmax:.2f}']):.3f} at {cwmax:.2f}. "
          f"For K11 no common threshold raises it (largest {max(wp(THRES['common'][f'K11|{t:.2f}']) for t in GRIDT):.3f}).")
bl.append(f"- Why: the mean Spearman between M19 method means on the wound sets and each method's FUSeg-tuned threshold is {pr['default']:.2f} at default thresholds, {pr['0.50']:.2f} at a common 0.50 and {pr['0.90']:.2f} at 0.90. "
          "This is consistent with the shared ranking at a high common threshold following how confident each model's probabilities are, a model trait that carries across datasets, rather than the datasets agreeing on which model segments wounds best. "
          f"With each model's best threshold per dataset the agreement falls to {wp(THRES['oracle']['M19']):.3f} (K11 {wp(THRES['oracle']['K11']):.3f}).")
bl.append(f"- Modality structure: the modality difference stays above 0 in all {len(md_all)} conditions for both M19 and K11 (lowest lower bound {md_lo[1]['ci'][0]:.3f}, {md_lo[0]}). "
          f"Estimates range {R['modality_diff_all_conditions']['est_range']['M19'][0]:.3f} to {R['modality_diff_all_conditions']['est_range']['M19'][1]:.3f} (M19) and "
          f"{R['modality_diff_all_conditions']['est_range']['K11'][0]:.3f} to {R['modality_diff_all_conditions']['est_range']['K11'][1]:.3f} (K11). "
          "The pooled sets with untrained or from-scratch methods are not in the rescore files and are not tested here.")
bl.append(f"- Post-processing barely reorders methods in Dice (lowest variant-to-variant rank correlation on a non-floor set: M19 {mins['M19|dice'][0]:.3f}), "
          f"but more in NSD (M19 {mins['M19|nsd2'][0]:.3f} on {mins['M19|nsd2'][1]}; medetec {REORDER['M19|nsd2']['medetec']['P0~P1']:.3f}).")
bl.append(f"- Resolution: resampling FUSeg to 256 px shifts M19 Dice by {RESO['M19|dice']['shift_mean']:+.3f} and NSD by {RESO['M19|nsd2']['shift_mean']:+.3f} on average (a fixed pixel tolerance is looser at lower resolution); "
          f"the method ranking on FUSeg changes more for NSD (rank rho {RESO['M19|nsd2']['spearman_fuseg_vs_r256']:.3f}) than for Dice ({RESO['M19|dice']['spearman_fuseg_vs_r256']:.3f}).")
bl.append(f"- MedSAM: official preprocessing changes MedSAM's reference-box Dice a lot, in both directions ({min(x['est'] for x in md):+.3f} to {max(x['est'] for x in md):+.3f}; interval excludes 0 on {md_sig} of {len(DS)} sets). "
          f"Prompting with nnU-Net boxes beats nnU-Net in Dice only for SAM on {', '.join(sam_up['SAM (nnU-Net box)']['dice']) or 'no set'}; in NSD only for SAM on {', '.join(sam_up['SAM (nnU-Net box)']['nsd2']) or 'no set'}; MedSAM with nnU-Net boxes on {', '.join(sam_up['MedSAM official (nnU-Net box)']['dice'] + sam_up['MedSAM official (nnU-Net box)']['nsd2']) or 'no set'}.")
bl.append(f"- Images without an nnU-Net box (prediction empty, scored 0 on a non-empty reference): {sum(z['no_box_lower'] for z in nbw)} to {sum(z['no_box_upper'] for z in nbw)} of {sum(z['n_images'] for z in nbw)} wound images and "
          f"{sum(z['no_box_lower'] for z in nbc)} to {sum(z['no_box_upper'] for z in nbc)} of {sum(z['n_images'] for z in nbc)} colonoscopy images (exact only for empty references).")
bl.append(f"- nnU-Net on held-out FUSeg training images: Dice {h['mean_stored']:.4f} (n {h['n']}) vs {h['fuseg_test_r1']['mean']:.4f} on the FUSeg test set (n {h['fuseg_test_r1']['n']}); difference {h['heldout_minus_test_r1']:+.4f}, interval [{h['diff_ci_independent_bootstrap'][0]:+.3f}, {h['diff_ci_independent_bootstrap'][1]:+.3f}].")
txt = "\n".join(L).replace("@@BOTTOM@@", "@@BOTTOM@@")
R["_bottom_auto"] = bl
with open(os.path.join(OUT, "ROBUST_V7.md"), "w") as f:
    f.write(txt.replace("@@BOTTOM@@", "\n".join(bl)))
with open(os.path.join(OUT, "robust_v7.json"), "w") as f:
    json.dump(clean(R), f, indent=1)
print(f"done {time.time() - T0:.0f}s", flush=True)

#!/usr/bin/env python3
"""Numbers requested by the statistical (R3) and domain (R2) reviews of
manuscript V6, recomputed under the current exclusions (exclusions_v6.json via
scripts/excl_cfg.py) and written to results/v7/review_v7.json and
results/v7/REVIEW_V7.md (the md is generated here; every number in it is read
from the computed values).

Data: scripts/perimage_v7.py (load, load_synth, GROUPS, SEEDCOLS, units).
Definitions mirror scripts/v7/stats_v7.py, practical_v7.py and bench_v7.py:
  agreement      Spearman (Pearson on average ranks) of per-method mean scores
                 between two image sets over a method set
  wound pairs    the 6 pairs of medetec, co2wounds, fuseg, azh
  seed sets      C16 = P9 + N7, the 16 architectures with 3 training seeds
  units          photograph for Medetec, AZH and the control; image otherwise
Where a quantity already exists in stats_v7.json, practical_v7.json,
bench_v7.json or robust_v7.json, the point estimate is recomputed here and
asserted equal (abs diff <= 1e-12, or equal after the 4-decimal rounding that
the source file applied).

Random seeds (numpy default_rng):
  31001  architecture bootstrap of the P9 and N7 gaps (B 2000)
  31002  half splits of units for shared vs independent halves (300 per dataset)
  6060   units-only bootstrap of the seed gap (B 1000): the exact unit draws of
         stats_v7.py section 3, so its units_only interval is reproduced
  [20261010, k]  unit draws of practical_v7.py (B 2000; k = index of the
         dataset in medetec, co2wounds, fuseg, azh, azh_train_slice), used for
         the selection-loss and miss-rate bootstraps
  31003  candidate bootstrap of the selection losses (B 2000)
  31004  half splits of units for choose-on-one-half selection (500 per dataset)
  31005  architecture resampling for the AZH miss difference (B 2000)
  0      bench_v7.method_takes (method bootstrap of benchmark agreement, B 2000)
  1      split halves of stats_v7.py (500 unit splits), reproduced for checks

Run from the rebuild folder:  nice -n 10 python scripts/v7/review_v7.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
os.environ.setdefault("B", "2000")
import perimage_v7 as V                                            # noqa: E402
import excl_cfg as C                                               # noqa: E402
import bench_v7 as BV                                              # noqa: E402  (main() runs only under __main__)

assert BV.B == 2000, BV.B
T0 = time.time()
OUT = C.V7OUT
TOL = 1e-12

SJ = json.load(open(os.path.join(OUT, "stats_v7.json")))
PJ = json.load(open(os.path.join(OUT, "practical_v7.json")))
BJ = json.load(open(os.path.join(OUT, "bench_v7.json")))
RJ = json.load(open(os.path.join(OUT, "robust_v7.json")))
OVL = json.load(open(os.path.join(C.V7IN, "azh_overlap_v7.json")))
BSC = json.load(open(os.path.join(C.V7IN, "black_share_v7.json")))["sets"]
for name, j in (("stats", SJ), ("practical", PJ), ("bench", BJ)):
    assert j["exclusions"] == C.EXCL_NAME, (name, j["exclusions"], C.EXCL_NAME)

REFS = list(V.REFS)
DS = REFS + [V.CONTROL]
WOUND = list(V.WOUND)
assert WOUND == ["medetec", "co2wounds", "fuseg", "azh"] and REFS[:4] == WOUND
WI = [DS.index(r) for r in WOUND]
NR = len(REFS)
PI, PJ_ = map(np.array, zip(*itertools.combinations(range(NR), 2)))
LAB = [V.MODALITY[r] for r in REFS]
SRC = [V.SOURCE[r] for r in REFS]
WOUNDP = np.array([LAB[i] == LAB[j] == "wound" for i, j in zip(PI, PJ_)])
WOUND_DS = WOUNDP & np.array([SRC[i] != SRC[j] for i, j in zip(PI, PJ_)])
assert WOUNDP.sum() == 6 and WOUND_DS.sum() == 5

VER = []          # verification log


def chk(what, mine, ref, tol=TOL):
    d = abs(float(mine) - float(ref))
    VER.append({"what": what, "recomputed": float(mine), "file": float(ref), "abs_diff": d, "tol": tol})
    assert d <= tol, (what, mine, ref, d)


def chk_r4(what, mine, ref):
    """Source file stored round(x, 4)."""
    VER.append({"what": what, "recomputed": float(mine), "file": float(ref),
                "abs_diff": abs(float(mine) - float(ref)), "tol": "equal after rounding to 4 decimals"})
    assert round(float(mine), 4) == float(ref), (what, mine, ref)


def zr(T):
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def corrmat(T):
    Z = zr(T)
    return np.einsum("...in,...jn->...ij", Z, Z)


def ci(a):
    a = np.asarray(a, float)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def est_ci(est, draws, extra=None):
    o = {"est": float(est), "ci95": ci(draws)}
    if extra:
        o.update(extra)
    return o


def fl(x):
    return float(x)


# ------------------------------------------------------------------ data
print("loading", flush=True)
DATA = {m: {d: V.load(d, m) for d in DS} for m in ("dice", "nsd")}
NC = DATA["dice"][DS[0]]["X"].shape[1]
MEANS = {m: np.stack([DATA[m][d]["X"].mean(0) for d in DS]) for m in DATA}
UINV, UCNT, USUM = {}, {}, {m: {} for m in DATA}
for d in DS:
    _, inv = np.unique(DATA["dice"][d]["units"], return_inverse=True)
    UINV[d] = inv
    UCNT[d] = np.bincount(inv).astype(float)
    for m in DATA:
        USUM[m][d] = np.zeros((UCNT[d].size, NC))
        np.add.at(USUM[m][d], inv, DATA[m][d]["X"])
GCOLS = {g: V.cols(V.GROUPS[g]) for g in V.GROUPS}
SETS = {"U": ["U"], "F": ["F"], "K": ["K"], "M19": ["M19"],
        "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M19": ["U", "M19"]}
SEVEN = list(SETS)


def set_cols(s):
    return np.concatenate([GCOLS[g] for g in SETS[s]])


R = {"exclusions": C.EXCL_NAME,
     "n_images": {d: int(UCNT[d].sum()) for d in DS}, "n_units": {d: int(UCNT[d].size) for d in DS}}
for d in DS:
    chk(f"n_units {d} vs stats_v7", R["n_units"][d], SJ["n_units"][d])
    chk(f"n_images {d} vs stats_v7", R["n_images"][d], SJ["n_images"][d])

# ------------------------------------------------------------------ seed sets
FULL3 = {c: mem for c, mem in V.SEEDCOLS.items() if len(mem) == 3}
P9 = [c for c in V.GROUPS["P10s"] if c in FULL3]
N7 = [c for c in V.GROUPS["N8s"] if c in FULL3]
SEED_SETS = {"P9": P9, "N7": N7, "C16": P9 + N7}
assert len(P9) == 9 and len(N7) == 7
for c in P9:
    assert FULL3[c][0] == c.replace("@seeds", "") and FULL3[c][1].endswith("_s1") and FULL3[c][2].endswith("_s2"), c
for c in N7:
    assert [x[-3:] for x in FULL3[c]] == ["_s0", "_s1", "_s2"], c
SCOL = {k: np.array([[V.NAMES_ALL.index(FULL3[c][s]) for c in v] for s in range(3)])
        for k, v in SEED_SETS.items()}
SEEDPAIRS = [(0, 1), (0, 2), (1, 2)]
XPERM = list(itertools.permutations(range(3), 2))
WPAIRS = list(itertools.combinations(range(4), 2))


def seed_gap(T3):
    """T3 [..., 3 seeds, 4 wound datasets, n] method means. Returns
    (same-dataset cross-seed, cross-dataset cross-seed, gap), each [...].
    Same-dataset: mean over the 4 datasets of the mean over 3 seed pairs.
    Cross-dataset: mean over 6 ordered seed pairs x 6 wound pairs (36)."""
    Z = zr(T3)
    same = np.mean([(Z[..., a, :, :] * Z[..., b, :, :]).sum(-1) for a, b in SEEDPAIRS], 0).mean(-1)
    cross = np.mean([(Z[..., a, i, :] * Z[..., b, j, :]).sum(-1) for a, b in XPERM for i, j in WPAIRS], 0)
    return same, cross, same - cross


def take_means(Mtensor, cols3):
    """Mtensor (D, C) or (B, D, C); cols3 (3, n) or (3, B, n) -> [..., 3, 4, n] over wound datasets."""
    out = []
    for s in range(3):
        c = cols3[s]
        if Mtensor.ndim == 2 and c.ndim == 1:
            out.append(Mtensor[WI][:, c])
        elif Mtensor.ndim == 2:                     # c (B, n)
            out.append(Mtensor[WI][:, c].transpose(1, 0, 2))
        else:                                       # Mtensor (B, D, C), c (n,)
            out.append(Mtensor[:, WI][:, :, c])
    return np.stack(out, -3)


GAPK = "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"

# =================================================================== 1  R3 M3
print("1 R3 M3", flush=True)
md = MEANS["dice"]
SPT = {}
for k in SEED_SETS:
    sm, cr, gp = seed_gap(take_means(md, SCOL[k]))
    SPT[k] = {"same_dataset_cross_seed": fl(sm), "cross_dataset_cross_seed": fl(cr), "gap": fl(gp)}
    ss = SJ["seeds"][f"dice|{k}"]["summary"]
    chk(f"seeds dice|{k} rbar_wound", sm, ss["rbar_wound"]["est"])
    chk(f"seeds dice|{k} crossseed|wound pairs", cr, ss["crossseed|wound pairs"]["est"])
    chk(f"seeds dice|{k} gap", gp, ss[GAPK]["est"])
rng = np.random.default_rng(31001)
B1 = 2000
tP = rng.integers(0, 9, size=(B1, 9))
tN = rng.integers(0, 7, size=(B1, 7))
gP = seed_gap(take_means(md, np.stack([SCOL["P9"][s][tP] for s in range(3)])))[2]
gN = seed_gap(take_means(md, np.stack([SCOL["N7"][s][tN] for s in range(3)])))[2]
dg = gP - gN
# same-session check (P9, all 11 clinical datasets)
Z9 = zr(np.stack([md[:NR][:, SCOL["P9"][s]] for s in range(3)]))      # 3 x 11 x 9
rs = {f"r{a}{b}": (Z9[a] * Z9[b]).sum(-1) for a, b in SEEDPAIRS}       # per dataset
rec = SJ["seeds"]["dice|P9"]["run_effect_check"]
for key in ("r01", "r02", "r12"):
    chk(f"seeds dice|P9 mean_{key}_over_11_refs", rs[key].mean(), rec[f"mean_{key}_over_11_refs"])
R["R3_M3_seed_gap_subgroups"] = {
    "point": SPT,
    "gap_P9_minus_N7": est_ci(SPT["P9"]["gap"] - SPT["N7"]["gap"], dg,
                              {"share_draws_le0": fl(np.mean(dg <= 0)), "B": B1, "rng": 31001,
                               "resampling": "architectures of P9 and of N7 resampled independently with replacement; Dice; wound datasets"}),
    "gap_P9_draws_ci95": ci(gP), "gap_N7_draws_ci95": ci(gN),
    "same_session_P9_11_refs": {
        "mean_r12": fl(rs["r12"].mean()), "mean_r01": fl(rs["r01"].mean()), "mean_r02": fl(rs["r02"].mean()),
        "mean_r01_r02": fl((rs["r01"].mean() + rs["r02"].mean()) / 2),
        "n_datasets_r12_gt_both_r01_r02": int(np.sum((rs["r12"] > rs["r01"]) & (rs["r12"] > rs["r02"]))),
        "n_datasets": NR,
        "per_dataset": {REFS[j]: {k2: fl(v[j]) for k2, v in rs.items()} for j in range(NR)}}}

# =================================================================== 2  R3 M4
print("2 R3 M4", flush=True)
NH = 300
rng = np.random.default_rng(31002)
HALF = {k: {"shared": [], "independent": []} for k in SEED_SETS}
HALF_DS = {k: {} for k in SEED_SETS}
for d in WOUND:
    nu = UCNT[d].size
    perm = np.argsort(rng.random((NH, nu)), axis=1)
    A = np.zeros((NH, nu), bool)
    A[np.arange(NH)[:, None], perm[:, :nu // 2]] = True
    Af, Bf = A.astype(float), (~A).astype(float)
    HA = (Af @ USUM["dice"][d]) / (Af @ UCNT[d])[:, None]      # NH x NC
    HB = (Bf @ USUM["dice"][d]) / (Bf @ UCNT[d])[:, None]
    for k in SEED_SETS:
        ZA = [zr(HA[:, SCOL[k][s]]) for s in range(3)]
        ZB = [zr(HB[:, SCOL[k][s]]) for s in range(3)]
        sh = np.mean([((ZA[a] * ZA[b]).sum(-1) + (ZB[a] * ZB[b]).sum(-1)) / 2 for a, b in SEEDPAIRS], 0)
        ind = np.mean([((ZA[a] * ZB[b]).sum(-1) + (ZB[a] * ZA[b]).sum(-1)) / 2 for a, b in SEEDPAIRS], 0)
        HALF[k]["shared"].append(sh.mean())
        HALF[k]["independent"].append(ind.mean())
        HALF_DS[k][d] = {"shared": fl(sh.mean()), "independent": fl(ind.mean())}
# units-only bootstrap: exactly the unit draws of stats_v7 section 3
B_H = 1000
rng = np.random.default_rng(6060)
UM = np.zeros((B_H, len(DS), NC))
for k, d in enumerate(DS):
    nu = UCNT[d].size
    draw = rng.integers(0, nu, size=(B_H, nu))
    Wt = np.zeros((B_H, nu))
    np.add.at(Wt, (np.repeat(np.arange(B_H), nu), draw.ravel()), 1.0)
    UM[:, k, :] = (Wt @ USUM["dice"][d]) / (Wt @ UCNT[d])[:, None]
UO = {}
for k in SEED_SETS:
    g = seed_gap(take_means(UM, SCOL[k]))[2]
    ref = SJ["seeds"][f"dice|{k}"]["summary"][GAPK]["units_only"]
    chk(f"seeds dice|{k} gap units_only interval lo (draws reproduced)", ci(g)[0], ref[0])
    chk(f"seeds dice|{k} gap units_only interval hi (draws reproduced)", ci(g)[1], ref[1])
    UO[k] = {"est": SPT[k]["gap"], "mean_of_draws": fl(np.mean(g)), "median_of_draws": fl(np.median(g)),
             "mean_minus_est": fl(np.mean(g) - SPT[k]["gap"]), "ci95": ci(g)}
R["R3_M4_shared_images"] = {
    "half_splits": {k: {"shared_halves": fl(np.mean(HALF[k]["shared"])),
                        "independent_halves": fl(np.mean(HALF[k]["independent"])),
                        "shared_minus_independent": fl(np.mean(HALF[k]["shared"]) - np.mean(HALF[k]["independent"])),
                        "per_dataset": HALF_DS[k]} for k in SEED_SETS},
    "half_splits_setup": {"n_splits_per_dataset": NH, "rng": 31002, "metric": "dice", "datasets": WOUND},
    "units_only_bootstrap_gap": UO,
    "units_only_setup": {"B": B_H, "rng": 6060, "what": "units resampled within every dataset (stats_v7 section-3 draws), architectures fixed"}}

# =================================================================== 3  R3 M5
print("3 R3 M5", flush=True)
PSEED = 20261010
PSETS = WOUND + [V.CONTROL]


def boot_weights(units, rg, b=2000):
    u, inv = np.unique(units, return_inverse=True)
    draws = rg.integers(0, len(u), size=(b, len(u)))
    counts = np.stack([np.bincount(x, minlength=len(u)) for x in draws]).astype(float)
    return counts[:, inv]


WP = {s: boot_weights(DATA["dice"][s]["units"], np.random.default_rng([PSEED, k])) for k, s in enumerate(PSETS)}
NN = "nnunet_2d"
CANDS = {"M19": list(V.GROUPS["M19"])}
for k in (0, 1, 2):
    CANDS[f"M19_s{k}"] = list(V.GROUPS[f"P10_s{k}"]) + list(V.GROUPS[f"N8_s{k}"]) + [NN]


def sel_losses(Mb):
    """Mb: dict wound dataset -> [..., n candidates] means. Returns chooser
    loss (mean over 12 off-diagonal choose-on-a / deploy-on-b), random loss
    (mean over 4 datasets of best minus mean candidate), and the matrix."""
    best = {b: Mb[b].max(-1) for b in WOUND}
    ch = {a: np.argmax(Mb[a], -1) for a in WOUND}
    mat = {a: {b: best[b] - np.take_along_axis(Mb[b], np.asarray(ch[a])[..., None], -1)[..., 0]
               for b in WOUND} for a in WOUND}
    off = np.mean([mat[a][b] for a in WOUND for b in WOUND if a != b], 0)
    rnd = np.mean([best[b] - Mb[b].mean(-1) for b in WOUND], 0)
    return off, rnd, mat


SEL = {}
for k, cand in CANDS.items():
    cc = V.cols(cand)
    Mp = {b: DATA["dice"][b]["X"][:, cc].mean(0) for b in WOUND}
    off, rnd, mat = sel_losses(Mp)
    ps = PJ["selection"][k]
    chk(f"practical selection/{k}/offdiag_mean", off, ps["offdiag_mean"])
    chk(f"practical selection/{k}/random_mean_over_B", rnd, ps["random_mean_over_B"])
    for b in WOUND:
        chk(f"practical selection/{k}/random_choice_expected_loss/{b}", np.mean(Mp[b].max() - Mp[b]),
            ps["random_choice_expected_loss"][b])
        for a in WOUND:
            chk(f"practical selection/{k}/matrix/{a}/{b}", mat[a][b], ps["matrix_rows_choose_on_cols_deploy_on"][a][b])
    SEL[k] = {"n_candidates": len(cand), "chooser_loss": fl(off), "random_loss": fl(rnd),
              "chooser_minus_random": fl(off - rnd),
              "chosen_on": {a: cand[int(np.argmax(Mp[a]))] for a in WOUND}}
    if k == "M19":
        fu = np.mean([mat["fuseg"][b] for b in WOUND if b != "fuseg"])
        fu_file = np.mean([ps["matrix_rows_choose_on_cols_deploy_on"]["fuseg"][b] for b in WOUND if b != "fuseg"])
        chk("practical selection/M19 FUSeg row mean over the other 3", fu, fu_file)
        SEL[k]["choose_on_fuseg_mean_loss_other3"] = fl(fu)
        SEL[k]["choose_on_fuseg_loss"] = {b: fl(mat["fuseg"][b]) for b in WOUND if b != "fuseg"}
        SEL[k]["row_mean_loss_other3"] = {a: fl(np.mean([mat[a][b] for b in WOUND if b != a])) for a in WOUND}
        # unit bootstrap (practical_v7 draws), paired
        Mb = {b: (WP[b] @ DATA["dice"][b]["X"][:, cc]) / WP[b].sum(1, keepdims=True) for b in WOUND}
        offb, rndb, _ = sel_losses(Mb)
        SEL[k]["unit_bootstrap"] = {"B": 2000, "rng": "practical_v7 unit draws default_rng([20261010, k])",
                                    "chooser_loss": est_ci(off, offb), "random_loss": est_ci(rnd, rndb),
                                    "chooser_minus_random": est_ci(off - rnd, offb - rndb,
                                                                   {"share_draws_ge0": fl(np.mean(offb - rndb >= 0))})}
        # candidate bootstrap
        rg = np.random.default_rng(31003)
        tk = rg.integers(0, len(cand), size=(2000, len(cand)))
        Mc = {b: Mp[b][tk] for b in WOUND}
        offc, rndc, _ = sel_losses(Mc)
        SEL[k]["candidate_bootstrap"] = {"B": 2000, "rng": 31003,
                                         "chooser_loss": est_ci(off, offc), "random_loss": est_ci(rnd, rndc),
                                         "chooser_minus_random": est_ci(off - rnd, offc - rndc,
                                                                        {"share_draws_ge0": fl(np.mean(offc - rndc >= 0))})}
        # choose on one half, deploy on the other half of the same dataset
        rg = np.random.default_rng(31004)
        hs = {}
        for d in WOUND:
            nu = UCNT[d].size
            perm = np.argsort(rg.random((500, nu)), axis=1)
            A = np.zeros((500, nu), bool)
            A[np.arange(500)[:, None], perm[:, :nu // 2]] = True
            Af, Bf = A.astype(float), (~A).astype(float)
            HA = ((Af @ USUM["dice"][d]) / (Af @ UCNT[d])[:, None])[:, cc]
            HB = ((Bf @ USUM["dice"][d]) / (Bf @ UCNT[d])[:, None])[:, cc]
            ia, ib = np.argmax(HA, 1), np.argmax(HB, 1)
            lab = HB.max(1) - HB[np.arange(500), ia]
            lba = HA.max(1) - HA[np.arange(500), ib]
            hs[d] = {"loss_mean": fl(np.mean((lab + lba) / 2)), "loss_A_to_B_mean": fl(lab.mean()),
                     "loss_B_to_A_mean": fl(lba.mean()), "share_same_pick_both_halves": fl(np.mean(ia == ib))}
        SEL[k]["half_split_same_dataset"] = {"n_splits": 500, "rng": 31004, "per_dataset": hs,
                                             "range": [min(v["loss_mean"] for v in hs.values()),
                                                       max(v["loss_mean"] for v in hs.values())]}
R["R3_M5_selection"] = SEL
R["R3_M5_selection"]["single_seed_advantage_range"] = [
    min(SEL[f"M19_s{k}"]["random_loss"] - SEL[f"M19_s{k}"]["chooser_loss"] for k in range(3)),
    max(SEL[f"M19_s{k}"]["random_loss"] - SEL[f"M19_s{k}"]["chooser_loss"] for k in range(3))]

# =================================================================== 4  R3 m7
print("4 R3 m7", flush=True)
M7 = {"U": ("co2wounds", "azh"), "K": ("azh", "fuseg"), "M19": ("co2wounds", "fuseg")}


def f_m7(pool):
    a, b = M7[pool]

    def fn(Z):
        ba = float(np.mean([BV.c(Z, s, a) for s in BV.GB[40]]))
        bb = float(np.mean([BV.c(Z, s, b) for s in BV.GB[40]]))
        return {"a": ba, "b": bb, "diff": ba - bb}
    return fn


M7R = {}
for pool, (a, b) in M7.items():
    res = BV.mboot(pool, WOUND + BV.GB[40], f_m7(pool), BV.method_takes(pool, 0), "v7")
    rel = BJ["rel"][pool]["40"]
    chk(f"bench rel/{pool}/40/{a}|bench", res["a"]["est"], rel[f"{a}|bench"]["est"])
    chk(f"bench rel/{pool}/40/{b}|bench", res["b"]["est"], rel[f"{b}|bench"]["est"])
    chk(f"bench rel/{pool}/40/{a}|bench interval lo (same method draws)", res["a"]["lo"], rel[f"{a}|bench"]["lo"])
    chk(f"bench rel/{pool}/40/{a}|bench interval hi (same method draws)", res["a"]["hi"], rel[f"{a}|bench"]["hi"])
    M7R[pool] = {"minuend": a, "subtrahend": b,
                 "bench_minuend": {"est": res["a"]["est"], "ci95": [res["a"]["lo"], res["a"]["hi"]]},
                 "bench_subtrahend": {"est": res["b"]["est"], "ci95": [res["b"]["lo"], res["b"]["hi"]]},
                 "difference": {"est": res["diff"]["est"], "ci95": [res["diff"]["lo"], res["diff"]["hi"]]}}
R["R3_m7_per_reference_benchmark_differences"] = {
    "setup": "bench_v7 definitions: generator B amp 40, agreement per reference = mean over synthetic seeds 2024, 7, 101; "
             "paired method bootstrap bench_v7.method_takes(pool, 0), B 2000; U from pools_v5 means, K and M19 per-image",
    **M7R}

# =================================================================== 5  R3 m9
print("5 R3 m9", flush=True)
ARCH = {}
for col, mem in V.SEEDCOLS.items():
    ARCH[col.split("@")[0]] = list(mem)
ORIG = [c.split("@")[0] for c in V.GROUPS["P10s"]]
NEW = [c.split("@")[0] for c in V.GROUPS["N8s"]]
D = DATA["dice"]["azh"]
ne = ~D["empty"]
allm = [m for a in ORIG + NEW for m in ARCH[a]] + [NN]
Xa = D["X"][:, V.cols(allm)]
Mi = (ne[:, None] & (Xa < 0.1)).astype(float)
mr = Mi.sum(0) / ne.sum()
Wa = WP["azh"]
BMR = (Wa @ Mi) / (Wa @ ne.astype(float))[:, None]
colix = {m: j for j, m in enumerate(allm)}
arch_pt = {a: mr[[colix[m] for m in ARCH[a]]].mean() for a in ORIG + NEW}
arch_dr = {a: BMR[:, [colix[m] for m in ARCH[a]]].mean(1) for a in ORIG + NEW}
p10, n8 = np.mean([arch_pt[a] for a in ORIG]), np.mean([arch_pt[a] for a in NEW])
p10d, n8d = np.mean([arch_dr[a] for a in ORIG], 0), np.mean([arch_dr[a] for a in NEW], 0)
g = PJ["miss"]["0.1"]["azh"]["groups"]
chk("practical miss/0.1/azh/groups/P10_archmean/est", p10, g["P10_archmean"]["est"])
chk("practical miss/0.1/azh/groups/N8_archmean/est", n8, g["N8_archmean"]["est"])
chk("practical miss/0.1/azh/groups/P10_archmean/ci lo (unit draws reproduced)", ci(p10d)[0], g["P10_archmean"]["ci"][0])
chk("practical miss/0.1/azh/groups/N8_archmean/ci hi (unit draws reproduced)", ci(n8d)[1], g["N8_archmean"]["ci"][1])
rg = np.random.default_rng(31005)
AO = np.stack([arch_dr[a] for a in ORIG])         # 10 x B
AN = np.stack([arch_dr[a] for a in NEW])          # 8 x B
to = rg.integers(0, 10, size=(2000, 10))
tn = rg.integers(0, 8, size=(2000, 8))
bi = np.arange(2000)
dd_arch = AO[to, bi[:, None]].mean(1) - AN[tn, bi[:, None]].mean(1)
R["R3_m9_azh_miss_original_minus_new"] = {
    "P10_archmean": fl(p10), "N8_archmean": fl(n8),
    "difference_units_only": est_ci(p10 - n8, p10d - n8d, {"B": 2000, "rng": "practical_v7 AZH unit draws default_rng([20261010, 3])"}),
    "difference_units_and_architectures": est_ci(p10 - n8, dd_arch,
                                                 {"B": 2000, "rng": "unit draws as above; architectures default_rng(31005), 10 original and 8 new resampled independently, one architecture draw per unit draw"}),
    "definition": "miss = Dice < 0.1 on a non-empty reference; per-model miss rate = misses / non-empty images; architecture mean over its available seeds; group = mean over architectures"}

# =================================================================== 6  R3 m17
print("6 R3 m17", flush=True)
c19s = V.cols(V.GROUPS["M19"], V.SYNTH_NAMES)
c19 = V.cols(V.GROUPS["M19"])
SX = {s: V.load_synth(s)["X"][:, c19s] for s in BV.GB[40]}
smeans = {s: SX[s].mean(0) for s in SX}
seedpair = [spearmanr(smeans[a], smeans[b])[0] for a, b in itertools.combinations(BV.GB[40], 2)]
pooled = np.concatenate([SX[s] for s in BV.GB[40]], 0)
assert pooled.shape[0] == 135
pm = pooled.mean(0)
wmean = {r: MEANS["dice"][DS.index(r), c19] for r in WOUND}
pooled_ag = {r: fl(spearmanr(pm, wmean[r])[0]) for r in WOUND}
perseed_ag = np.mean([[spearmanr(smeans[s], wmean[r])[0] for r in WOUND] for s in BV.GB[40]])
chk("bench rel/M19/40/mean|bench", perseed_ag, BJ["rel"]["M19"]["40"]["mean|bench"]["est"])
R["R3_m17_synthetic_reliability"] = {
    "between_synthetic_seeds_mean_spearman": fl(np.mean(seedpair)),
    "between_synthetic_seeds_pairs": {f"{a}~{b}": fl(v) for (a, b), v in zip(itertools.combinations(BV.GB[40], 2), seedpair)},
    "pooled_135_images_vs_wound_mean": fl(np.mean(list(pooled_ag.values()))),
    "pooled_135_images_vs_wound": pooled_ag,
    "mean_over_seeds_vs_wound_mean": fl(perseed_ag),
    "n_methods": len(c19), "n_images_pooled": int(pooled.shape[0])}

# =================================================================== 7  R3 M1
print("7 R3 M1", flush=True)
pv = {s: SJ["exact"][f"dice|{s}"]["full_55"]["p"] for s in SEVEN}
order = sorted(SEVEN, key=lambda s: pv[s])
mtests = len(order)
holm, run = {}, 0.0
for i, s in enumerate(order):
    run = max(run, min(1.0, (mtests - i) * pv[s]))
    holm[s] = run
R["R3_M1_holm"] = {"raw_p": pv, "holm_adjusted_p": holm, "largest_adjusted_p": max(holm.values()),
                   "largest_adjusted_set": max(holm, key=holm.get), "n_tests": mtests,
                   "source": "stats_v7.json exact/dice|<set>/full_55/p (exact relabelling test, 55 pairs, Dice, modality difference)",
                   "all_below_0.05": bool(max(holm.values()) < 0.05)}

# =================================================================== 8  R3 m16 (and 13, 14 helpers)
print("8 R3 m16", flush=True)
PT = {}
for m in ("dice", "nsd"):
    for s in SEVEN:
        Rm = corrmat(MEANS[m][:NR][:, set_cols(s)])
        rp = Rm[PI, PJ_]
        PT[f"{m}|{s}"] = {"wound pairs": fl(rp[WOUNDP].mean()), "wound different-source pairs": fl(rp[WOUND_DS].mean()),
                          "_R": Rm}
        chk(f"stats point/{m}|{s}/wound pairs", PT[f"{m}|{s}"]["wound pairs"], SJ["point"][f"{m}|{s}"]["wound pairs"])
        chk(f"stats point/{m}|{s}/wound different-source pairs", PT[f"{m}|{s}"]["wound different-source pairs"],
            SJ["point"][f"{m}|{s}"]["wound different-source pairs"])
R["R3_m16_different_source_wound_pairs"] = {
    s: {"six_pairs": PT[f"dice|{s}"]["wound pairs"], "five_different_source_pairs": PT[f"dice|{s}"]["wound different-source pairs"],
        "five_minus_six": PT[f"dice|{s}"]["wound different-source pairs"] - PT[f"dice|{s}"]["wound pairs"]} for s in SEVEN}

# =================================================================== 9  R3 m2 / R2 #11
print("9 split-half reliability", flush=True)
rng = np.random.default_rng(1)
H1, H2 = {}, {}
for d in DS:
    nu = UCNT[d].size
    perm = np.argsort(rng.random((500, nu)), axis=1)
    A = np.zeros((500, nu), bool)
    A[np.arange(500)[:, None], perm[:, :nu // 2]] = True
    Af, Bf = A.astype(float), (~A).astype(float)
    H1[d] = (Af @ USUM["dice"][d]) / (Af @ UCNT[d])[:, None]
    H2[d] = (Bf @ USUM["dice"][d]) / (Bf @ UCNT[d])[:, None]
REL = {}
for s in ("U", "F", "K", "M19"):
    c = set_cols(s)
    vals = {}
    for d in REFS:
        r = (zr(H1[d][:, c]) * zr(H2[d][:, c])).sum(-1)
        vals[d] = float(np.median(2 * r / (1 + r)))
        chk(f"stats reliability/dice|{s}/{d}/spearman_brown_median", vals[d],
            SJ["reliability"][f"dice|{s}"][d]["spearman_brown_median"])
    REL[s] = {"per_dataset": vals,
              "range_11_clinical": [min(vals.values()), max(vals.values())],
              "range_4_wound": [min(vals[d] for d in WOUND), max(vals[d] for d in WOUND)]}
REL["U_F_K_pooled_range_11_clinical"] = [min(REL[s]["range_11_clinical"][0] for s in "UFK"),
                                        max(REL[s]["range_11_clinical"][1] for s in "UFK")]
R["R3_m2_split_half_reliability"] = {
    "source": "stats_v7.json reliability/dice|<set>/<dataset>/spearman_brown_median (recomputed here and verified)",
    "definition": "500 random half splits of units (default_rng(1), stats_v7 stream); Spearman between the two halves' method means; Spearman-Brown 2r/(1+r); median over splits",
    **REL}

# =================================================================== 10  R3 m1
print("10 R3 m1", flush=True)
STRAT = {}
for s in ("U+M19", "U+F"):
    gs = SETS[s]
    for m in ("dice", "nsd"):
        T = MEANS[m][:NR]
        Wp = np.concatenate([(rankdata(T[:, GCOLS[g]], axis=1) - 0.5) / GCOLS[g].size for g in gs], 1)
        Zp = Wp - Wp.mean(1, keepdims=True)
        Zp /= np.sqrt((Zp ** 2).sum(1, keepdims=True))
        Rp = Zp @ Zp.T
        within = {g: corrmat(T[:, GCOLS[g]]) for g in gs}
        w = {g: GCOLS[g].size - 1 / GCOLS[g].size for g in gs}
        Rw = sum(w[g] * within[g] for g in gs) / sum(w.values())
        Rn = sum(GCOLS[g].size * within[g] for g in gs) / sum(GCOLS[g].size for g in gs)
        st = SJ["stratified"][f"{m}|{s}"]
        chk(f"stats stratified/{m}|{s}/pearson_on_percentiles/wound pairs", Rp[PI, PJ_][WOUNDP].mean(),
            st["pearson_on_percentiles"]["wound pairs"])
        chk(f"stats stratified/{m}|{s}/size_weighted_average (n_g weights)/wound pairs", Rn[PI, PJ_][WOUNDP].mean(),
            st["size_weighted_average"]["wound pairs"])
        dpair = Rp[PI, PJ_] - Rw[PI, PJ_]
        kmax = int(np.argmax(np.abs(dpair)))
        STRAT[f"{m}|{s}"] = {
            "max_abs_single_pair_diff_pearsonpct_vs_ng_minus_1_over_ng": fl(np.abs(dpair).max()),
            "pair_of_max": f"{REFS[PI[kmax]]}~{REFS[PJ_[kmax]]}",
            "wound_pair_mean_diff": fl(Rp[PI, PJ_][WOUNDP].mean() - Rw[PI, PJ_][WOUNDP].mean()),
            "max_abs_wound_single_pair_diff": fl(np.abs(dpair[WOUNDP]).max()),
            "mean_over_55_pairs_diff": fl(dpair.mean()),
            "pearson_on_percentiles_wound_pairs": fl(Rp[PI, PJ_][WOUNDP].mean()),
            "weighted_avg_ng_minus_1_over_ng_wound_pairs": fl(Rw[PI, PJ_][WOUNDP].mean()),
            "weighted_avg_ng_wound_pairs_stats_v7": fl(Rn[PI, PJ_][WOUNDP].mean()),
            "weights_ng_minus_1_over_ng": {g: fl(w[g]) for g in gs},
            "max_abs_pair_diff_ng_vs_ng_minus_1_over_ng": fl(np.abs(Rw[PI, PJ_] - Rn[PI, PJ_]).max())}
R["R3_m1_stratification_identity"] = STRAT

# =================================================================== 11  R2 M3 scale
print("11 R2 M3 scale", flush=True)
mt = OVL["azh_test"]["matches"]
assert len(mt) == OVL["azh_test"]["n_match"] == 61
kept_azh = set(DATA["dice"]["azh"]["ids"])
sc_all = np.array([v["scale"] for v in mt.values()])
sc_kept = np.array([v["scale"] for k, v in mt.items() if k in kept_azh])
sc_ctrl = np.array([v["scale"] for v in OVL["validation_control"]["matches"].values() if v["scale"] is not None])


def kept_sizes(s):
    drop = {e[0] for e in V.EXCL.get(s, [])}
    return sorted({tuple(BSC[s]["size"][i]) for i in range(len(BSC[s]["ids"])) if i not in drop})


sz = {s: kept_sizes(s) for s in ("azh", "fuseg")}
assert sz["azh"] == [(224, 224)] and sz["fuseg"] == [(512, 512)], sz
AZ, FU, INP = 224, 512, 256
scale = float(np.median(sc_all))
mag_pt = (INP / AZ) / (scale * INP / FU)
mag_nn = 1 / scale
dwd = {}
for s in ("azh", "fuseg"):
    D = DATA["dice"][s]
    nem = ~D["empty"]
    Xs = D["X"][:, V.cols(allm)]
    det = nem[:, None] & (Xs >= 0.1)
    dwd[s] = {m: float(Xs[det[:, j], j].mean()) for j, m in enumerate(allm)}
arch_dwd = {a: float(np.mean([dwd["azh"][m] for m in ARCH[a]])) for a in ORIG + NEW}
v6m = [n for n in V.GROUPS["M"] if n != NN]
tab6 = PJ["v6"]["miss"]["0.1"]["table"]["azh"]
for m in v6m + [NN]:
    chk_r4(f"practical v6/miss/0.1/table/azh/{m}/dice_when_detected", dwd["azh"][m], tab6[m]["dice_when_detected"])
lesion = {}
for s in ("azh", "fuseg"):
    D = DATA["dice"][s]
    f = D["frac"][~D["empty"]]
    lesion[s] = float(np.median(f) * INP * INP)
    chk_r4(f"practical v6/azh_explanation/{s}/pretrained_input_lesion_px_median", lesion[s],
           PJ["v6"]["azh_explanation"][s]["pretrained_input_lesion_px_median"])
R["R2_M3_scale"] = {
    "sift_scale_fuseg_px_per_azh_px": {
        "median_all_61_matched_test_images": scale, "n_all": int(sc_all.size),
        "quartiles_all": [fl(np.percentile(sc_all, 25)), fl(np.percentile(sc_all, 75))],
        "min_all": fl(sc_all.min()), "max_all": fl(sc_all.max()),
        "median_matched_and_kept": fl(np.median(sc_kept)), "n_matched_and_kept": int(sc_kept.size),
        "median_control_crops": fl(np.median(sc_ctrl)), "n_control_crops": int(sc_ctrl.size),
        "source": "results/v7/azh_overlap_v7.json azh_test/matches/*/scale (similarity transform AZH -> FUSeg frame, scale = FUSeg pixels per AZH pixel)"},
    "image_sizes_px": {"azh": AZ, "fuseg": FU, "pretrained_input_long_side": INP,
                       "source": "results/v7/black_share_v7.json sets/<set>/size, kept images"},
    "magnification_vs_fuseg_training_input": {
        "pretrained_letterbox": fl(mag_pt), "nnunet_native": fl(mag_nn),
        "formula_pretrained": "(256/224) / (scale x 256/512) = 512 / (224 x scale)",
        "formula_nnunet": "1 / scale"},
    "median_lesion_px_in_pretrained_input": {"azh": lesion["azh"], "fuseg": lesion["fuseg"],
                                             "definition": "median over non-empty references of lesion fraction x 256 x 256"},
    "dice_when_detected_azh": {
        "P18_architecture_mean": fl(np.mean([arch_dwd[a] for a in ORIG + NEW])),
        "P18_n_architectures": len(ORIG + NEW), "P18_n_models": sum(len(ARCH[a]) for a in ORIG + NEW),
        "P10_architecture_mean": fl(np.mean([arch_dwd[a] for a in ORIG])),
        "N8_architecture_mean": fl(np.mean([arch_dwd[a] for a in NEW])),
        "nnunet": dwd["azh"][NN],
        "v6_style_10_seed0_models_mean": fl(np.mean([dwd["azh"][m] for m in v6m])),
        "per_architecture": arch_dwd,
        "definition": "mean Dice over non-empty AZH references with Dice >= 0.1, per model; architecture = mean over its seeds; group = mean over architectures"},
    "dice_when_detected_fuseg": {"P18_architecture_mean": fl(np.mean([np.mean([dwd["fuseg"][m] for m in ARCH[a]]) for a in ORIG + NEW])),
                                 "nnunet": dwd["fuseg"][NN]}}

# =================================================================== 12  R2 M5 / R3 M7 oracle grid
print("12 oracle grid", flush=True)
od = RJ["threshold"]["oracle_detail"]["M19"]
oi = RJ["threshold"]["oracle_individual"]
ORC = {}
SMEM = sorted({x for v in RJ["definitions"]["seed_members"].values() for x in v})
assert len(SMEM) == 52 and NN in SMEM and len(oi["medetec"]) > len(SMEM)
for d in WOUND:
    thr = [oi[d][x]["thr"] for x in SMEM]
    share = float(np.mean(np.isclose(thr, 0.05)))
    chk(f"robust threshold/oracle_detail/M19/{d}/share_models_oracle_at_grid_min_0.05 (from oracle_individual)",
        share, od[d]["share_models_oracle_at_grid_min_0.05"])
    ORC[d] = {"share_at_grid_min_0.05": od[d]["share_models_oracle_at_grid_min_0.05"],
              "n_at_grid_min": int(np.sum(np.isclose(thr, 0.05))), "n_models": len(thr),
              "median_oracle_thr": od[d]["median_oracle_thr"]}
ORC["models"] = ("the 52 individual models behind M19, that is 51 pretrained seed models and nnU-Net (robust_v7 definitions/seed_members); "
                 "the Kvasir-trained models also listed in threshold/oracle_individual are not counted")
ORC["non_wound_share_range"] = [min(od[d]["share_models_oracle_at_grid_min_0.05"] for d in REFS[4:]),
                                max(od[d]["share_models_oracle_at_grid_min_0.05"] for d in REFS[4:])]
R["R2_M5_oracle_grid_censoring"] = ORC

# =================================================================== 13  R2 #13
R["R2_13_nsd_vs_dice_wound_pairs"] = {s: {"dice": PT[f"dice|{s}"]["wound pairs"], "nsd": PT[f"nsd|{s}"]["wound pairs"]}
                                      for s in ("U", "F", "K")}

# =================================================================== 14  R2 M6
print("14 floor", flush=True)
cU = set_cols("U")
nf = [r for r in WOUND if r != "fuseg"]
Ru = PT["dice|U"]["_R"]
u3 = np.mean([Ru[REFS.index(a), REFS.index(b)] for a, b in itertools.combinations(nf, 2)])
FLO = {"U_wound_pairs_without_fuseg_3pairs": fl(u3), "U_wound_pairs_6": PT["dice|U"]["wound pairs"],
       "pairs": [f"{a}~{b}" for a, b in itertools.combinations(nf, 2)], "mean_dice": {}, "below_0.1": {}}
for g in ("U", "F", "K", "M19"):
    FLO["mean_dice"][g] = {}
    for k, d in enumerate(DS):
        v = MEANS["dice"][k, GCOLS[g]].mean()
        chk(f"stats floor/mean_dice/{g}/{d}", v, SJ["floor"]["mean_dice"][g][d])
        FLO["mean_dice"][g][d] = fl(v)
    FLO["below_0.1"][g] = [d for d in DS if FLO["mean_dice"][g][d] < 0.1]
    assert FLO["below_0.1"][g] == SJ["floor"]["flag_lt_0.1"][g], g
    FLO[f"n_nonwound_below_0.1|{g}"] = sum(d in REFS[4:] for d in FLO["below_0.1"][g])
R["R2_M6_floor"] = FLO

# =================================================================== 15  R2 #7
seg = "segformer_mitb2"
segd = DATA["dice"]["azh"]["X"][:, V.cols([seg + "@seeds"])][:, 0].mean()
nnd = DATA["dice"]["azh"]["X"][:, V.cols([NN])][:, 0].mean()
chk("practical spread/M19/azh/means/segformer_mitb2@seeds", segd, PJ["spread"]["M19"]["azh"]["means"][seg + "@seeds"])
chk("practical spread/M19/azh/means/nnunet_2d", nnd, PJ["spread"]["M19"]["azh"]["means"][NN])
pa = PJ["miss"]["0.1"]["azh"]
chk("practical miss/0.1/azh/per_architecture/segformer_mitb2/mean", arch_pt[seg], pa["per_architecture"][seg]["mean"])
chk("practical miss/0.1/azh/per_model/nnunet_2d/miss_rate", mr[colix[NN]], pa["per_model"][NN]["miss_rate"])
m19means = {c: MEANS["dice"][DS.index("azh"), V.cols([c])][0] for c in V.GROUPS["M19"]}
R["R2_7_segformer_vs_nnunet_azh"] = {
    "segformer_mitb2_seedavg_mean_dice": fl(segd), "nnunet_mean_dice": fl(nnd),
    "segformer_mitb2_miss_rate_seed_mean": fl(arch_pt[seg]), "segformer_mitb2_miss_rate_per_seed": pa["per_architecture"][seg]["per_seed"],
    "nnunet_miss_rate": fl(mr[colix[NN]]), "nnunet_miss_ci95": pa["per_model"][NN]["ci"],
    "segformer_mitb2_miss_ci95_of_seed_mean": pa["per_architecture"][seg]["ci_of_seed_mean"],
    "azh_top_mean_dice_in_M19": max(m19means, key=m19means.get)}

# =================================================================== 16  R3 M7 oracle delta
dv = RJ["threshold"]["delta_vs_default"]["M19|oracle|wound pairs"]
R["R3_M7_oracle_minus_main"] = {"M19_wound_pairs_change": {"est": dv["est"], "ci95": dv["ci"]},
                                "M19_wound_pairs_oracle": RJ["threshold"]["oracle"]["M19"]["wound pairs"],
                                "M19_wound_pairs_default": RJ["threshold"]["default"]["M19"]["wound pairs"],
                                "source": "robust_v7.json threshold/delta_vs_default/M19|oracle|wound pairs (read only)"}

# ------------------------------------------------------------------ write
for k in list(PT):
    PT[k].pop("_R")
R["verification"] = {"n_checks": len(VER), "all_passed": True, "checks": VER}
R["runtime_s"] = round(time.time() - T0, 1)
R["definitions"] = __doc__
os.makedirs(OUT, exist_ok=True)
json.dump(R, open(os.path.join(OUT, "review_v7.json"), "w"), indent=1)
print(f"saved {C.V7OUT_REL}/review_v7.json, {len(VER)} checks passed, {R['runtime_s']} s", flush=True)

# ================================================================ markdown


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def f4(x):
    return f"{x:.4f}".replace("-", "−")


def fe(x):
    return f"{x:.1e}".replace("-", "\u2212")


def pc(x):
    return f"{100 * x:.1f}%"


def pp(x):
    return f"{100 * x:.1f}".replace("-", "−")


def iv(c, f=f3):
    return f"{f(c[0])} to {f(c[1])}"


L = []
w = L.append
w("# REVIEW V7: numbers requested by reviews R2 and R3\n")
w(f"Generated by `scripts/v7/review_v7.py`; every value is in `{C.V7OUT_REL}/review_v7.json`. "
  f"Exclusions: `{C.EXCL_NAME}` (AZH test {R['n_images']['azh']} images in {R['n_units']['azh']} photographs; "
  f"control {R['n_images']['azh_train_slice']} images in {R['n_units']['azh_train_slice']} photographs). "
  "Intervals are 2.5 and 97.5 percentiles of the bootstrap draws. Minus signs are U+2212.\n")
w("## Common definitions\n")
w("- **Agreement**: Spearman correlation (Pearson on average ranks) of per-method mean scores between two image sets, over a method set.")
w("- **Wound pairs**: the 6 pairs of Medetec, CO2Wounds-V2, FUSeg and AZH. **Different-source wound pairs**: the 5 pairs without FUSeg~AZH (one clinic).")
w("- **Seed sets**: P9 = the 9 original architectures with 3 training seeds, N7 = the 7 new architectures with 3 seeds, C16 = P9 + N7.")
w("- **Same-dataset cross-seed agreement**: for each wound dataset, the mean over the 3 seed pairs of the agreement between seed a and seed b on that dataset; then the mean over the 4 wound datasets. "
  "**Cross-dataset cross-seed agreement**: the mean over 6 ordered seed pairs (a different from b) and 6 wound pairs of the agreement between seed a on one dataset and seed b on the other (36 values). **Gap** = same-dataset minus cross-dataset.")
w("- **Units**: photograph for Medetec, AZH and the control; image otherwise.")
w("- **Selection loss** (choose on a, deploy on b): best mean Dice on b minus the mean Dice on b of the candidate with the highest mean Dice on a. "
  "**Chooser loss**: mean over the 12 ordered pairs of different wound datasets. **Random loss**: for each wound dataset the mean over candidates of (best minus candidate), then the mean over the 4 datasets.")
w("- **Miss**: Dice < 0.1 on a non-empty reference. Miss rate = misses / non-empty images, per individual trained model; architecture value = mean over its available seeds; group value = mean over architectures.\n")

w("## 1. R3 M3: is the seed gap specific to the original architectures?\n")
p = R["R3_M3_seed_gap_subgroups"]
w("| Set | Same-dataset cross-seed | Cross-dataset cross-seed | Gap |")
w("|---|---|---|---|")
for k in ("P9", "N7", "C16"):
    w(f"| {k} | {f3(p['point'][k]['same_dataset_cross_seed'])} | {f3(p['point'][k]['cross_dataset_cross_seed'])} | {f3(p['point'][k]['gap'])} |")
gd = p["gap_P9_minus_N7"]
w(f"\nGap P9 minus gap N7 (Dice, wound datasets): {f3(gd['est'])} ({iv(gd['ci95'])}); share of draws at or below 0: {pc(gd['share_draws_le0'])}. "
  f"Architectures of P9 and N7 resampled independently with replacement, B {gd['B']}, default_rng({gd['rng']}). "
  f"In the same draws the P9 gap interval is {iv(p['gap_P9_draws_ci95'])} and the N7 gap interval {iv(p['gap_N7_draws_ci95'])}.\n")
ss = p["same_session_P9_11_refs"]
w(f"Same-session check (P9, Dice, 11 clinical datasets; seeds 1 and 2 share a training session, seed 0 does not): mean r12 {f3(ss['mean_r12'])}, "
  f"mean r01 {f3(ss['mean_r01'])}, mean r02 {f3(ss['mean_r02'])} (mean of r01 and r02: {f3(ss['mean_r01_r02'])}). "
  f"r12 exceeds both r01 and r02 on {ss['n_datasets_r12_gt_both_r01_r02']} of {ss['n_datasets']} datasets.\n")

w("## 2. R3 M4: shared test images in the seed gap\n")
q = R["R3_M4_shared_images"]
w(f"Random half splits of units ({q['half_splits_setup']['n_splits_per_dataset']} per wound dataset, default_rng({q['half_splits_setup']['rng']}), Dice). "
  "For each split and seed pair (a, b): **shared** = agreement between seed a and seed b scored on the same half (both halves, averaged); "
  "**independent** = seed a on one half against seed b on the other half (both directions, averaged). Averaged over the 3 seed pairs, the splits and the 4 wound datasets.\n")
w("| Set | Shared halves | Independent halves | Shared minus independent |")
w("|---|---|---|---|")
for k in ("C16", "P9", "N7"):
    h = q["half_splits"][k]
    w(f"| {k} | {f3(h['shared_halves'])} | {f3(h['independent_halves'])} | {f3(h['shared_minus_independent'])} |")
w(f"\nUnits-only bootstrap of the gap (architectures fixed; units resampled within every dataset with the stats_v7 section-3 draws, B {q['units_only_setup']['B']}, default_rng(6060); "
  "its intervals reproduce stats_v7 exactly):\n")
w("| Set | Estimate | Mean of draws | Mean minus estimate | Interval |")
w("|---|---|---|---|---|")
for k in ("C16", "P9", "N7"):
    u = q["units_only_bootstrap_gap"][k]
    w(f"| {k} | {f3(u['est'])} | {f3(u['mean_of_draws'])} | {f3(u['mean_minus_est'])} | {iv(u['ci95'])} |")
w("")

w("## 3. R3 M5: selection loss with uncertainty\n")
sl = R["R3_M5_selection"]["M19"]
ub, cb = sl["unit_bootstrap"], sl["candidate_bootstrap"]
w("M19 seed-averaged candidates (19). Unit bootstrap: units resampled within each wound dataset, independently per dataset, with the practical_v7 draws (default_rng([20261010, k]), B 2000); chooser and random losses are paired in every draw.\n")
w("| Quantity | Estimate | Unit bootstrap | Candidate bootstrap |")
w("|---|---|---|---|")
for key, lab in (("chooser_loss", "Chooser loss"), ("random_loss", "Random loss"), ("chooser_minus_random", "Chooser minus random")):
    w(f"| {lab} | {f3(ub[key]['est'])} | {iv(ub[key]['ci95'])} | {iv(cb[key]['ci95'])} |")
w(f"\nCandidate bootstrap: the 19 candidates resampled with replacement (default_rng(31003), B 2000), point-estimate means. "
  f"Share of draws with chooser minus random at or above 0: units {pc(ub['chooser_minus_random']['share_draws_ge0'])}, candidates {pc(cb['chooser_minus_random']['share_draws_ge0'])}.\n")
hs = sl["half_split_same_dataset"]
w(f"Choose on one random half of a wound dataset, deploy on the other half ({hs['n_splits']} unit splits per dataset, default_rng({hs['rng']}); "
  "the loss is the mean of half A to half B and half B to half A, then the mean over splits):\n")
w("| Dataset | Loss | Share of splits where both halves pick the same candidate |")
w("|---|---|---|")
for d in WOUND:
    w(f"| {d} | {f3(hs['per_dataset'][d]['loss_mean'])} | {pc(hs['per_dataset'][d]['share_same_pick_both_halves'])} |")
w(f"\nRange over the 4 wound datasets: {iv(hs['range'])}.\n")
w("Single-seed candidate sets (every architecture that has that seed, as the single model, plus nnU-Net); values verified against practical_v7.json:\n")
w("| Candidate set | n | Chooser loss | Random loss | Random minus chooser |")
w("|---|---|---|---|---|")
for k in ("M19", "M19_s0", "M19_s1", "M19_s2"):
    x = R["R3_M5_selection"][k]
    w(f"| {k} | {x['n_candidates']} | {f3(x['chooser_loss'])} | {f3(x['random_loss'])} | {f3(-x['chooser_minus_random'])} |")
ra = R["R3_M5_selection"]["single_seed_advantage_range"]
w(f"\nAdvantage of the chooser over a random pick with single trained models: {iv(ra)}.\n")
w(f"Choosing on FUSeg (M19): mean loss on the other three wound datasets {f3(sl['choose_on_fuseg_mean_loss_other3'])} "
  f"(" + ", ".join(f"{b} {f3(v)}" for b, v in sl["choose_on_fuseg_loss"].items()) + "). Row means for every choose-on dataset: "
  + ", ".join(f"{a} {f3(v)}" for a, v in sl["row_mean_loss_other3"].items()) + ".\n")

w("## 4. R3 m7: per-reference benchmark agreement, paired differences\n")
w("Generator B amplitude 40; agreement with a reference = mean over synthetic seeds 2024, 7, 101 (bench_v7 definition). Paired method bootstrap with bench_v7.method_takes(pool, 0), B 2000 (the same method draws as bench_v7, whose per-reference intervals are reproduced exactly).\n")
w("| Set | Comparison | First | Second | Difference |")
w("|---|---|---|---|---|")
for pool in ("U", "K", "M19"):
    x = R["R3_m7_per_reference_benchmark_differences"][pool]
    w(f"| {pool} | {x['minuend']} minus {x['subtrahend']} | {f3(x['bench_minuend']['est'])} | {f3(x['bench_subtrahend']['est'])} | "
      f"{f3(x['difference']['est'])} ({iv(x['difference']['ci95'])}) |")
w("")

w("## 5. R3 m9: AZH miss rate, original ten minus new eight architectures\n")
x = R["R3_m9_azh_miss_original_minus_new"]
w(f"Original ten {pc(x['P10_archmean'])}, new eight {pc(x['N8_archmean'])}. Difference {pp(x['difference_units_only']['est'])} percentage points: "
  f"{iv(x['difference_units_only']['ci95'], pp)} with the architectures fixed (paired unit bootstrap, practical_v7 AZH draws, B 2000); "
  f"{iv(x['difference_units_and_architectures']['ci95'], pp)} when the architectures are also resampled (default_rng(31005), one architecture draw per unit draw, B 2000).\n")

w("## 6. R3 m17: reliability of the synthetic benchmark\n")
x = R["R3_m17_synthetic_reliability"]
w(f"Over M19 ({x['n_methods']} methods), generator B amplitude 40: mean agreement between the three synthetic seeds {f3(x['between_synthetic_seeds_mean_spearman'])} "
  f"(" + ", ".join(f"{k} {f3(v)}" for k, v in x["between_synthetic_seeds_pairs"].items()) + "). "
  f"Pooling the 3 x 45 = {x['n_images_pooled']} images into one set gives a mean agreement with the 4 wound datasets of {f3(x['pooled_135_images_vs_wound_mean'])}, "
  f"against {f3(x['mean_over_seeds_vs_wound_mean'])} for the mean over seeds (bench_v7 rel/M19/40/mean|bench).\n")

w("## 7. R3 M1: Holm adjustment of the exact relabelling tests\n")
x = R["R3_M1_holm"]
w("Exact relabelling test of the modality difference (55 pairs, Dice) from stats_v7.json; Holm step-down over the seven method sets.\n")
w("| Set | Raw p | Holm-adjusted p |")
w("|---|---|---|")
for s in SEVEN:
    w(f"| {s} | {x['raw_p'][s]:.5f} | {x['holm_adjusted_p'][s]:.5f} |")
w(f"\nLargest adjusted p: {x['largest_adjusted_p']:.5f} ({x['largest_adjusted_set']}).\n")

w("## 8. R3 m16: wound-pair agreement over the five different-source pairs (Dice)\n")
w("| Set | Six wound pairs | Five different-source pairs | Five minus six |")
w("|---|---|---|---|")
for s in SEVEN:
    y = R["R3_m16_different_source_wound_pairs"][s]
    w(f"| {s} | {f3(y['six_pairs'])} | {f3(y['five_different_source_pairs'])} | {f3(y['five_minus_six'])} |")
w("")

w("## 9. R3 m2 / R2 #11: within-group split-half reliability (Dice)\n")
x = R["R3_m2_split_half_reliability"]
w(f"Definition: {x['definition']}. Values from stats_v7.json, recomputed here and verified.\n")
w("| Group | Range over 11 clinical datasets | Range over 4 wound datasets |")
w("|---|---|---|")
for s in ("U", "F", "K", "M19"):
    w(f"| {s} | {iv(x[s]['range_11_clinical'])} | {iv(x[s]['range_4_wound'])} |")
w(f"\nU, F and K together over the 11 clinical datasets: {iv(x['U_F_K_pooled_range_11_clinical'])}.\n")

w("## 10. R3 m1: stratification identity\n")
w("Pearson correlation on within-group percentiles (r − 0.5)/n_g compared with the average of within-group Spearman correlations weighted by n_g − 1/n_g (the Methods formula). "
  "stats_v7 calls the n_g-weighted average 'size-weighted'; both are shown.\n")
w("| Set | Metric | Max single-pair difference (pair) | Wound-pair mean difference | Wound pairs, Pearson on percentiles | Wound pairs, (n_g − 1/n_g)-weighted | Wound pairs, n_g-weighted |")
w("|---|---|---|---|---|---|---|")
for s in ("U+M19", "U+F"):
    for m in ("dice", "nsd"):
        y = R["R3_m1_stratification_identity"][f"{m}|{s}"]
        w(f"| {s} | {m} | {f4(y['max_abs_single_pair_diff_pearsonpct_vs_ng_minus_1_over_ng'])} ({y['pair_of_max']}) | {fe(y['wound_pair_mean_diff'])} | "
          f"{f3(y['pearson_on_percentiles_wound_pairs'])} | {f3(y['weighted_avg_ng_minus_1_over_ng_wound_pairs'])} | {f3(y['weighted_avg_ng_wound_pairs_stats_v7'])} |")
w("")

w("## 11. R2 M3: input scale on AZH\n")
x = R["R2_M3_scale"]
s_ = x["sift_scale_fuseg_px_per_azh_px"]
w(f"SIFT similarity transforms between AZH test images and the FUSeg photographs they were cut from (azh_overlap_v7.json): all {s_['n_all']} matched test images are used, "
  f"including the {s_['n_all'] - s_['n_matched_and_kept']} excluded from the analysis. Median scale {f3(s_['median_all_61_matched_test_images'])} FUSeg pixels per AZH pixel "
  f"(quartiles {iv(s_['quartiles_all'])}, range {iv([s_['min_all'], s_['max_all']])}); over the {s_['n_matched_and_kept']} matched images that are kept, {f3(s_['median_matched_and_kept'])}; "
  f"over the {s_['n_control_crops']} control crops, {f3(s_['median_control_crops'])}.\n")
mg = x["magnification_vs_fuseg_training_input"]
w(f"Image sizes: AZH {x['image_sizes_px']['azh']} px, FUSeg {x['image_sizes_px']['fuseg']} px (square, kept images). The pretrained models letterbox to {x['image_sizes_px']['pretrained_input_long_side']} px, "
  f"so they see AZH tissue at {mg['formula_pretrained']} = {f2(mg['pretrained_letterbox'])} times their FUSeg training magnification; nnU-Net works on native pixels and sees it at 1/scale = {f2(mg['nnunet_native'])} times.\n")
lz = x["median_lesion_px_in_pretrained_input"]
w(f"Median lesion size in the pretrained models' input frame ({lz['definition']}): AZH {lz['azh']:,.0f} px, FUSeg {lz['fuseg']:,.0f} px.\n")
dw = x["dice_when_detected_azh"]
w(f"Dice when detected on AZH ({dw['definition']}): 18 pretrained architectures ({dw['P18_n_models']} models) {f3(dw['P18_architecture_mean'])} "
  f"(original ten {f3(dw['P10_architecture_mean'])}, new eight {f3(dw['N8_architecture_mean'])}); nnU-Net {f3(dw['nnunet'])}. "
  f"V6 definition (mean over the 10 seed-0 models): {f3(dw['v6_style_10_seed0_models_mean'])}. On FUSeg: pretrained {f3(x['dice_when_detected_fuseg']['P18_architecture_mean'])}, nnU-Net {f3(x['dice_when_detected_fuseg']['nnunet'])}.\n")

w("## 12. R2 M5 / R3 M7: oracle thresholds at the grid minimum\n")
x = R["R2_M5_oracle_grid_censoring"]
w("Share of models whose per-dataset oracle threshold is the grid minimum 0.05. Models: " + x["models"] +
  ". Source: robust_v7.json threshold/oracle_detail/M19, verified from threshold/oracle_individual.\n")
w("| Dataset | Share at 0.05 | Models | Median oracle threshold |")
w("|---|---|---|---|")
for d in WOUND:
    w(f"| {d} | {pc(x[d]['share_at_grid_min_0.05'])} | {x[d]['n_at_grid_min']} of {x[d]['n_models']} | {x[d]['median_oracle_thr']:.2f} |")
w(f"\nOn the 7 non-wound datasets the share is {pc(x['non_wound_share_range'][0])} to {pc(x['non_wound_share_range'][1])}.\n")

w("## 13. R2 #13: wound-pair agreement, NSD against Dice\n")
w("| Group | Dice | NSD |")
w("|---|---|---|")
for s in ("U", "F", "K"):
    y = R["R2_13_nsd_vs_dice_wound_pairs"][s]
    w(f"| {s} | {f3(y['dice'])} | {f3(y['nsd'])} |")
w("")

w("## 14. R2 M6: floor\n")
x = R["R2_M6_floor"]
w(f"Untrained (U) wound-pair agreement without FUSeg ({', '.join(x['pairs'])}): {f3(x['U_wound_pairs_without_fuseg_3pairs'])}, against {f3(x['U_wound_pairs_6'])} over the 6 pairs (Dice).\n")
w("Group mean Dice per dataset (stats_v7.json floor/mean_dice, verified):\n")
w("| Dataset | U | F | K | M19 |")
w("|---|---|---|---|---|")
for d in DS:
    w(f"| {d} | " + " | ".join(f3(x['mean_dice'][g][d]) for g in ("U", "F", "K", "M19")) + " |")
w("")
for g in ("U", "F", "K", "M19"):
    w(f"- {g} below 0.1 on: {', '.join(x['below_0.1'][g]) or 'none'} ({x[f'n_nonwound_below_0.1|{g}']} of 7 non-wound datasets).")
w("")

w("## 15. R2 #7: SegFormer (MiT-B2) against nnU-Net on AZH\n")
x = R["R2_7_segformer_vs_nnunet_azh"]
w(f"Mean Dice: SegFormer MiT-B2 (seed-averaged) {f3(x['segformer_mitb2_seedavg_mean_dice'])}, nnU-Net {f3(x['nnunet_mean_dice'])}; highest in M19 on AZH: {x['azh_top_mean_dice_in_M19']}. "
  f"Miss rate: SegFormer MiT-B2 {pc(x['segformer_mitb2_miss_rate_seed_mean'])} (mean over its 3 seeds; interval {iv(x['segformer_mitb2_miss_ci95_of_seed_mean'], pc)}), "
  f"nnU-Net {pc(x['nnunet_miss_rate'])} (interval {iv(x['nnunet_miss_ci95'], pc)}).\n")

w("## 16. R3 M7: oracle threshold against the main protocol\n")
x = R["R3_M7_oracle_minus_main"]
w(f"M19 wound pairs (Dice): main protocol {f3(x['M19_wound_pairs_default']['est'])}, oracle {f3(x['M19_wound_pairs_oracle']['est'])} "
  f"({iv(x['M19_wound_pairs_oracle']['ci'])}); paired change {f3(x['M19_wound_pairs_change']['est'])} ({iv(x['M19_wound_pairs_change']['ci95'])}). Source: {x['source']}.\n")

w("## Verification\n")
w(f"{len(VER)} checks passed (point estimates recomputed here against the existing result files, abs diff <= 1e-12 unless the file stored 4-decimal values). By file:\n")
cnt = {}
for v in VER:
    key = v["what"].split(" ")[0]
    cnt[key] = cnt.get(key, 0) + 1
for k2, n in cnt.items():
    w(f"- {k2}: {n}")
w(f"\nLargest abs difference among the 1e-12 checks: {max(v['abs_diff'] for v in VER if v['tol'] == TOL):.1e}.\n")
w(f"Runtime {R['runtime_s']} s.")
txt = "\n".join(L) + "\n"
assert chr(0x2014) not in txt
open(os.path.join(OUT, "REVIEW_V7.md"), "w").write(txt)
print(f"saved {C.V7OUT_REL}/REVIEW_V7.md")

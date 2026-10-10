#!/usr/bin/env python3
"""V6 reanalysis of the agreement statistics, written to results/v6/stats_v6.json
and results/v6/STATS_V6.md.

All data come from scripts/perimage_v6.py (duplicates removed, resampling unit =
photograph for Medetec and AZH, image otherwise). Definitions follow
scripts/stats_v5.py:
  agreement      Spearman correlation (Pearson on average ranks) between the
                 per-method mean scores of two datasets, over a method set
  levels         same dataset (azh vs azh_train_slice), same source (3 pairs),
                 different source same modality (13 pairs), across modality
                 (39 pairs); all over the 55 pairs of the 11 REFS
  modality diff  mean same-modality agreement (16 pairs) minus mean
                 across-modality agreement (39 pairs)
  categories     stats_v5 f_comp: in x in, in x out, out same modality,
                 out across modality, for a domain (wound or colonoscopy)
Method sets U, F, K, M and mixtures U+F, U+K, U+M; ALL57 = U+F+K+M.

Sections
  1 split-half reliability by unit (500 splits)
  2 disattenuated agreement on the same splits
  3 hierarchical bootstrap (methods within group x units within dataset)
  4 modality test without the 3 within-source pairs; leave-one-out checks
  5 stratified agreement vs size-weighted within-group agreement
  6 empty-mask-excluded Dice modality difference
  7 group mean Dice per dataset and share of near-zero images
Run from the rebuild folder:  python scripts/v6/stats_v6.py
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sympy.utilities.iterables import multiset_permutations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage_v6 as V                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "v6")
os.makedirs(OUT, exist_ok=True)
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

GROUP_SETS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
              "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"],
              "ALL57": ["U", "F", "K", "M"]}
V5_POOLS = ["U", "F", "K", "M", "U+F", "U+K", "U+M"]
GCOLS = {g: V.cols(V.GROUPS[g]) for g in ("U", "F", "K", "M", "S")}


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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(A[..., mask], axis=-1)


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


# ------------------------------------------------------------------- data
print("loading", flush=True)
DATA = {m: {} for m in METRICS}
for m in METRICS:
    for d in DS:
        DATA[m][d] = V.load(d, m)
for d in DS:
    assert DATA["dice"][d]["ids"] == DATA["nsd"][d]["ids"]
    assert list(DATA["dice"][d]["units"]) == list(DATA["nsd"][d]["units"])
UNITS = {}
for d in DS:
    u = DATA["dice"][d]["units"]
    uniq, inv = np.unique(u, return_inverse=True)
    UNITS[d] = inv
MEANS = {m: np.stack([DATA[m][d]["X"].mean(0) for d in DS]) for m in METRICS}   # 12 x 59
MEANS["dice_noempty"] = np.stack([DATA["dice"][d]["X"][~DATA["dice"][d]["empty"]].mean(0)
                                  for d in DS])
USUM = {m: {d: np.zeros((UNITS[d].max() + 1, MEANS["dice"].shape[1])) for d in DS} for m in METRICS}
UCNT = {d: np.bincount(UNITS[d]).astype(float) for d in DS}
for m in METRICS:
    for d in DS:
        np.add.at(USUM[m][d], UNITS[d], DATA[m][d]["X"])

V5 = json.load(open(os.path.join(ROOT, "results", "stats_v5.json")))
S = {"definitions": {
    "data": "scripts/perimage_v6.py load(); duplicates removed; units = photograph for Medetec and AZH (incl. control), image otherwise",
    "agreement": "Spearman (Pearson on average ranks) of per-method mean scores between two datasets over a method set",
    "pairs": "55 pairs of the 11 REFS; same modality 16 (of which 3 within source: " + ", ".join(WITHIN_SRC_NAMES) + "; 13 different source), across modality 39; control = azh vs azh_train_slice",
    "modality_diff": "mean same-modality pair agreement (16) minus mean across-modality agreement (39)",
    "modality_diff_excl_within_source": "mean of the 13 different-source same-modality pairs minus mean of the 39 across-modality pairs; relabeling test drops the 3 within-source pairs by identity (52 pairs)",
    "method_sets": {s: {"groups": g, "n": int(len(set_cols(s)))} for s, g in GROUP_SETS.items()},
    "n_splits": N_SPLIT, "seed_split": SEED_SPLIT, "B_hier": B_HIER, "seed_hier": SEED_HIER},
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

# by-image split for ALL57 Dice (to separate the unit effect from v5's image split)
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
            with np.errstate(invalid="ignore"):
                import warnings
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
    """Exactly the method draws of stats_v5.boot (same RNG stream)."""
    nm = sum((V.GROUPS[g] for g in GROUP_SETS[pool]), [])
    pos = {n: k for k, n in enumerate(nm)}
    idx = [np.array([pos[n] for n in V.GROUPS[g]]) for g in GROUP_SETS[pool]]
    colmap = V.cols(nm)
    r = np.random.default_rng(seed)
    return np.stack([colmap[np.concatenate([r.choice(i, len(i)) for i in idx])] for _ in range(b)])


rng = np.random.default_rng(SEED_HIER)
UMEANS = {m: np.zeros((B_HIER, len(DS), MEANS["dice"].shape[1])) for m in METRICS}
for k, d in enumerate(DS):
    nu = UCNT[d].size
    draw = rng.integers(0, nu, size=(B_HIER, nu))
    W = np.zeros((B_HIER, nu))
    np.add.at(W, (np.repeat(np.arange(B_HIER), nu), draw.ravel()), 1.0)
    for m in METRICS:
        UMEANS[m][:, k, :] = (W @ USUM[m][d]) / (W @ UCNT[d])[:, None]


def gather(Mtensor, take):
    """Mtensor (B, D, 59) or (D, 59); take (B, n) -> (B, D, n)."""
    if Mtensor.ndim == 2:
        return Mtensor[:, take].transpose(1, 0, 2)
    return np.take_along_axis(Mtensor, take[:, None, :].repeat(Mtensor.shape[1], 1), axis=2)


BOOT = {}
QK = ["modality_diff", "modality_diff_excl_within_source", "same dataset", "same source",
      "different source, same modality", "across modality", "wound pairs",
      "wound different-source pairs", "fuseg~azh", "colonoscopy pairs"] + \
     [f"comp|{d}|{k}" for d in COMP for k in COMP[d]]
for s in V5_POOLS:
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
    L = np.array(list(multiset_permutations(lab)))
    same = L[:, ia] == L[:, ib]
    stat = (same * r).sum(1) / same.sum(1) - ((~same) * r).sum(1) / (~same).sum(1)
    obs_same = np.array(lab)[ia] == np.array(lab)[ib]
    obs = r[obs_same].mean() - r[~obs_same].mean()
    # relabelings that give the same pair partition (e.g. swapping the two
    # singleton labels dermoscopy and ultrasound) always tie with the observed
    # statistic, so the smallest attainable p is n_equiv / n_relabelings
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
    for s in ("M", "U"):
        R = POINT[f"{m}|{s}"]["_R"]
        LODO[f"{m}|{s}"] = {}
        for k in range(NR):
            keys = [x for x in ALLK if x != k]
            LODO[f"{m}|{s}"][REFS[k]] = {"full": exact_test(R, keys),
                                         "excl_within_source": exact_test(R, keys, DROP)}
S["leave_one_dataset_out"] = LODO

LOMO = {}
for m in METRICS:
    mc = GCOLS["M"]
    LOMO[m] = {}
    for k, n in enumerate(V.GROUPS["M"]):
        c = np.delete(mc, k)
        P = summ(corrmat(MEANS[m][:, c]))
        LOMO[m][n] = {q: float(P[q]) for q in ("modality_diff", "modality_diff_excl_within_source", "wound pairs")}
S["leave_one_method_out_M"] = LOMO

# ================================================== 5  stratified agreement
print("stratified", flush=True)
STR = {}
for s in ("U+F", "U+K", "U+M"):
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
        out["v5_stratified_modality_diff"] = V5["stratified"][s][m]["modality_diff"]
        out["v5_stratified_wound_in_x_in"] = V5["stratified"][s][m]["categories_wound"]["in x in"]
        out["within_group"] = {g: {q: float(summ(within[g])[q]) for q in ("wound pairs", "modality_diff")} for g in gs}
        STR[f"{m}|{s}"] = out
S["stratified"] = STR

# ================================================== 6  empty-mask-excluded Dice
NOEMPTY = {}
for s in V5_POOLS:
    P = POINT[f"dice_noempty|{s}"]
    v = V5["hier"][f"dice_noempty|{s}"]["modality_diff"]
    NOEMPTY[s] = {"modality_diff": P["modality_diff"], "p": EX[f"dice_noempty|{s}"]["full_55"]["p"],
                  "v5_est": v["est"], "v5_p": V5["exact"][f"dice_noempty|{s}"]["modality"]["p"],
                  "dice_with_empty": POINT[f"dice|{s}"]["modality_diff"],
                  "excl_within_source": P["modality_diff_excl_within_source"]}
S["dice_noempty"] = NOEMPTY
S["n_empty"] = {d: int(DATA["dice"][d]["empty"].sum()) for d in DS}

# ================================================== 7  floor effect
FLOOR = {"mean_dice": {}, "share_dice_lt_0.05_median_over_methods": {}, "flag_lt_0.1": {}}
for g in ("U", "F", "K", "M", "S"):
    FLOOR["mean_dice"][g] = {d: float(MEANS["dice"][k, GCOLS[g]].mean()) for k, d in enumerate(DS)}
    FLOOR["flag_lt_0.1"][g] = [d for d in DS if FLOOR["mean_dice"][g][d] < 0.1]
    FLOOR["share_dice_lt_0.05_median_over_methods"][g] = {
        d: float(np.median((DATA["dice"][d]["X"][:, GCOLS[g]] < 0.05).mean(0))) for d in DS}
mchk = [abs(V5["mean_dice"][g][d] - FLOOR["mean_dice"][g][d])
        for g in ("U", "F", "K", "M", "S") for d in (DS if g != "S" else REFS)]
FLOOR["check_vs_v5_mean_dice_max_abs_diff"] = float(max(mchk))
S["floor"] = FLOOR

for k in POINT:
    POINT[k].pop("_R")
S["point"] = POINT
S["runtime_s"] = round(time.time() - T0, 1)
json.dump(S, open(os.path.join(OUT, "stats_v6.json"), "w"), indent=1)
print("saved results/v6/stats_v6.json", f"{time.time() - T0:.0f}s", flush=True)

# ================================================================ markdown
L = []
w = L.append
w("# STATS V6: reliability, disattenuation, hierarchical bootstrap, robustness\n")
w("Generated by `scripts/v6/stats_v6.py` from `scripts/perimage_v6.py`. Numbers in `results/v6/stats_v6.json`.\n")
w("## Definitions\n")
for k, v in S["definitions"].items():
    if k != "method_sets":
        w(f"- **{k}**: {v}")
w("- **method sets**: " + ", ".join(f"{s} ({v['n']})" for s, v in S["definitions"]["method_sets"].items()))
w(f"- Units per dataset: " + ", ".join(f"{d} {S['n_units'][d]}/{S['n_images'][d]}" for d in DS) + " (units/images).")
w("\n## Checks against stats_v5\n")
w(f"- Point estimates (hier and comp, {S['check_point_estimates_vs_v5']['n_compared']} values): max abs diff {S['check_point_estimates_vs_v5']['max_abs_diff']:.2e}.")
w(f"- Method-only intervals regenerated with the stats_v5 RNG stream ({S['check_v5_intervals_reproduced']['n_compared']} intervals): max abs diff {S['check_v5_intervals_reproduced']['max_abs_diff']:.2e}.")
w(f"- Exact modality tests (diff and p, {S['check_v5_exact_reproduced']['n_compared']} cells): max abs diff {S['check_v5_exact_reproduced']['max_abs_diff']:.2e}.")
w(f"- Group mean Dice vs stats_v5 mean_dice: max abs diff {FLOOR['check_vs_v5_mean_dice_max_abs_diff']:.2e}.\n")

w("## 1. Split-half reliability (Spearman-Brown, median [2.5, 97.5] over 500 unit splits)\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| dataset | " + " | ".join(["U", "F", "K", "M", "ALL57"]) + (" | v5 ALL57 (image split) | ALL57 image split (here) |" if m == "dice" else " |"))
    w("|---" * (6 + (2 if m == "dice" else 0)) + "|")
    for d in DS:
        row = [fci(REL[f"{m}|{s}"][d]["spearman_brown_median"], REL[f"{m}|{s}"][d]["lo"], REL[f"{m}|{s}"][d]["hi"])
               for s in ("U", "F", "K", "M", "ALL57")]
        if m == "dice":
            row += [f3(V5["reliability"][d]["spearman_brown"]), f3(REL_IMG[d])]
        w(f"| {d} | " + " | ".join(row) + " |")

w("\n## 2. Disattenuated agreement\n")
w("Per split s: r_AB = mean(corr(A1,B1), corr(A2,B2)); r_AA = corr(A1,A2); dis = r_AB / sqrt(r_AA r_BB), undefined when r_AA r_BB <= 0. "
  "Category value per split = mean over its pairs; reported: mean over splits, median, [2.5, 97.5] of the split distribution "
  "(split-to-split spread, not a sampling interval), and the ratio-of-means version mean(r_AB)/sqrt(mean r_AA mean r_BB). "
  "Values above 1 are not clipped. The mean over splits is unstable when a half-level reliability "
  "is near zero in some splits (M on etis_larib Dice, K on azh/azh_train_slice NSD, M on co2wounds NSD): use the median or the "
  "ratio of means there; the huge means in those rows come from a few splits with r_AA r_BB just above 0.\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| set | category | observed (full data) | half-level r_AB | disattenuated mean | median [2.5, 97.5] | ratio of means | share >1 | undefined |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in GROUP_SETS:
        for c, v in DIS[f"{m}|{s}"]["summary"].items():
            w(f"| {s} | {c} | {f3(v['observed_full_mean'])} | {f3(v['half_r_ab_mean'])} | {f3(v['dis_mean'])} | "
              f"{fci(v['dis_median'], v['split_lo'], v['split_hi'])} | {f3(v['dis_ratio_of_means'])} | "
              f"{v['share_split_values_gt1']:.2f} | {v['n_undefined_pair_splits']}/{v['n_pair_splits']} |")

w("\n## 3. Hierarchical bootstrap vs method-only intervals\n")
w(f"Hierarchical: methods resampled within each group of the set and units resampled within each dataset independently, B = {B_HIER}, seed {SEED_HIER}. "
  "Units-only: units resampled, all methods kept. v5 method-only: interval as stored in stats_v5.json (source key in the JSON). "
  "Percentile intervals. Resampling units adds noise to every per-method mean, which attenuates agreement in each draw, so "
  "units-only and hierarchical intervals for agreement-type quantities sit partly below the point estimate (attenuation bias); "
  "this makes them conservative for 'greater than 0' claims.\n")
SHOW = ["modality_diff", "wound pairs", "same dataset", "modality_diff_excl_within_source"]
for m in METRICS:
    w(f"\n### {m}: headline quantities\n")
    w("| set | quantity | est | v5 method-only | units-only | hierarchical | P(hier <= 0) |")
    w("|---|---|---|---|---|---|---|")
    for s in V5_POOLS:
        for q in SHOW:
            b = BOOT[f"{m}|{s}"][q]
            iv = b["v5_method_only"]
            ivs = f"[{f3(iv[0])}, {f3(iv[1])}]" if iv else "n/a (repro " + f"[{f3(b['method_only_repro_B2000'][0])}, {f3(b['method_only_repro_B2000'][1])}])"
            w(f"| {s} | {q} | {f3(b['est'])} | {ivs} | [{f3(b['units_only'][0])}, {f3(b['units_only'][1])}] | "
              f"[{f3(b['hierarchical'][0])}, {f3(b['hierarchical'][1])}] | {b['hier_p_le0']:.3f} |")
    w(f"\n### {m}: category means (domain as in stats_v5 comp)\n")
    w("| set | domain | category | est | v5 method-only | units-only | hierarchical |")
    w("|---|---|---|---|---|---|---|")
    for s in V5_POOLS:
        doms = [k.split("|")[2] for k in V5["comp"] if k.startswith(f"{m}|{s}|")]
        for dom in doms:
            for k in COMP[dom]:
                b = BOOT[f"{m}|{s}"][f"comp|{dom}|{k}"]
                iv = b["v5_method_only"]
                w(f"| {s} | {dom} | {k} | {f3(b['est'])} | [{f3(iv[0])}, {f3(iv[1])}] | "
                  f"[{f3(b['units_only'][0])}, {f3(b['units_only'][1])}] | [{f3(b['hierarchical'][0])}, {f3(b['hierarchical'][1])}] |")

w("\n## 4. Modality test without the 3 within-source pairs\n")
w("Within-source pairs dropped by identity: " + ", ".join(WITHIN_SRC_NAMES) + ". 52 pairs remain (13 same modality, 39 across); "
  "relabeling over all 13,860 distinct assignments, the 3 pairs stay dropped under every relabeling. "
  "Swapping the two singleton labels (dermoscopy, ultrasound) gives the same pair partition, so the smallest attainable p is "
  "2/13,860 = 0.00014 (stats_v5 lists min_p 1/13,860, which cannot occur); '(min)' marks it.\n")
w("| metric | set | diff (55 pairs) | p | diff (52 pairs) | p | diff-source same-modality mean | across mean | same-source mean | control |")
w("|---|---|---|---|---|---|---|---|---|---|")
for m in ("dice", "nsd", "dice_noempty"):
    for s in GROUP_SETS:
        e = EX[f"{m}|{s}"]
        lv = e["levels_excl"]
        w(f"| {m} | {s} | {f3(e['full_55']['diff'])} | {fp(e['full_55'])} | {f3(e['excl_within_source_52']['diff'])} | "
          f"{fp(e['excl_within_source_52'])} | {f3(lv['different source, same modality (13)'])} | {f3(lv['across modality (39)'])} | "
          f"{f3(lv['same source (3, excluded)'])} | {f3(lv['same dataset (control)'])} |")
w("\n### Leave one dataset out (modality difference, exact p)\n")
w("Exact p over the distinct relabelings of the 10 remaining labels; '(min)' marks the smallest attainable p "
  "(2/5040 dropping a wound set; 4/6300 dropping a colonoscopy set, because the two size-4 labels also swap; 1/1260 dropping ph2 or tn3k_test).\n")
w("| metric | set | dropped | diff | p | diff excl within-source | p |")
w("|---|---|---|---|---|---|---|")
for k, v in LODO.items():
    m, s = k.split("|")
    for d, r in v.items():
        w(f"| {m} | {s} | {d} | {f3(r['full']['diff'])} | {fp(r['full'])} | {f3(r['excl_within_source']['diff'])} | {fp(r['excl_within_source'])} |")
w("\n### Leave one method out, M\n")
w("| metric | dropped | modality diff | excl within-source | wound pairs |")
w("|---|---|---|---|---|")
for m in METRICS:
    for n, r in LOMO[m].items():
        w(f"| {m} | {n} | {f3(r['modality_diff'])} | {f3(r['modality_diff_excl_within_source'])} | {f3(r['wound pairs'])} |")
    arr = np.array([[r["modality_diff"], r["wound pairs"]] for r in LOMO[m].values()])
    w(f"| {m} | range | {f3(arr[:, 0].min())} to {f3(arr[:, 0].max())} | | {f3(arr[:, 1].min())} to {f3(arr[:, 1].max())} |")

w("\n## 5. Stratified agreement equals the size-weighted within-group agreement\n")
w("| metric | set | quantity | stratified Spearman | Pearson on percentiles | exact weighted identity | size-weighted average | unweighted average | pooled unstratified | v5 stratified |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, v in STR.items():
    m, s = k.split("|")
    for q, v5k in (("wound pairs", "v5_stratified_wound_in_x_in"), ("modality_diff", "v5_stratified_modality_diff")):
        w(f"| {m} | {s} | {q} | {f3(v['stratified_spearman'][q])} | {f3(v['pearson_on_percentiles'][q])} | "
          f"{f3(v['weighted_identity'][q])} | {f3(v['size_weighted_average'][q])} | {f3(v['unweighted_average'][q])} | "
          f"{f3(v['pooled_unstratified'][q])} | {f3(v[v5k])} |")
w("\nMax abs per-pair difference, stratified vs size-weighted: " +
  ", ".join(f"{k} {v['max_abs_pair_diff_stratified_vs_weighted']:.3f}" for k, v in STR.items()) +
  ". Pearson on percentiles vs exact identity: max " +
  f"{max(v['max_abs_pair_diff_pearsonpct_vs_identity'] for v in STR.values()):.1e}.\n")
w("Why: within-group percentiles (rank - 0.5)/n_g have mean exactly 0.5 in every group, so there is no between-group "
  "covariance. The pooled covariance and variances are therefore sums of the within-group ones weighted by n_g, and the Pearson "
  "correlation of the pooled percentiles is sum_g n_g v_g r_g / sum_g n_g v_g, with r_g the within-group Spearman and "
  "v_g = (1 - 1/n_g^2)/12 without ties. That is a size-weighted average with weights n_g - 1/n_g, practically n_g. "
  "Spearman on the pooled percentiles re-ranks values that are already near-uniform grids, so it is close to the Pearson value. "
  "Hence stratified agreement adds no information beyond the within-group agreements: it reports their weighted mean.\n")

w("## 6. Empty-mask-excluded Dice modality difference\n")
w("Empty reference masks: " + ", ".join(f"{d} {n}" for d, n in S["n_empty"].items() if n) + ".\n")
w("| set | diff (no empty) | p | stats_v5 est | stats_v5 p | diff with empties | no-empty excl within-source |")
w("|---|---|---|---|---|---|---|")
for s, v in NOEMPTY.items():
    w(f"| {s} | {f3(v['modality_diff'])} | {fp(EX['dice_noempty|' + s]['full_55'])} | {f3(v['v5_est'])} | {v['v5_p']:.5f} | {f3(v['dice_with_empty'])} | {f3(v['excl_within_source'])} |")

w("\n## 7. Floor effect\n")
w("Group mean Dice (mean over methods of per-method mean over images). Bold = below 0.1.\n")
w("| dataset | " + " | ".join(["U", "F", "K", "M", "S"]) + " |")
w("|---|---|---|---|---|---|")
for d in DS:
    cells = []
    for g in ("U", "F", "K", "M", "S"):
        x = FLOOR["mean_dice"][g][d]
        cells.append(f"**{x:.3f}**" if x < 0.1 else f"{x:.3f}")
    w(f"| {d} | " + " | ".join(cells) + " |")
w("\nShare of images with Dice < 0.05 (median over the group's methods), non-wound datasets.\n")
w("| dataset | U | F | K | M | S |")
w("|---|---|---|---|---|---|")
for d in DS:
    if d in V.WOUND or d == V.CONTROL:
        continue
    w(f"| {d} | " + " | ".join(f"{FLOOR['share_dice_lt_0.05_median_over_methods'][g][d]:.2f}" for g in ("U", "F", "K", "M", "S")) + " |")
w(f"\nRuntime {S['runtime_s']} s.\n")
open(os.path.join(OUT, "STATS_V6.md"), "w").write("\n".join(L))
print("saved results/v6/STATS_V6.md", f"{time.time() - T0:.0f}s")

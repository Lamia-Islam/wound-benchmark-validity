#!/usr/bin/env python3
"""Blind replication of the V7 headline numbers from the raw JSON score files.

Written from scratch. It does not import or copy code from scripts/perimage_v6.py,
scripts/perimage_v7.py, scripts/v6/, scripts/v7/ (other than this file) or
scripts/v7_morning/. The only project import is woundbench.combined.ALL_METHOD_FAMILY,
used to split the 32 local methods into U (untrained) and F (trained from scratch).

Inputs (all read-only):
  results/perimage32/<set>.json                      32 local methods, per image
  results/kaggle/scores/{pretrained,nnunet,kvasir}/  run 1, per image
  results/kaggle2/scores/{pretrained_s1,pretrained_s2,modern2_s0,modern2_s1,modern2_s2}/  run 2
  results/kaggle2/rescore/<family>/<set>.json        threshold grid and P0 variants
  results/exclusions_v5.json                         images dropped before any mean
  results/pool_decomposition_cells.jsonl             U and F mean Dice on synthetic sets

Outputs: results/v7/blind_v7.json and results/v7/BLIND_V7.md.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from woundbench.combined import ALL_METHOD_FAMILY  # noqa: E402  (family map only)

WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
COLON = ["kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib"]
REFS = WOUND + COLON + ["ph2", "tn3k_test"]
MODALITY = {**{s: "wound" for s in WOUND}, **{s: "colon" for s in COLON},
            "ph2": "derm", "tn3k_test": "us"}
CONTROL = "azh_train_slice"
CLIN = REFS + [CONTROL]
SYN_SEEDS = [2024, 7, 101]
SYN40 = [f"genB_amp40_s{s}" for s in SYN_SEEDS]

P10 = ["unet_r18_in", "unetpp_r18_in", "deeplabv3p_r18_in", "fpn_r18_in", "pspnet_r18_in",
       "linknet_r18_in", "manet_r18_in", "segformer_mitb0_in", "unet_mbv2_in", "unet_effb0_in"]
N8 = ["unet_r50", "deeplabv3p_r50", "upernet_convnext_t", "unet_swin_t", "segformer_mitb2",
      "unet_effb3", "fpn_regnety016", "manet_r50"]
NN = "nnunet_2d"


def jload(*p):
    with open(os.path.join(RES, *p)) as f:
        return json.load(f)


EXCL = jload("exclusions_v5.json")


def keep_mask(s, n):
    """Boolean mask of kept images; synthetic sets and sets without entries keep all."""
    k = np.ones(n, bool)
    base = "fuseg" if s == "fuseg_r256" else s
    for e in EXCL.get(base, []):
        k[int(e[0])] = False
    return k


# ---------------------------------------------------------------- per-image columns
# COL[metric][set][model] = 1-D array over kept images
COL = {"dice": {}, "nsd": {}}
EMPTY = {}


def put(s, names, dice, nsd, gt_empty):
    D = np.asarray(dice, float)
    N = np.asarray(nsd, float)
    assert D.shape == (len(gt_empty), len(names)) and N.shape == D.shape
    assert not np.isnan(D).any() and not np.isnan(N).any()
    k = keep_mask(s, D.shape[0])
    ge = np.asarray(gt_empty, bool)
    if s in EMPTY:
        assert np.array_equal(EMPTY[s], ge[k]), s
    EMPTY[s] = ge[k]
    for j, nm in enumerate(names):
        assert nm not in COL["dice"].setdefault(s, {}), (s, nm)
        COL["dice"][s][nm] = D[k, j]
        COL["nsd"].setdefault(s, {})[nm] = N[k, j]


local_names = None
for s in CLIN:
    d = jload("perimage32", s + ".json")
    local_names = d["names"]
    put(s, d["names"], d["dice"], d["nsd"], d["gt_empty"])


def trained(n):
    f = ALL_METHOD_FAMILY[n]
    return f in ("learned", "CNN") or f.startswith("deep")


U = [n for n in local_names if not trained(n)]
F = [n for n in local_names if trained(n)]
assert len(U) == 18 and len(F) == 14

RUN1 = ["pretrained", "nnunet", "kvasir"]
RUN2 = ["pretrained_s1", "pretrained_s2", "modern2_s0", "modern2_s1", "modern2_s2"]
for s in CLIN + SYN40:
    for fam in RUN1:
        d = jload("kaggle", "scores", fam, s + ".json")
        put(s, d["names"], d["dice"], d["nsd"], d["gt_empty"])
    for fam in RUN2:
        d = jload("kaggle2", "scores", fam, s + ".json")
        put(s, d["names"], d["dice"], d["nsd"], d["gt_empty"])

K = jload("kaggle", "scores", "kvasir", "medetec.json")["names"]
assert len(K) == 14


def seed_model(arch, seed):
    """Name of the individual model of architecture arch at training seed seed."""
    if arch in P10:
        return arch if seed == 0 else f"{arch}_s{seed}"
    return f"{arch}_s{seed}"


def seeds_of(arch, s="medetec"):
    return [t for t in (0, 1, 2) if seed_model(arch, t) in COL["dice"][s]]


SEEDS = {a: seeds_of(a) for a in P10 + N8}
assert SEEDS["manet_r18_in"] == [0, 1] and SEEDS["manet_r50"] == [1]
assert sum(len(v) == 3 for v in SEEDS.values()) == 16

# seed-averaged columns, named '<arch>@seeds'
for m in ("dice", "nsd"):
    for s in COL[m]:
        for a in P10 + N8:
            cols = [COL[m][s][seed_model(a, t)] for t in SEEDS[a]]
            COL[m][s][a + "@seeds"] = np.mean(cols, axis=0)

SA = lambda archs: [a + "@seeds" for a in archs]  # noqa: E731
GROUPS = {
    "U": U, "F": F, "K": K,
    "M": P10 + [NN],
    "M11s": SA(P10) + [NN],
    "M19": SA(P10) + SA(N8) + [NN],
    "N8s": SA(N8),
    "M_s1": [seed_model(a, 1) for a in P10 if 1 in SEEDS[a]] + [NN],
    "M_s2": [seed_model(a, 2) for a in P10 if 2 in SEEDS[a]] + [NN],
}
GROUPS["U+M"] = U + GROUPS["M"]
GROUPS["U+M19"] = U + GROUPS["M19"]
assert [len(GROUPS[g]) for g in ("M", "M11s", "M19", "N8s", "M_s1", "M_s2", "U+M", "U+M19")] == \
    [11, 11, 19, 8, 11, 10, 29, 37]


def means(m, s, names, col=COL):
    return np.array([col[m][s][n].mean() for n in names])


def rho(x, y):
    """Spearman with average ranks = Pearson of the average ranks."""
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


PAIRS = list(itertools.combinations(REFS, 2))
assert len(PAIRS) == 55
WPAIRS = list(itertools.combinations(WOUND, 2))


def pair_matrix(m, names, col=COL, sets=REFS):
    mu = {s: means(m, s, names, col) for s in sets}
    return {(a, b): rho(mu[a], mu[b]) for a, b in itertools.combinations(sets, 2)}


def wound_mean(m, names, col=COL, wound=WOUND):
    mu = {s: means(m, s, names, col) for s in wound}
    return float(np.mean([rho(mu[a], mu[b]) for a, b in itertools.combinations(wound, 2)]))


def modality_diff(R, labels):
    same = [v for (a, b), v in R.items() if labels[a] == labels[b]]
    across = [v for (a, b), v in R.items() if labels[a] != labels[b]]
    return float(np.mean(same) - np.mean(across)), len(same), len(across)


def all_labelings():
    """Every distinct assignment of the multiset of 11 modality labels to the 11 refs."""
    out = []
    idx = range(11)
    for w in itertools.combinations(idx, 4):
        rest = [i for i in idx if i not in w]
        for c in itertools.combinations(rest, 5):
            rest2 = [i for i in rest if i not in c]
            for d in rest2:  # derm takes one of the remaining two, us the other
                lab = {}
                for i in w:
                    lab[REFS[i]] = "wound"
                for i in c:
                    lab[REFS[i]] = "colon"
                lab[REFS[d]] = "derm"
                lab[REFS[[i for i in rest2 if i != d][0]]] = "us"
                out.append(lab)
    return out


LABELINGS = all_labelings()
assert len(LABELINGS) == 13860


def exact_p(R):
    obs, ns, na = modality_diff(R, MODALITY)
    assert (ns, na) == (16, 39)
    stats = np.array([modality_diff(R, lab)[0] for lab in LABELINGS])
    return obs, float(np.mean(stats >= obs - 1e-12)), int(np.sum(stats >= obs - 1e-12))


OUT = {"checks": {}, "items": []}
OUT["checks"]["group_sizes"] = {g: len(v) for g, v in GROUPS.items()}
OUT["checks"]["seeds_per_arch"] = SEEDS
OUT["checks"]["kept_images"] = {s: int(len(EMPTY[s])) for s in CLIN}


def rec(item, label, reported, mine, decimals=3, note=""):
    ok = round(mine, decimals) == round(reported, decimals) or \
        abs(mine - reported) <= 0.5 * 10 ** (-decimals) + 1e-12
    OUT["items"].append({"item": item, "quantity": label, "reported": reported,
                         "replicated": mine, "abs_diff": abs(mine - reported),
                         "decimals": decimals, "status": "PASS" if ok else "FAIL",
                         "note": note})


# ---------------------------------------------------------------- 1. wound pairs
for g, rep in (("M", 0.126), ("M11s", 0.274), ("M19", 0.211), ("N8s", 0.258)):
    rec(1, f"wound-pair agreement, Dice, {g}", rep, wound_mean("dice", GROUPS[g]))
rec(1, "wound-pair agreement, NSD, M19", 0.473, wound_mean("nsd", GROUPS["M19"]))

# ---------------------------------------------------------------- 2. mixed pools
rec(2, "wound-pair agreement, Dice, U+M19", 0.821, wound_mean("dice", GROUPS["U+M19"]))
rec(2, "wound-pair agreement, Dice, U+M", 0.801, wound_mean("dice", GROUPS["U+M"]))

# ---------------------------------------------------------------- 3 and 4. modality
EXACT = {}
for g, rep in (("M", 0.461), ("M19", 0.225), ("M11s", 0.329), ("N8s", 0.055),
               ("U+M19", 1.101), ("M_s1", 0.024), ("M_s2", 0.180)):
    R = pair_matrix("dice", GROUPS[g])
    obs, p, cnt = exact_p(R)
    EXACT[g] = {"diff": obs, "p": p, "count_ge": cnt}
    lab = {"M_s1": "M seed 1 (10 pretrained s1 + nnU-Net)",
           "M_s2": "M seed 2 (9 pretrained s2 + nnU-Net)"}.get(g, g)
    rec(3, f"modality difference, Dice, {lab}", rep, obs)
rec(4, "exact relabeling p, M19 Dice modality difference", 0.00577, EXACT["M19"]["p"], 5,
    f"{EXACT['M19']['count_ge']} of 13860 labelings with statistic >= observed")
rec(4, "exact relabeling p, U+M19 Dice modality difference", 0.00014, EXACT["U+M19"]["p"], 5,
    f"{EXACT['U+M19']['count_ge']} of 13860 (2 is the minimum: ph2 and tn3k labels swap)")
OUT["exact_tests"] = EXACT

# ---------------------------------------------------------------- 5. seeds
C16 = [a for a in P10 + N8 if len(SEEDS[a]) == 3]
P9 = [a for a in C16 if a in P10]
N7 = [a for a in C16 if a in N8]
assert len(C16) == 16 and len(P9) == 9 and len(N7) == 7


def seed_means(m, s, archs, t):
    return np.array([COL[m][s][seed_model(a, t)].mean() for a in archs])


def seed_block(archs, m="dice"):
    sp = [(0, 1), (0, 2), (1, 2)]
    rbar = {s: float(np.mean([rho(seed_means(m, s, archs, a), seed_means(m, s, archs, b))
                              for a, b in sp])) for s in WOUND + [CONTROL]}
    same = float(np.mean([rbar[s] for s in WOUND]))
    single = [float(np.mean([rho(seed_means(m, a, archs, t), seed_means(m, b, archs, t))
                             for a, b in WPAIRS])) for t in (0, 1, 2)]
    avg = wound_mean(m, SA(archs))
    cross = [rho(seed_means(m, a, archs, s), seed_means(m, b, archs, t))
             for a, b in WPAIRS for s in (0, 1, 2) for t in (0, 1, 2) if s != t]
    assert len(cross) == 36
    cross = float(np.mean(cross))
    return {"rbar_by_set": rbar, "same_dataset_diff_seed": same, "single_seed": single,
            "single_seed_mean": float(np.mean(single)), "seed_average": avg,
            "cross_dataset_cross_seed": cross, "gap": same - cross}


SEEDRES = {"C16": seed_block(C16), "P9": seed_block(P9), "N7": seed_block(N7)}
OUT["seed_analysis"] = SEEDRES
rec(5, "same wound dataset, different seed, mean Spearman (C16)", 0.446,
    SEEDRES["C16"]["same_dataset_diff_seed"])
rec(5, "wound-pair agreement, single seeds (mean of 3), C16", 0.177,
    SEEDRES["C16"]["single_seed_mean"])
rec(5, "wound-pair agreement, seed averages, C16", 0.179, SEEDRES["C16"]["seed_average"])
rec(5, "different wound dataset and different seed (C16)", 0.100,
    SEEDRES["C16"]["cross_dataset_cross_seed"])
rec(5, "gap, C16", 0.346, SEEDRES["C16"]["gap"])
rec(5, "gap, 9 original architectures", 0.493, SEEDRES["P9"]["gap"])
rec(5, "gap, 7 new architectures", 0.055, SEEDRES["N7"]["gap"])

# ---------------------------------------------------------------- 6. benchmark
SYN_LOCAL = {}
with open(os.path.join(RES, "pool_decomposition_cells.jsonl")) as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            SYN_LOCAL[(int(c["amp"]), int(c["seed"]))] = dict(zip(c["names"], c["dice"]))


def syn_means(names, seed):
    s = f"genB_amp40_s{seed}"
    out = []
    for n in names:
        if n in COL["dice"][s]:
            out.append(COL["dice"][s][n].mean())
        else:
            out.append(float(SYN_LOCAL[(40, seed)][n]))
    return np.array(out)


def bench(names):
    per_ref = {}
    for r in WOUND:
        mr = means("dice", r, names)
        per_ref[r] = float(np.mean([rho(syn_means(names, sd), mr) for sd in SYN_SEEDS]))
    return float(np.mean(list(per_ref.values()))), per_ref


BENCH = {}
for g, rep in (("M", 0.353), ("M11s", 0.408), ("M19", 0.208), ("N8s", -0.127),
               ("U+M19", 0.647), ("U+M", 0.721)):
    v, pr = bench(GROUPS[g])
    BENCH[g] = {"mean": v, "per_ref": pr}
    rec(6, f"benchmark amp40 agreement with 4 wound sets, {g}", rep, v,
        note="U synthetic means from results/pool_decomposition_cells.jsonl" if "U" in g else "")
# synthetic image counts
OUT["checks"]["synthetic_images"] = {s: int(len(COL["dice"][s][NN])) for s in SYN40}
OUT["benchmark"] = BENCH

# ---------------------------------------------------------------- 7. practical
cand = GROUPS["M19"]
MU = {s: means("dice", s, cand) for s in WOUND + [CONTROL]}
loss = {}
for a in WOUND:
    pick = int(np.argmax(MU[a]))
    for b in WOUND:
        loss[(a, b)] = float(MU[b].max() - MU[b][pick])
off = [loss[(a, b)] for a in WOUND for b in WOUND if a != b]
rand = {b: float(np.mean(MU[b].max() - MU[b])) for b in WOUND}
rec(7, "M19 selection loss, choose on another wound set (mean of 12)", 0.047, float(np.mean(off)))
rec(7, "M19 random-pick expected loss (mean of 4 wound sets)", 0.063, float(np.mean(list(rand.values()))))
azh_best = cand[int(np.argmax(MU["azh"]))]
OUT["practical"] = {"loss_matrix": {f"{a}->{b}": v for (a, b), v in loss.items()},
                    "random": rand, "azh_best": azh_best,
                    "best_per_set": {s: cand[int(np.argmax(MU[s]))] for s in MU}}
rec(7, f"AZH best M19 method mean Dice ({azh_best})", 0.713, float(MU["azh"].max()),
    note="reported best is segformer_mitb2@seeds; replicated best is " + azh_best)
rec(7, "AZH nnU-Net mean Dice", 0.631, float(COL["dice"]["azh"][NN].mean()))


def miss_rate(s, model):
    ne = ~EMPTY[s]
    return float(np.mean(COL["dice"][s][model][ne] < 0.1))


def arch_miss(s, a):
    return float(np.mean([miss_rate(s, seed_model(a, t)) for t in SEEDS[a]]))


ne_azh = int((~EMPTY["azh"]).sum())
new_arch = float(np.mean([arch_miss("azh", a) for a in N8]))
old_arch = float(np.mean([arch_miss("azh", a) for a in P10]))
new_model = float(np.mean([miss_rate("azh", seed_model(a, t)) for a in N8 for t in SEEDS[a]]))
old_model = float(np.mean([miss_rate("azh", seed_model(a, t)) for a in P10 for t in SEEDS[a]]))
OUT["practical"]["azh_miss"] = {
    "non_empty_images": ne_azh,
    "new8_equal_weight_per_architecture": new_arch, "new8_equal_weight_per_model_22": new_model,
    "orig10_equal_weight_per_architecture": old_arch, "orig10_equal_weight_per_model_29": old_model,
    "per_architecture": {a: arch_miss("azh", a) for a in P10 + N8},
    "nnunet": miss_rate("azh", NN)}
rec(7, "AZH miss rate, new 8 architectures, all seeds (%)", 24.9, 100 * new_arch, 1,
    f"equal weight per architecture (PRACTICAL_V7 definition); equal weight per model (22 models) gives {100 * new_model:.2f}%")
rec(7, "AZH miss rate, original 10 architectures, all seeds (%)", 30.3, 100 * old_arch, 1,
    f"equal weight per architecture; per model (29 models) {100 * old_model:.2f}%")
rec(7, "AZH miss rate, nnU-Net (%)", 1.4, 100 * miss_rate("azh", NN), 1)
rec(7, "AZH miss rate, segformer_mitb2, mean over 3 seeds (%)", 11.9,
    100 * arch_miss("azh", "segformer_mitb2"), 1)

# ---------------------------------------------------------------- 8. robustness (rescore files)
RFAM = ["pretrained_s0", "pretrained_s1", "pretrained_s2", "modern2_s0", "modern2_s1",
        "modern2_s2", "nnunet"]
RSETS = CLIN + ["fuseg_r256"]
GRID = None
RC = {"p0": {}, "common50": {}, "oracle": {}}  # [set][model] per kept image
for s in RSETS:
    for k in RC:
        RC[k][s] = {}
    for fam in RFAM:
        d = jload("kaggle2", "rescore", fam, s + ".json")
        if GRID is None:
            GRID = list(d["grid_thresholds"])
        assert list(d["grid_thresholds"]) == GRID and d["variants"][0] == "P0" and d["metrics"][0] == "dice"
        kp = keep_mask(s, d["n"])
        if s in EMPTY:
            assert np.array_equal(np.asarray(d["gt_empty"], bool)[kp], EMPTY[s]), (fam, s)
        G = np.asarray(d["grid_dice"], float)[kp]          # images x models x grid
        V = np.asarray(d["var"], float)[kp]                # images x models x variant x metric
        gi = GRID.index(0.5)
        for j, nm in enumerate(d["names"]):
            best = int(np.argmax(G[:, j, :].mean(0)))
            RC["p0"][s][nm] = V[:, j, 0, 0]
            RC["common50"][s][nm] = G[:, j, gi]
            RC["oracle"][s][nm] = G[:, j, best]


def rmodel(a, t):
    """Model name in the rescore files (seed 0 of the original 10 has no suffix)."""
    return seed_model(a, t)


for k in RC:
    for s in RSETS:
        for a in P10 + N8:
            ts = [t for t in (0, 1, 2) if rmodel(a, t) in RC[k][s]]
            assert ts == SEEDS[a], (k, s, a, ts)
            RC[k][s][a + "@seeds"] = np.mean([RC[k][s][rmodel(a, t)] for t in ts], axis=0)

RCOL = {k: {"dice": RC[k]} for k in RC}
m19 = GROUPS["M19"]
ROB = {
    "p0_wound": wound_mean("dice", m19, RCOL["p0"]),
    "common50_wound": wound_mean("dice", m19, RCOL["common50"]),
    "oracle_wound": wound_mean("dice", m19, RCOL["oracle"]),
    "r256_wound_all_rescore": wound_mean("dice", m19, RCOL["p0"],
                                         ["medetec", "co2wounds", "fuseg_r256", "azh"]),
}
# mixed variant: run-1 scores for the three native sets, rescore P0 for fuseg_r256
MIX = {"dice": {s: COL["dice"][s] for s in WOUND}}
MIX["dice"]["fuseg_r256"] = RC["p0"]["fuseg_r256"]
ROB["r256_wound_mixed"] = wound_mean("dice", m19, MIX, ["medetec", "co2wounds", "fuseg_r256", "azh"])
ROB["max_abs_p0_minus_run1_mean"] = float(max(
    abs(RC["p0"][s][n].mean() - COL["dice"][s][n].mean()) for s in CLIN for n in m19))
OUT["robustness"] = ROB
rec(8, "M19 wound pairs, rescore P0 (check, should equal 0.211)", 0.211, ROB["p0_wound"])
rec(8, "M19 wound pairs, one common threshold 0.50", 0.412, ROB["common50_wound"])
rec(8, "M19 wound pairs, per-dataset oracle threshold", 0.122, ROB["oracle_wound"])
rec(8, "M19 wound pairs, fuseg replaced by fuseg_r256 (P0)", 0.171, ROB["r256_wound_all_rescore"],
    note=f"with run-1 scores for the other three sets: {ROB['r256_wound_mixed']:.6f}")

# ---------------------------------------------------------------- independent route check
OUT["checks"]["scipy_spearman_M19_wound"] = float(np.mean([
    spearmanr(means("dice", a, m19), means("dice", b, m19))[0] for a, b in WPAIRS]))

# ---------------------------------------------------------------- write
OUT["n_pass"] = sum(i["status"] == "PASS" for i in OUT["items"])
OUT["n_fail"] = sum(i["status"] == "FAIL" for i in OUT["items"])
with open(os.path.join(RES, "v7", "blind_v7.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)

for i in OUT["items"]:
    print(f"{i['item']} | {i['quantity']} | {i['reported']} | {i['replicated']:.6f} | "
          f"{i['abs_diff']:.2e} | {i['status']} | {i['note']}")
print("PASS", OUT["n_pass"], "FAIL", OUT["n_fail"])
print(json.dumps(OUT["checks"], default=float)[:1500])

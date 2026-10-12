#!/usr/bin/env python3
"""V7 synthetic-benchmark analysis: bench_v6 rerun with the phase-2 modern group.

Everything is defined exactly as in scripts/v6/bench_v6.py (imported, not
rewritten; see "Reuse" below):
  agreement(A, B | P)  Spearman correlation (average ranks) between per-method
                       mean Dice on set A and on set B, over method set P.
  bench_a(R)           mean over synthetic seeds 2024, 7, 101 of
                       agreement(genB_amp{a}_s{seed}, R).
  other(R)             mean over the three other clinical wound datasets R' of
                       agreement(R', R).
  diff_a(R)            bench_a(R) - other(R). Positive: the synthetic benchmark
                       ranks the methods more like R than the other clinical
                       wound datasets do. "mean" row: average over the four
                       wound references (Medetec, CO2Wounds-V2, FUSeg, AZH).

Method sets
  Primary (as V6 with M replaced): U (18), F (14), K (14), M19 (19),
  U+F (32), U+K (32), U+M19 (37).
  M19 = 18 ImageNet-pretrained architectures, each scored as the mean over its
  available training seeds, + nnU-Net (one training).
  Sensitivity: M (= M11, V6), U+M, M17 (M19 without the two MAnet
  architectures), U+M17, per-seed groups Ms0, Ms1, Ms2 (every architecture that
  has that training seed, as a single model, + nnU-Net) and U+Ms0..2.
  Group sizes: Ms0 18 (manet_r50 has no seed 0), Ms1 19, Ms2 17 (no seed 2 for
  either MAnet).

Seed-averaged synthetic scores
  For an architecture, the per-image synthetic Dice is the mean over its
  available training seeds of the per-image Dice of each seed's model (the same
  rule perimage_v7.load7 applies on clinical sets). Seed 0 of the ten run-1
  architectures is read from results/kaggle/scores/pretrained, seeds 1 and 2
  from results/kaggle2/scores/pretrained_s{1,2}; the eight phase-2
  architectures from results/kaggle2/scores/modern2_s{0,1,2}.

Uncertainty (unchanged from V6)
  method bootstrap   methods resampled with replacement within each group
                     (mixtures keep their composition), B = 2,000 draws,
                     default_rng(0) per method set, percentile 95% intervals.
  hierarchical       modern groups only: methods within group AND,
                     independently per dataset, clinical units (Medetec
                     photograph, AZH photograph, image otherwise) and synthetic
                     images; default_rng(1), 2,000 draws.
  data-only          as hierarchical with methods fixed; default_rng(2).

Reuse
  Imported from bench_v6 without change: sp, zr, c, f_rel, f_texture,
  f_matched, combine, ci, method_takes, mboot, unit_struct, split, excl0,
  table, names_of (the last four act on bench_v6's GRP/POOLS/TT dictionaries,
  which this script extends with new keys; no V6 entry is modified).
  Copied (because the V6 versions read the per-image data through the V6
  loader, which has no phase-2 columns): per_image -> per_image7,
  hboot -> hboot7 (same RNG calls in the same order), the generator-ordering
  block of bench_v6.main -> gen_order, and the leave-one-method-out block.

Output: results/v7/bench_v7.json and results/v7/BENCH_V7.md
"""
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "v6"))
import perimage_v6 as V6                                           # noqa: E402
import perimage_v7 as V7                                           # noqa: E402
import bench_v6 as B6                                              # noqa: E402

B = B6.B
SEEDS, AMPS, GENS = B6.SEEDS, B6.AMPS, B6.GENS
WOUND, GB, NR, SYNTH = B6.WOUND, B6.GB, B6.NR, B6.SYNTH
OUT = os.path.join(ROOT, "results", "v7")
V6JSON = os.path.join(ROOT, "results", "v6", "bench_v6.json")
TRAIN_SEEDS = V7.SEEDS
ARCH = V7.ARCH


# ------------------------------------------------------- synthetic helpers

def synth_seed(s, arch, seed):
    """Per-image synthetic Dice of one architecture-seed model, or None."""
    d, name = V7._src(arch, seed)
    p = os.path.join(d, s + ".json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    if name not in g["names"]:
        return None
    assert g["set"] == s and g["n"] == len(g["dice"]) == 45, (s, arch, seed)
    assert s not in V6.EXCL
    return np.asarray(g["dice"], float)[:, g["names"].index(name)]


def synth_avg(s, arch):
    """Seed-averaged per-image synthetic Dice (mean over available seeds)."""
    vs, ks = [], []
    for k in TRAIN_SEEDS:
        v = synth_seed(s, arch, k)
        if v is not None:
            vs.append(v); ks.append(k)
    assert ks == V7.seeds_of(arch), (s, arch, ks)          # same seeds as clinical
    return np.mean(vs, axis=0)


_CLIN = {}


def clin(ref):
    if ref not in _CLIN:
        _CLIN[ref] = V7.load7(ref, "dice")
    return _CLIN[ref]


def _parse(n):
    m = re.match(r"^(.+)@s(\d)$", n)
    return (m.group(1), int(m.group(2))) if m else None


def clin_X(ref, names):
    D = clin(ref)
    cols = []
    for n in names:
        ps = _parse(n)
        if ps:
            X, archs = V7.load_seed(ref, "dice", ps[1])
            cols.append(X[:, archs.index(ps[0])])
        else:
            cols.append(D["X"][:, V7.NAMES7.index(n)])
    return np.stack(cols, 1)


_SX = {}


def synth_X7(s, names):
    cols = []
    for n in names:
        if (s, n) not in _SX:
            ps = _parse(n)
            if ps:
                v = synth_seed(s, *ps)
            elif n.endswith("@avg"):
                v = synth_avg(s, n[:-4])
            else:
                g = next(k for k in ("K", "M", "S") if n in B6.GRP[k])
                v = B6.synth_X(s, g)[:, B6.GRP[g].index(n)]
            assert v is not None, (s, n)
            _SX[(s, n)] = v
        cols.append(_SX[(s, n)])
    return np.stack(cols, 1)


# ------------------------------------------------------- groups and table

def seed_names(seed):
    return [f"{a}@s{seed}" for a in ARCH if seed in V7.seeds_of(a)] + ["nnunet_2d"]


NEW_GRP = {"M19": list(V7.GROUPS7["M19"]), "M17": list(V7.GROUPS7["M17"])}
for k in TRAIN_SEEDS:
    NEW_GRP[f"Ms{k}"] = seed_names(k)
PRIMARY = ["U", "F", "K", "M19", "U+F", "U+K", "U+M19"]
SENS = ["M", "U+M", "M17", "U+M17", "Ms0", "U+Ms0", "Ms1", "U+Ms1", "Ms2", "U+Ms2"]
MODERN = ["M19", "M", "M17", "Ms0", "Ms1", "Ms2"]


def extend_bench_v6():
    """Add the new groups to bench_v6's GRP/POOLS and their per-method means
    to TT (wound sets and the 21 synthetic sets). Means of seed-averaged
    columns are computed as the mean of per-seed means (a second route to the
    per-image average used in per_image7; checked equal there)."""
    for g, nm in NEW_GRP.items():
        assert g not in B6.GRP
        B6.GRP[g] = nm
        B6.POOLS[g] = [g]
        B6.POOLS["U+" + g] = ["U", g]
    for s in WOUND + SYNTH:
        add = {}
        for a in ARCH:
            per = {}
            for k in TRAIN_SEEDS:
                if s in WOUND:
                    X, archs = V7.load_seed(s, "dice", k)
                    v = X[:, archs.index(a)] if a in archs else None
                else:
                    v = synth_seed(s, a, k)
                if v is not None:
                    per[k] = float(v.mean())
                    add[f"{a}@s{k}"] = per[k]
            add[f"{a}@avg"] = float(np.mean(list(per.values())))
        for n in add:
            assert n not in B6.TT[s]
        B6.TT[s].update(add)
        assert "nnunet_2d" in B6.TT[s]


# ------------------------------------------- per-image data (copied, adapted)

def per_image7(names, sets):
    """Copy of bench_v6.per_image reading through clin_X / synth_X7."""
    out, checks = {}, []
    for s in sets:
        if s in WOUND:
            X = clin_X(s, names)
            S, n = B6.unit_struct(X, list(clin(s)["units"]))
        else:
            X = synth_X7(s, names)
            S, n = B6.unit_struct(X, list(range(len(X))))
        checks.append(float(np.abs(X.mean(0) - B6.P.vec(B6.TT, s, names)).max()))
        out[s] = (S, n, X.shape[0])
    return out, max(checks)


def hboot7(g, sets, fn, resample_methods=True, seed=1):
    """Copy of bench_v6.hboot (same RNG calls, same order)."""
    data, _ = per_image7(B6.GRP[g], sets)
    full = {s: S.sum(0) / n.sum() for s, (S, n, _) in data.items()}
    obs = fn(B6.zr(full))
    rng = np.random.default_rng(seed)
    k = len(B6.GRP[g])
    draws = {kk: [] for kk in obs}
    for _ in range(B):
        t = rng.choice(k, k) if resample_methods else np.arange(k)
        M = {}
        for s, (S, n, _) in data.items():
            u = rng.integers(0, len(n), len(n))
            M[s] = (S[u].sum(0) / n[u].sum())[t]
        r = fn(B6.zr(M))
        for kk in obs:
            draws[kk].append(r[kk])
    return B6.ci(obs, draws)


def gen_order(R, pools):
    """Copy of the ordering block in bench_v6.main, over the given pools."""
    order = {}
    for t in ("M+A", "M+C", "mean4"):
        order[t] = {}
        for pool in pools:
            m = R["matched"][pool]
            o = {"seedmean": sorted(GENS, key=lambda g: -m[f"{g}_seedmean|{t}"]["est"])}
            for s in SEEDS:
                o[f"s{s}"] = sorted(GENS, key=lambda g: -m[f"{g}_s{s}|{t}"]["est"])
            order[t][pool] = {k: ">".join(v) for k, v in o.items()}
        sm = {order[t][p]["seedmean"] for p in pools}
        pair = {}
        for a, b in itertools.combinations(GENS, 2):
            wins = sum(R["matched"][p][f"{a}_s{s}|{t}"]["est"] > R["matched"][p][f"{b}_s{s}|{t}"]["est"]
                       for p in pools for s in SEEDS)
            wins_sm = sum(R["matched"][p][f"{a}_seedmean|{t}"]["est"] > R["matched"][p][f"{b}_seedmean|{t}"]["est"]
                          for p in pools)
            pair[f"{a}>{b}"] = {"pool_seed_cells": f"{wins}/{len(pools) * len(SEEDS)}",
                                "pools_seedmean": f"{wins_sm}/{len(pools)}"}
        order[t]["_stable_seedmean_ordering_across_pools"] = len(sm) == 1
        order[t]["_distinct_seedmean_orderings"] = sorted(sm)
        order[t]["_pairwise"] = pair
    return order


def lomo(g):
    nm = B6.GRP[g]
    full = B6.table(WOUND + GB[40] + GB[11], g)
    base = B6.f_rel(40)(B6.zr(full))
    out = {"_all": {"bench40_mean": base["mean|bench"], "pairs6": base["pairs6"],
                    "rel40_mean": base["mean|diff"]}}
    for i, n in enumerate(nm):
        keep = np.array([j for j in range(len(nm)) if j != i])
        r = B6.f_rel(40)(B6.zr({s: v[keep] for s, v in full.items()}))
        out[n] = {"bench40_mean": r["mean|bench"], "pairs6": r["pairs6"], "rel40_mean": r["mean|diff"]}
    return out


# ------------------------------------------------------------------ checks

def max_abs(a, b, path=""):
    """Max abs difference over all numeric leaves of two nested dicts with the
    same keys (NaN == NaN counts as 0). Raises on a key mismatch."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        return max([max_abs(a[k], b[k], path + "/" + str(k)) for k in a] + [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        return abs(float(a) - float(b))
    assert a == b, (path, a, b)
    return 0.0


def run_checks():
    C = {}
    # (a) synthetic helper, seed 0 of the ten run-1 architectures = V6 columns
    d0 = 0.0
    dm = 0.0
    for s in SYNTH:
        XM = B6.synth_X(s, "M")
        for a in V7.ARCH_OLD:
            v = synth_seed(s, a, 0)
            w = XM[:, B6.GRP["M"].index(f"{a}_in")]
            d0 = max(d0, float(np.abs(v - w).max()))
            dm = max(dm, abs(float(v.mean()) - B6.TT[s][f"{a}_in"]))
    C["synth_seed0_vs_v6_perimage_max_abs"] = d0
    C["synth_seed0_mean_vs_pools_v5_max_abs"] = dm
    # (b) clinical seed 0 = V6 columns
    d1 = 0.0
    for r in WOUND:
        D = V6.load(r)
        X0, archs = V7.load_seed(r, "dice", 0)
        for a in V7.ARCH_OLD:
            d1 = max(d1, float(np.abs(X0[:, archs.index(a)] - D["X"][:, V6.NAMES_ALL.index(f"{a}_in")]).max()))
    C["clinical_seed0_vs_v6_max_abs"] = d1
    # (c) seed-averaged per-image mean = mean of per-seed means (two routes)
    d2 = 0.0
    for s in WOUND + SYNTH:
        X = clin_X(s, NEW_GRP["M19"]) if s in WOUND else synth_X7(s, NEW_GRP["M19"])
        d2 = max(d2, float(np.abs(X.mean(0) - B6.P.vec(B6.TT, s, NEW_GRP["M19"])).max()))
    C["avg_perimage_mean_vs_mean_of_seed_means_max_abs"] = d2
    # (d) clinical seed-averaged columns of load7 = average of load_seed columns
    d3 = 0.0
    for r in WOUND:
        per = {k: V7.load_seed(r, "dice", k) for k in TRAIN_SEEDS}
        for a in ARCH:
            vs = [X[:, ar.index(a)] for X, ar in per.values() if a in ar]
            d3 = max(d3, float(np.abs(np.mean(vs, 0) - clin(r)["X"][:, V7.NAMES7.index(f"{a}@avg")]).max()))
    C["load7_avg_vs_load_seed_average_max_abs"] = d3
    # (e) synthetic image order identical across all score files of a set
    bad = []
    for s in SYNTH:
        ref = json.load(open(os.path.join(V6.RES, "kaggle", "scores", "pretrained", s + ".json")))
        for grp in ["modern2_s0", "modern2_s1", "modern2_s2", "pretrained_s1", "pretrained_s2"]:
            g = json.load(open(os.path.join(V7.K2, grp, s + ".json")))
            if g["gt_fraction"] != ref["gt_fraction"] or g["gt_empty"] != ref["gt_empty"]:
                bad.append(f"{grp}/{s}")
    C["synthetic_gt_order_mismatches"] = bad
    # (f) seeds available per architecture
    C["seeds_per_arch"] = {a: V7.seeds_of(a) for a in ARCH}
    C["n_arch_seed_models"] = int(sum(len(v) for v in C["seeds_per_arch"].values()))
    for k in ("synth_seed0_vs_v6_perimage_max_abs", "clinical_seed0_vs_v6_max_abs",
              "avg_perimage_mean_vs_mean_of_seed_means_max_abs",
              "load7_avg_vs_load_seed_average_max_abs"):
        assert C[k] < 1e-12, (k, C[k])
    assert C["synth_seed0_mean_vs_pools_v5_max_abs"] < 1e-12
    assert not bad, bad
    assert C["n_arch_seed_models"] == 51
    return C


# ------------------------------------------------------------------ main

def main():
    extend_bench_v6()
    R = {"definitions": __doc__, "B": B,
         "pools": {p: len(B6.names_of(p)) for p in PRIMARY + SENS},
         "groups": {g: B6.GRP[g] for g in NEW_GRP}}
    R["checks"] = run_checks()
    print("checks ok", flush=True)

    R["rel"], R["texture"], R["matched"] = {}, {}, {}
    sets = WOUND + SYNTH
    for pool in PRIMARY + SENS:
        takes = B6.method_takes(pool, 0)
        res = B6.mboot(pool, sets, B6.combine(B6.f_rel(40), B6.f_rel(11), B6.f_texture, B6.f_matched), takes)
        R["rel"][pool] = {"40": B6.split(res, "0:"), "11": B6.split(res, "1:")}
        R["texture"][pool] = B6.split(res, "2:")
        R["matched"][pool] = B6.split(res, "3:")
        print(pool, "method bootstrap done", flush=True)

    R["rel_hier"], R["rel_dataonly"], R["texture_hier"] = {}, {}, {}
    hsets = WOUND + GB[40] + GB[11]
    for g in MODERN:
        fn = B6.combine(B6.f_rel(40), B6.f_rel(11), B6.f_texture)
        h = hboot7(g, hsets, fn, True, 1)
        d = hboot7(g, hsets, fn, False, 2)
        R["rel_hier"][g] = {"40": B6.split(h, "0:"), "11": B6.split(h, "1:")}
        R["texture_hier"][g] = B6.split(h, "2:")
        R["rel_dataonly"][g] = {"40": B6.split(d, "0:"), "11": B6.split(d, "1:")}
        print(g, "hierarchical done", flush=True)
    R["loader_max_abs_diff"] = {g: per_image7(B6.GRP[g], WOUND + SYNTH)[1] for g in MODERN}

    within = [(p, r) for p in ("U", "F", "K", "M19") for r in WOUND]
    allp = [(p, r) for p in PRIMARY for r in WOUND]
    R["texture_counts"] = {
        "within_group_per_dataset": {
            "total": len(within),
            "exclude_zero": [f"{p}|{r}" for p, r in within if B6.excl0(R["texture"][p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(within)},
        "all_sets_per_dataset": {
            "total": len(allp),
            "exclude_zero": [f"{p}|{r}" for p, r in allp if B6.excl0(R["texture"][p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(allp)},
        "mean_row_exclude_zero": [p for p in PRIMARY if B6.excl0(R["texture"][p]["mean"])],
        "modern_sensitivity_mean_row_exclude_zero": [p for p in SENS if B6.excl0(R["texture"][p]["mean"])]}
    R["gen_order"] = gen_order(R, PRIMARY)
    R["lomo"] = {g: lomo(g) for g in ("M19", "M")}

    md = {}
    for g in ("U", "F", "K", "M", "S", "M19", "M17", "Ms0", "Ms1", "Ms2"):
        md[g] = {s: float(np.mean(B6.P.vec(B6.TT, s, B6.GRP[g]))) for s in WOUND + SYNTH}
        for a in AMPS:
            md[g][f"genB_amp{a}_seedmean"] = float(np.mean([md[g][s] for s in GB[a]]))
        for gen in GENS:
            md[g][f"{gen}_seedmean"] = float(np.mean([md[g][f"{gen}_s{s}"] for s in SEEDS]))
    R["mean_dice"] = md

    # reproduction of V6 (M11 and the unchanged groups) from this run
    V = json.load(open(V6JSON))
    rep = {}
    for p in ("U", "F", "K", "M", "U+F", "U+K", "U+M"):
        rep[f"rel|{p}"] = max_abs(R["rel"][p], V["rel"][p])
        rep[f"texture|{p}"] = max_abs(R["texture"][p], V["texture"][p])
        rep[f"matched|{p}"] = max_abs(R["matched"][p], V["matched"][p])
    rep["rel_hier|M"] = max_abs(R["rel_hier"]["M"], V["rel_hier"]["M"])
    rep["rel_dataonly|M"] = max_abs(R["rel_dataonly"]["M"], V["rel_dataonly"]["M"])
    rep["texture_hier|M"] = max_abs(R["texture_hier"]["M"], V["texture_hier"]["M"])
    rep["lomo|M"] = max_abs(R["lomo"]["M"], V["lomo_M"])
    for g in ("U", "F", "K", "M", "S"):
        rep[f"mean_dice|{g}"] = max_abs(md[g], V["mean_dice"][g])
    # independent route for the headline: scipy.stats.spearmanr on per-method
    # means built straight from per-image data (clinical: perimage_v7; synthetic:
    # score files), without bench_v6's table/zr; U synthetic means from pools_v5
    from scipy.stats import spearmanr
    ind = {}
    for pool in ("M19", "U+M19", "M17", "Ms0"):
        nm = B6.names_of(pool)
        modern = [n for n in nm if n not in B6.GRP["U"]]
        vec = {}
        for r in WOUND:
            vec[r] = np.concatenate([V6.load(r)["X"][:, V6.cols([n for n in nm if n in B6.GRP["U"]])].mean(0)
                                     if pool.startswith("U+") else np.zeros(0),
                                     clin_X(r, modern).mean(0)])
        for sname in GB[40]:
            vec[sname] = np.concatenate([np.array([B6.TT[sname][n] for n in nm if n in B6.GRP["U"]]),
                                         np.array([np.mean([synth_seed(sname, *_parse(n)).mean()]) if _parse(n)
                                                   else (np.mean([synth_seed(sname, n[:-4], k).mean() for k in V7.seeds_of(n[:-4])])
                                                         if n.endswith("@avg") else B6.synth_X(sname, "M")[:, B6.GRP["M"].index(n)].mean())
                                                   for n in modern])])
        bench = np.mean([np.mean([spearmanr(vec[sn], vec[r])[0] for sn in GB[40]]) for r in WOUND])
        other = np.mean([spearmanr(vec[a], vec[b])[0] for a, b in itertools.combinations(WOUND, 2)])
        ind[pool] = {"bench": abs(bench - R["rel"][pool]["40"]["mean|bench"]["est"]),
                     "other": abs(other - R["rel"][pool]["40"]["mean|other"]["est"])}
    R["checks"]["scipy_route_rel40_mean_abs_diff"] = ind
    assert max(max(v.values()) for v in ind.values()) < 1e-12, ind
    R["checks"]["reproduce_v6_max_abs"] = rep
    R["checks"]["reproduce_v6_all_exact"] = all(v == 0.0 for v in rep.values())

    # headline: V6 next to V7, amplitude 40, mean over the 4 wound references
    hl = {}
    for v6p, v7p in (("U", "U"), ("M", "M19"), ("U+M", "U+M19")):
        hl[f"{v6p} -> {v7p}"] = {
            k: {"v6": V["rel"][v6p]["40"][f"mean|{k}"], "v7": R["rel"][v7p]["40"][f"mean|{k}"]}
            for k in ("bench", "other", "diff")}
    hl["M -> M19 hierarchical"] = {
        k: {"v6": V["rel_hier"]["M"]["40"][f"mean|{k}"], "v7": R["rel_hier"]["M19"]["40"][f"mean|{k}"]}
        for k in ("bench", "other", "diff")}
    R["headline_amp40_mean"] = hl

    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, "bench_v7.json"), "w"), indent=1)
    write_md(R)
    print("saved results/v7/bench_v7.json and BENCH_V7.md")


# ------------------------------------------------------------------ report

f3 = B6.f3


def write_md(R):
    P = R["pools"]
    L = ["# Synthetic benchmark analysis (V7, phase-2 modern models)", "",
         "Script: `scripts/v7/bench_v7.py`. Numbers: `results/v7/bench_v7.json`. "
         "Same definitions, draws and random seeds as `scripts/v6/bench_v6.py` (functions imported from it).", "",
         "## Definitions", "",
         "- Agreement: Spearman correlation (average ranks) of per-method mean Dice between two sets, over a method set.",
         "- bench(R): agreement of synthetic generator B (amplitude 40 or 11) with wound reference R, averaged over synthetic seeds 2024, 7, 101.",
         "- other(R): mean agreement of the other three clinical wound datasets with R. Averaged over the 4 references it equals the mean of the 6 wound pairs.",
         "- diff(R) = bench(R) - other(R). Positive means the synthetic benchmark ranks methods more like R than the other clinical wound datasets do. The 'mean' row averages Medetec, CO2Wounds-V2, FUSeg and AZH.",
         "- M19 (primary modern group, " + str(P["M19"]) + " methods): 18 ImageNet-pretrained architectures, each scored per image as the mean over its available training seeds, plus nnU-Net (one training). "
         "Seeds per architecture: 3, except manet_r18 (seeds 0, 1) and manet_r50 (seed 1 only), because 3 of 54 planned trainings diverged.",
         "- Sensitivity sets: M (= M11, V6: 10 run-1 seed-0 architectures + nnU-Net), M17 (M19 without both MAnet architectures, " + str(P["M17"]) + "), "
         "per-seed sets Ms0 (" + str(P["Ms0"]) + "), Ms1 (" + str(P["Ms1"]) + "), Ms2 (" + str(P["Ms2"]) + "): every architecture with that training seed as one model, plus nnU-Net. "
         "Mixtures with U add the 18 untrained methods.",
         "- Synthetic per-image scores of M19 columns are averaged over training seeds per image, the same rule as `perimage_v7.load7` on clinical sets. Each synthetic set has 45 images.",
         "- Method bootstrap: methods resampled with replacement within groups, 2,000 draws, percentile 95% intervals, `default_rng(0)` per method set.",
         "- Hierarchical bootstrap (modern sets only): methods within group plus, independently per dataset, clinical units (Medetec photograph, AZH photograph, otherwise image) and synthetic images; 2,000 draws, `default_rng(1)`. Data-only: methods fixed, `default_rng(2)`.", ""]

    hl = R["headline_amp40_mean"]
    L += ["## Headline: V6 next to V7 (amplitude 40, mean over the 4 wound references)", "",
          "| Set (V6 -> V7) | n methods V6 / V7 | bench V6 | bench V7 | other V6 | other V7 | diff V6 | diff V7 |",
          "|---|---|---|---|---|---|---|---|"]
    nmap = {"U -> U": ("U", "U"), "M -> M19": ("M", "M19"), "U+M -> U+M19": ("U+M", "U+M19"),
            "M -> M19 hierarchical": ("M", "M19")}
    for k, v in hl.items():
        a, b = nmap[k]
        L.append(f"| {k} | {P[a]} / {P[b]} | {f3(v['bench']['v6'])} | {f3(v['bench']['v7'])} | {f3(v['other']['v6'])} | "
                 f"{f3(v['other']['v7'])} | {f3(v['diff']['v6'])} | {f3(v['diff']['v7'])} |")
    L += ["", "U is unchanged (same 18 methods, same draws); its row is a reproduction. Intervals: method bootstrap except the hierarchical row.", ""]

    def rel_rows(pools, amp, key="rel"):
        out = []
        for p in pools:
            d = R[key][p][str(amp)]
            for r in WOUND + ["mean"]:
                out.append(f"| {p} ({P[p]}) | {r} | {f3(d[r + '|bench'])} | {f3(d[r + '|other'])} | {f3(d[r + '|diff'])} |")
        return out

    for amp in (40, 11):
        L += [f"## 1. Modern models only: does the synthetic benchmark agree with each wound dataset more or less than the wound datasets agree with each other? (amplitude {amp})", "",
              "Method bootstrap. Each cell: estimate [95% interval]. Pairs: 3 synthetic seeds per bench value, 3 other wound datasets per other value.", "",
              "| Set (n methods) | Ref | bench | other | diff |", "|---|---|---|---|---|"]
        L += rel_rows(["M19", "U+M19"], amp)
        L += ["", "Hierarchical (methods + units + synthetic images) and data-only intervals:", "",
              "| Set | Ref | bench (hier) | other (hier) | diff (hier) | diff (data-only) |", "|---|---|---|---|---|---|"]
        for g in ("M19",):
            h = R["rel_hier"][g][str(amp)]; d = R["rel_dataonly"][g][str(amp)]
            for r in WOUND + ["mean"]:
                L.append(f"| {g} | {r} | {f3(h[r + '|bench'])} | {f3(h[r + '|other'])} | {f3(h[r + '|diff'])} | {f3(d[r + '|diff'])} |")
        L += ["", "Reading, per set (mean over the 4 references):", ""]
        for p in ("M19", "U+M19"):
            c = R["rel"][p][str(amp)]["mean|diff"]
            refs_pos = [r for r in WOUND if R["rel"][p][str(amp)][r + "|diff"]["lo"] > 0]
            refs_neg = [r for r in WOUND if R["rel"][p][str(amp)][r + "|diff"]["hi"] < 0]
            more = [r for r in WOUND if R["rel"][p][str(amp)][r + "|diff"]["est"] > 0]
            extra = ""
            if p in R["rel_hier"]:
                ch = R["rel_hier"][p][str(amp)]["mean|diff"]
                extra = f" Hierarchical: {f3(ch)}, {'excludes' if B6.excl0(ch) else 'includes'} zero."
            L.append(f"- {p} ({P[p]} methods): diff {f3(c)}; interval {'excludes' if B6.excl0(c) else 'includes'} zero.{extra} "
                     f"Point estimate of diff above zero for {len(more)} of 4 references ({', '.join(more) or 'none'}); "
                     f"interval above zero for {', '.join(refs_pos) or 'none'}; below zero for {', '.join(refs_neg) or 'none'}.")
        L.append("")

    L += ["## 2. All seven primary sets (V6 layout with M replaced by M19)", ""]
    for amp in (40, 11):
        L += [f"### Amplitude {amp}, method bootstrap", "", "| Set (n) | Ref | bench | other | diff |", "|---|---|---|---|---|"]
        L += rel_rows(PRIMARY, amp)
        L.append("")

    L += ["## 3. Sensitivity of the modern result to the definition of the modern group", "",
          "Mean over the 4 wound references. Method bootstrap intervals; hierarchical where available.", "",
          "| Set (n) | amp | bench | other | diff | diff (hierarchical) |", "|---|---|---|---|---|---|"]
    for p in ["M19", "U+M19"] + SENS:
        for amp in ("40", "11"):
            d = R["rel"][p][amp]
            h = f3(R["rel_hier"][p][amp]["mean|diff"]) if p in R["rel_hier"] else "n/a"
            L.append(f"| {p} ({P[p]}) | {amp} | {f3(d['mean|bench'])} | {f3(d['mean|other'])} | {f3(d['mean|diff'])} | {h} |")
    L += ["", "Per-reference diff at amplitude 40 for every modern set (method bootstrap):", "",
          "| Set | " + " | ".join(WOUND) + " |", "|---" * 5 + "|"]
    for p in ["M19", "U+M19"] + SENS:
        L.append(f"| {p} | " + " | ".join(f3(R["rel"][p]["40"][r + "|diff"]) for r in WOUND) + " |")
    L.append("")

    L += ["## 4. Texture effect (amplitude 40 minus 11)", "",
          "Mean over seeds of agreement at amplitude 40 minus the same at 11; method bootstrap.", "",
          "| Set | " + " | ".join(WOUND) + " | mean |", "|---" * 6 + "|"]
    for p in PRIMARY + SENS:
        L.append(f"| {p} | " + " | ".join(f3(R["texture"][p][r]) for r in WOUND + ["mean"]) + " |")
    tc = R["texture_counts"]
    w = tc["within_group_per_dataset"]; a = tc["all_sets_per_dataset"]
    L += ["", f"- Within-group (U, F, K, M19) per-dataset intervals excluding zero: {len(w['exclude_zero'])} of {w['total']} "
          f"({', '.join(w['exclude_zero']) or 'none'}); expected by chance at 95% if every effect were zero: {w['expected_by_chance_if_all_null']:.1f}.",
          f"- All 7 primary sets (mixtures overlap with groups, so not independent): {len(a['exclude_zero'])} of {a['total']}; chance {a['expected_by_chance_if_all_null']:.1f}.",
          f"- 'mean' intervals excluding zero, primary sets: {', '.join(tc['mean_row_exclude_zero']) or 'none'}; sensitivity sets: {', '.join(tc['modern_sensitivity_mean_row_exclude_zero']) or 'none'}.",
          "- Hierarchical texture effect (mean row): " + "; ".join(f"{g} {f3(R['texture_hier'][g]['mean'])}" for g in MODERN) + ".", ""]

    L += ["## 5. Generator families (primary sets)", ""]
    for t, lab in (("M+A", "mean of Medetec and AZH"), ("M+C", "mean of Medetec and CO2Wounds-V2"),
                   ("mean4", "mean of all 4 wound datasets")):
        L += [f"### Agreement with {lab}", "",
              "| Set | seed | nrA | nrB0 | nrC0 | nrC34 | B0 - C34 [95%] |", "|---|---|---|---|---|---|---|"]
        for p in ("M19", "U+M19"):
            m = R["matched"][p]
            for s in list(SEEDS) + ["seedmean"]:
                ss = f"s{s}" if s != "seedmean" else "seedmean"
                L.append(f"| {p} | {s} | " + " | ".join(f"{m[f'{g}_{ss}|{t}']['est']:+.3f}" for g in GENS)
                         + f" | {f3(m[f'B0-C34_{ss}|{t}'])} |")
        o = R["gen_order"][t]
        L += ["", "Seed-mean orderings: " + "; ".join(f"{p}: {o[p]['seedmean']}" for p in PRIMARY) + ".",
              f"Same ordering in all 7 primary sets: {'yes' if o['_stable_seedmean_ordering_across_pools'] else 'no'}.",
              "Pairwise (cells where first beats second, out of 21 set x seed cells; and out of 7 seed means): "
              + "; ".join(f"{k} {v['pool_seed_cells']}, {v['pools_seedmean']}" for k, v in o["_pairwise"].items()) + ".", ""]

    L += ["## 6. Leave-one-method-out, M19 (19 methods, 18 after dropping one)", "",
          "| Dropped | bench40 mean | 6-pair wound | diff40 mean |", "|---|---|---|---|"]
    for n, v in R["lomo"]["M19"].items():
        L.append(f"| {n} | {v['bench40_mean']:+.3f} | {v['pairs6']:+.3f} | {v['rel40_mean']:+.3f} |")
    lv = [v["rel40_mean"] for k, v in R["lomo"]["M19"].items() if k != "_all"]
    big = max((k for k in R["lomo"]["M19"] if k != "_all"), key=lambda k: abs(R["lomo"]["M19"][k]["rel40_mean"] - R["lomo"]["M19"]["_all"]["rel40_mean"]))
    L += ["", f"Range of diff40 mean over the 19 leave-one-out sets: {min(lv):+.3f} to {max(lv):+.3f}. "
          f"The largest single shift comes from dropping {big} ({R['lomo']['M19']['_all']['rel40_mean']:+.3f} to {R['lomo']['M19'][big]['rel40_mean']:+.3f}); "
          "its synthetic score also varies most across training seeds (PRACTICAL_V7.md, section 5e).", ""]

    L += ["## 7. Mean Dice by group", "", "| Set | U | F | K | M | M19 | Ms0 | Ms1 | Ms2 |", "|---|---|---|---|---|---|---|---|---|"]
    md = R["mean_dice"]
    rows = WOUND + [f"genB_amp{a}_seedmean" for a in AMPS] + [f"{g}_seedmean" for g in GENS]
    for s in rows:
        L.append(f"| {s} | " + " | ".join(f"{md[g][s]:.3f}" for g in ("U", "F", "K", "M", "M19", "Ms0", "Ms1", "Ms2")) + " |")

    C = R["checks"]
    rep = C["reproduce_v6_max_abs"]
    L += ["", "## Checks", "",
          f"- Synthetic helper, seed 0 of the 10 run-1 architectures vs the V6 per-image columns (21 sets x 45 images): max abs difference {C['synth_seed0_vs_v6_perimage_max_abs']:.1e}; their means vs pools_v5: {C['synth_seed0_mean_vs_pools_v5_max_abs']:.1e}.",
          f"- Clinical seed 0 vs V6 columns (4 wound sets): {C['clinical_seed0_vs_v6_max_abs']:.1e}.",
          f"- Mean of the seed-averaged per-image vector vs mean of per-seed means (19 M19 columns, 25 sets): {C['avg_perimage_mean_vs_mean_of_seed_means_max_abs']:.1e}.",
          f"- perimage_v7 seed-averaged columns vs an independent average of load_seed columns: {C['load7_avg_vs_load_seed_average_max_abs']:.1e}.",
          f"- Synthetic reference order (gt_fraction and gt_empty) identical across all 6 score folders for all 21 sets: {'yes' if not C['synthetic_gt_order_mismatches'] else 'no: ' + ', '.join(C['synthetic_gt_order_mismatches'])}.",
          f"- Architecture-seed models found: {C['n_arch_seed_models']} (54 planned, 3 diverged).",
          f"- Reproduction of V6 from this run (U, F, K, M, U+F, U+K, U+M method bootstrap; M hierarchical, data-only, texture, leave-one-out; mean Dice): "
          f"all exactly equal: {'yes' if C['reproduce_v6_all_exact'] else 'no'}; largest difference {max(rep.values()):.1e}.",
          "- Independent route (scipy.stats.spearmanr on means built from per-image files, no bench_v6 table or rank code), amplitude 40 mean bench and other: max abs difference "
          + ", ".join(f"{k} {max(v.values()):.0e}" for k, v in C["scipy_route_rel40_mean_abs_diff"].items()) + ".",
          "- Per-image loader means vs table means, modern sets: " + ", ".join(f"{k} {v:.1e}" for k, v in R["loader_max_abs_diff"].items()) + ".",
          "", "## Limitations", "",
          "- nnU-Net has one training, so in M19 it is the only member whose score is not a seed average; its score carries more training noise than the others.",
          "- manet_r50 rests on one seed and manet_r18 on two; M17 drops both. The diverged trainings are missing, not imputed.",
          "- Seed averaging smooths each architecture's score, so M19 rankings are less noisy than any single-seed ranking; per-seed sets Ms0 to Ms2 show the single-training picture.",
          "- U and F have only per-method mean Dice on synthetic sets, so no hierarchical bootstrap for U, F or the mixtures.",
          "- The four reference-level diffs share data (each wound pair enters two 'other' terms), so per-reference intervals are not independent.",
          "- In mixtures (U+M19 etc.) a Spearman over pooled methods mixes within-group ordering with between-group difficulty offsets.",
          "- Floor effect on nrA: modern models score near zero there, so their nrA agreements carry little information.", ""]
    open(os.path.join(OUT, "BENCH_V7.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V7 synthetic-benchmark analysis (port of scripts/v6/bench_v6.py to perimage_v7).

Definitions (unchanged from V6)
  agreement(A, B | P)  Spearman correlation (average ranks for ties) between the
                       per-method mean Dice on set A and on set B, over the
                       methods of method set P.
  bench_a(R)           mean over synthetic seeds 2024, 7, 101 of
                       agreement(genB_amp{a}_s{seed}, R).
  other(R)             mean over the three other wound datasets R' of
                       agreement(R', R).
  diff_a(R)            bench_a(R) - other(R).
  "mean" row           average over the four wound references.

Data sources
  U, F                 per-method mean Dice from pools_v5 (no per-image
                       synthetic scores exist for them), exactly as V6.
  K, M and every new   per-image Dice from perimage_v7.load (clinical, V6
  modern group         exclusions) and perimage_v7.load_synth (synthetic, 45
                       images per set); method mean = mean over images.
                       Seed-averaged columns '<base>_in@seeds' and
                       '<base>@seeds' are per-image means over the available
                       training seeds (MAnet: manet_r18_in seeds 0,1;
                       manet_r50 seed 1 only).
  S                    pools_v5 means (SAM/MedSAM, only in the mean Dice table).

Method sets
  V6 reproduction      U, F, K, M, U+F, U+K, U+M (V6 groups, V6 code path).
  V7                   M11s (P10s + nnU-Net), M19 (P10s + N8s + nnU-Net),
                       N8s, P10s, U+M11s, U+M19, U+F+K+M (V6 groups, no V6
                       bench counterpart), U+F+K+M19.
  Seed groups          P10_s0..s2, N8_s0..s2 (single training seed, only
                       models that exist) and M19_s0..s2 = P10_sk + N8_sk +
                       nnU-Net (one group; nnU-Net has a single training).

Uncertainty (unchanged from V6)
  method bootstrap     methods resampled with replacement within each group
                       (mixtures keep their composition), B = 2,000 draws,
                       percentile 95% intervals, numpy default_rng(0) created
                       afresh for every method set.
  hierarchical         one group at a time (M, K, M11s, M19, N8s, P10s):
                       methods within the group AND, independently per
                       dataset, clinical units (Medetec photograph, AZH
                       photograph, image otherwise) and synthetic images;
                       default_rng(1), 2,000 draws.
  data-only            as hierarchical with methods fixed; default_rng(2).

Output: results/v7/bench_v7.json. results/v7/BENCH_V7.md is written from it by
scripts/v7/write_md_v7.py.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import perimage_v7 as V7                                           # noqa: E402
import pools_v5 as P                                               # noqa: E402

B = int(os.environ.get("B", "2000"))
SEEDS = (2024, 7, 101)
AMPS = (11, 28, 40)
GENS = ("nrA", "nrB0", "nrC0", "nrC34")
WOUND = list(V7.WOUND)
assert WOUND == ["medetec", "co2wounds", "fuseg", "azh"]
GB = {a: [f"genB_amp{a}_s{s}" for s in SEEDS] for a in AMPS}
NR = [f"{g}_s{s}" for g in GENS for s in SEEDS]
SYNTH = sum((GB[a] for a in AMPS), []) + NR
assert sorted(SYNTH) == sorted(V7.SYNTH)
MAN = json.load(open("/home/claude/kaggle_kit/manifest.json"))
TT = P.load()["dice"]
OUT = os.path.join(ROOT, "results", "v7")
V6JSON = os.path.join(ROOT, "results", "v6", "bench_v6.json")
OLDV7 = os.path.join(ROOT, "results", "v7_morning", "bench_v7.json")

GRP = {g: list(v) for g, v in V7.GROUPS.items()}
for k in (0, 1, 2):
    GRP[f"M19_s{k}"] = GRP[f"P10_s{k}"] + GRP[f"N8_s{k}"] + ["nnunet_2d"]
TT_ONLY = ("U", "F", "S")            # always from pools_v5

V6POOLS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
           "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"]}
NEWPOOLS = {"M11s": ["M11s"], "M19": ["M19"], "N8s": ["N8s"], "P10s": ["P10s"],
            "U+M11s": ["U", "M11s"], "U+M19": ["U", "M19"],
            "U+F+K+M": ["U", "F", "K", "M"], "U+F+K+M19": ["U", "F", "K", "M19"]}
SEEDPOOLS = {g: [g] for g in ("P10_s0", "P10_s1", "P10_s2", "N8_s0", "N8_s1", "N8_s2",
                              "M19_s0", "M19_s1", "M19_s2")}
POOLS = {**V6POOLS, **NEWPOOLS, **SEEDPOOLS}
HIER6 = ("M", "K")
HIER7 = ("M11s", "M19", "N8s", "P10s")
V7FAMILY = ["U", "F", "K", "M19", "U+F", "U+K", "U+M19"]   # V6 family with M -> M19

# ------------------------------------------------------------ data access
CLIN = {r: V7.load(r, "dice") for r in WOUND}
SYN = {s: V7.load_synth(s, "dice") for s in SYNTH}


def X_of(s, names):
    if s in CLIN:
        return CLIN[s]["X"][:, V7.cols(names)]
    return SYN[s]["X"][:, V7.cols(names, SYN[s]["names"])]


def gvec(s, g, src):
    """Per-method mean Dice of group g on set s. src='tt' takes K and M from
    pools_v5 (the V6 code path); src='v7' from the perimage_v7 per-image data."""
    if g in TT_ONLY or (src == "tt" and g in ("K", "M")):
        return P.vec(TT, s, GRP[g])
    return X_of(s, GRP[g]).mean(0)


def names_of(pool):
    return sum((GRP[g] for g in POOLS[pool]), [])


def table(sets, pool, src="v7"):
    return {s: np.concatenate([gvec(s, g, src) for g in POOLS[pool]]) for s in sets}


# ------------------------------------------------------------ statistics (copied from bench_v6)

def sp(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def zr(M):
    keys = list(M)
    R = rankdata(np.vstack([M[k] for k in keys]), axis=1)
    R = R - R.mean(1, keepdims=True)
    nrm = np.sqrt((R ** 2).sum(1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(nrm > 0, R / np.where(nrm > 0, nrm, 1), np.nan)
    return dict(zip(keys, Z))


def c(Z, a, b):
    return float(Z[a] @ Z[b])


def f_rel(amp):
    def fn(M):
        out = {}
        for r in WOUND:
            b = float(np.mean([c(M, s, r) for s in GB[amp]]))
            o = float(np.mean([c(M, x, r) for x in WOUND if x != r]))
            out[f"{r}|bench"], out[f"{r}|other"], out[f"{r}|diff"] = b, o, b - o
        for k in ("bench", "other", "diff"):
            out[f"mean|{k}"] = float(np.mean([out[f"{r}|{k}"] for r in WOUND]))
        out["pairs6"] = float(np.mean([c(M, a, b) for a, b in itertools.combinations(WOUND, 2)]))
        return out
    return fn


def f_bench_seed(amp):
    """Extra in V7: agreement of each single synthetic seed with each reference
    (mean over the 4 references), so the spread over synthetic seeds is visible."""
    def fn(M):
        out = {}
        for s in GB[amp]:
            for r in WOUND:
                out[f"{s}|{r}"] = c(M, s, r)
            out[f"{s}|mean"] = float(np.mean([c(M, s, r) for r in WOUND]))
        return out
    return fn


def f_texture(M):
    out = {}
    for r in WOUND + ["mean"]:
        rs = WOUND if r == "mean" else [r]
        ag = {a: np.mean([c(M, s, x) for s in GB[a] for x in rs]) for a in (11, 40)}
        out[r] = float(ag[40] - ag[11])
    return out


def f_matched(M):
    out = {}
    tg = {"M+A": ("medetec", "azh"), "M+C": ("medetec", "co2wounds"), "mean4": WOUND}
    for s in SEEDS:
        for g in GENS:
            for t, rs in tg.items():
                out[f"{g}_s{s}|{t}"] = float(np.mean([c(M, f"{g}_s{s}", r) for r in rs]))
        for t in tg:
            out[f"B0-C34_s{s}|{t}"] = out[f"nrB0_s{s}|{t}"] - out[f"nrC34_s{s}|{t}"]
    for t in tg:
        out[f"B0-C34_seedmean|{t}"] = float(np.mean([out[f"B0-C34_s{s}|{t}"] for s in SEEDS]))
        for g in GENS:
            out[f"{g}_seedmean|{t}"] = float(np.mean([out[f"{g}_s{s}|{t}"] for s in SEEDS]))
    return out


def combine(*fns):
    def fn(M):
        o = {}
        for i, f in enumerate(fns):
            o.update({f"{i}:{k}": v for k, v in f(M).items()})
        return o
    return fn


def ci(obs, draws):
    out = {}
    for k in obs:
        d = np.asarray(draws[k], float)
        out[k] = {"est": obs[k], "lo": float(np.nanpercentile(d, 2.5)),
                  "hi": float(np.nanpercentile(d, 97.5)),
                  "n_nan": int(np.isnan(d).sum())}
    return out


def method_takes(pool, seed=0):
    nm = names_of(pool)
    assert len(nm) == len(set(nm)), pool
    idx = [np.array([nm.index(n) for n in GRP[g]]) for g in POOLS[pool]]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(i, len(i)) for i in idx]) for _ in range(B)]


def mboot(pool, sets, fn, takes, src="v7"):
    full = table(sets, pool, src)
    obs = fn(zr(full))
    draws = {k: [] for k in obs}
    for t in takes:
        r = fn(zr({s: v[t] for s, v in full.items()}))
        for k in obs:
            draws[k].append(r[k])
    return ci(obs, draws)


def unit_struct(X, units):
    u = sorted(set(units))
    pos = {k: i for i, k in enumerate(u)}
    ix = np.array([pos[k] for k in units])
    S = np.zeros((len(u), X.shape[1])); n = np.zeros(len(u))
    np.add.at(S, ix, X); np.add.at(n, ix, 1)
    return S, n


def per_image(g, sets):
    out, checks = {}, []
    for s in sets:
        X = X_of(s, GRP[g])
        units = list(CLIN[s]["units"]) if s in CLIN else list(range(len(X)))
        S, n = unit_struct(X, units)
        if g in ("K", "M"):
            checks.append(float(np.abs(X.mean(0) - P.vec(TT, s, GRP[g])).max()))
        out[s] = (S, n, X.shape[0])
    return out, (max(checks) if checks else None)


def hboot(g, sets, fn, resample_methods=True, seed=1):
    data, _ = per_image(g, sets)
    full = {s: S.sum(0) / n.sum() for s, (S, n, _) in data.items()}
    obs = fn(zr(full))
    rng = np.random.default_rng(seed)
    k = len(GRP[g])
    draws = {kk: [] for kk in obs}
    for _ in range(B):
        t = rng.choice(k, k) if resample_methods else np.arange(k)
        M = {}
        for s, (S, n, _) in data.items():
            u = rng.integers(0, len(n), len(n))
            M[s] = (S[u].sum(0) / n[u].sum())[t]
        r = fn(zr(M))
        for kk in obs:
            draws[kk].append(r[kk])
    return ci(obs, draws)


def split(res, prefix):
    return {k[len(prefix):]: v for k, v in res.items() if k.startswith(prefix)}


def excl0(cc):
    return cc["lo"] > 0 or cc["hi"] < 0


# ------------------------------------------------------------ blocks

def run_mboot(pools, src):
    rel, tex, mat, bseed = {}, {}, {}, {}
    sets = WOUND + SYNTH
    for pool in pools:
        takes = method_takes(pool, 0)
        res = mboot(pool, sets, combine(f_rel(40), f_rel(11), f_texture, f_matched,
                                        f_bench_seed(40)), takes, src)
        rel[pool] = {"40": split(res, "0:"), "11": split(res, "1:")}
        tex[pool] = split(res, "2:")
        mat[pool] = split(res, "3:")
        bseed[pool] = split(res, "4:")
        print(pool, f"method bootstrap done ({src})", flush=True)
    return rel, tex, mat, bseed


def run_hier(groups):
    rh, rd, th = {}, {}, {}
    hsets = WOUND + GB[40] + GB[11]
    for g in groups:
        fn = combine(f_rel(40), f_rel(11), f_texture)
        h = hboot(g, hsets, fn, True, 1)
        d = hboot(g, hsets, fn, False, 2)
        rh[g] = {"40": split(h, "0:"), "11": split(h, "1:")}
        th[g] = split(h, "2:")
        rd[g] = {"40": split(d, "0:"), "11": split(d, "1:")}
        print(g, "hierarchical done", flush=True)
    return rh, rd, th


def texture_counts(tex, single, allp):
    within = [(p, r) for p in single for r in WOUND]
    al = [(p, r) for p in allp for r in WOUND]
    return {
        "within_group_per_dataset": {
            "groups": list(single), "total": len(within),
            "exclude_zero": [f"{p}|{r}" for p, r in within if excl0(tex[p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(within)},
        "all_sets_per_dataset": {
            "sets": list(allp), "total": len(al),
            "exclude_zero": [f"{p}|{r}" for p, r in al if excl0(tex[p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(al)},
        "mean_row_exclude_zero": [p for p in allp if excl0(tex[p]["mean"])]}


def gen_order(matched, pools):
    order = {}
    for t in ("M+A", "M+C", "mean4"):
        order[t] = {}
        for pool in pools:
            m = matched[pool]
            o = {"seedmean": sorted(GENS, key=lambda g: -m[f"{g}_seedmean|{t}"]["est"])}
            for s in SEEDS:
                o[f"s{s}"] = sorted(GENS, key=lambda g: -m[f"{g}_s{s}|{t}"]["est"])
            order[t][pool] = {k: ">".join(v) for k, v in o.items()}
        sm = {order[t][p]["seedmean"] for p in pools}
        pair = {}
        for a, b in itertools.combinations(GENS, 2):
            wins = sum(matched[p][f"{a}_s{s}|{t}"]["est"] > matched[p][f"{b}_s{s}|{t}"]["est"]
                       for p in pools for s in SEEDS)
            wins_sm = sum(matched[p][f"{a}_seedmean|{t}"]["est"] > matched[p][f"{b}_seedmean|{t}"]["est"]
                          for p in pools)
            pair[f"{a}>{b}"] = {"pool_seed_cells": f"{wins}/{len(pools) * len(SEEDS)}",
                                "pools_seedmean": f"{wins_sm}/{len(pools)}"}
        order[t]["_stable_seedmean_ordering_across_pools"] = len(sm) == 1
        order[t]["_distinct_seedmean_orderings"] = sorted(sm)
        order[t]["_pairwise"] = pair
    return order


def lomo(pool, src):
    full = table(WOUND + GB[40] + GB[11], pool, src)
    base = f_rel(40)(zr(full))
    out = {"_all": {"bench40_mean": base["mean|bench"], "pairs6": base["pairs6"],
                    "rel40_mean": base["mean|diff"]}}
    nm = names_of(pool)
    for i, n in enumerate(nm):
        keep = np.array([j for j in range(len(nm)) if j != i])
        r = f_rel(40)(zr({s: v[keep] for s, v in full.items()}))
        out[n] = {"bench40_mean": r["mean|bench"], "pairs6": r["pairs6"], "rel40_mean": r["mean|diff"]}
    return out


def mean_dice(groups, src):
    md = {}
    for g in groups:
        md[g] = {s: float(np.mean(gvec(s, g, src))) for s in WOUND + SYNTH}
        for a in AMPS:
            md[g][f"genB_amp{a}_seedmean"] = float(np.mean([md[g][s] for s in GB[a]]))
        for gen in GENS:
            md[g][f"{gen}_seedmean"] = float(np.mean([md[g][f"{gen}_s{s}"] for s in SEEDS]))
    return md


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
    """Leaf-by-leaf comparison of two JSON trees. Returns counts and max abs
    difference over numeric leaves; non-numeric leaves must be equal."""
    la, lb = dict(leaves(a)), dict(leaves(b))
    missing = sorted(set(lb) - set(la)); extra = sorted(set(la) - set(lb))
    nnum = nexact = 0; mx = 0.0; bad = []
    for k in set(la) & set(lb):
        x, y = la[k], lb[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y:
                bad.append(k)
            continue
        nnum += 1
        if (x != x and y != y) or x == y:
            nexact += 1
        else:
            mx = max(mx, abs(x - y)); bad.append(k)
    return {"n_leaves_compared": len(set(la) & set(lb)), "n_numeric": nnum,
            "n_numeric_exact": nexact, "max_abs_diff": mx, "mismatched": sorted(bad)[:50],
            "n_mismatched": len(bad), "missing_in_rerun": missing[:50], "n_missing": len(missing),
            "extra_in_rerun": len(extra)}


# ------------------------------------------------------------------ main

def v6_block(src):
    """Exactly the V6 bench computations for the V6 groups."""
    R = {"B": B, "pools": {p: len(names_of(p)) for p in V6POOLS}}
    rel, tex, mat, _ = run_mboot(V6POOLS, src)
    R["rel"], R["texture"], R["matched"] = rel, tex, mat
    rh, rd, th = run_hier(HIER6) if src == "v7" else (None, None, None)
    if rh is not None:
        R["rel_hier"], R["rel_dataonly"], R["texture_hier"] = rh, rd, th
    R["texture_counts"] = texture_counts(tex, ("U", "F", "K", "M"), list(V6POOLS))
    # V6 stored these without the group/set lists
    for k in ("within_group_per_dataset",):
        R["texture_counts"][k].pop("groups")
    R["texture_counts"]["all_sets_per_dataset"].pop("sets")
    R["nr_texture"] = {k: MAN["synth"][k]["texture"] for k in NR}
    R["genB_texture"] = {k: MAN["synth"][k]["texture"] for a in AMPS for k in GB[a]}
    R["gen_order"] = gen_order(mat, list(V6POOLS))
    R["lomo_M"] = lomo("M", src)
    R["mean_dice"] = mean_dice(("U", "F", "K", "M", "S"), src)
    return json.loads(json.dumps(R))


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"definitions": __doc__, "B": B,
         "groups": {g: GRP[g] for g in GRP if g not in ("S2o", "S2n")},
         "seedcols": V7.SEEDCOLS}
    V6 = json.load(open(V6JSON))

    # ---------------- checks on data
    chk = {}
    for g in ("K", "M"):
        chk[f"{g}_v7_per_image_means_vs_pools_v5"] = max(
            float(np.abs(X_of(s, GRP[g]).mean(0) - P.vec(TT, s, GRP[g])).max()) for s in WOUND + SYNTH)
    # seed-averaged synthetic columns vs an independent route: mean over seeds of
    # each seed model's mean Dice read straight from the score files
    d_sc = 0.0
    for s in SYNTH:
        raw = {}
        for f in V7.SYNTH_FOLDERS:
            j = json.load(open(os.path.join(*V7.SRC[f], s + ".json")))
            for k, n in enumerate(j["names"]):
                raw[n] = float(np.mean(np.asarray(j["dice"], float)[:, k]))
        for col, mem in V7.SEEDCOLS.items():
            d_sc = max(d_sc, abs(np.mean([raw[m] for m in mem]) - X_of(s, [col]).mean()))
    chk["seedcol_synth_means_vs_raw_files"] = d_sc
    # nnU-Net synthetic column vs pools_v5
    chk["n_images_per_synth_set"] = sorted({int(SYN[s]["X"].shape[0]) for s in SYNTH})
    R["checks"] = chk

    # ---------------- 1. V6 reproduction, two routes
    rep = {}
    keys = ["B", "pools", "rel", "texture", "matched", "texture_counts", "nr_texture",
            "genB_texture", "gen_order", "lomo_M", "mean_dice"]
    for src in ("tt", "v7"):
        R6 = v6_block(src)
        kk = keys + (["rel_hier", "rel_dataonly", "texture_hier"] if src == "v7" else [])
        rep[src] = compare({k: R6[k] for k in kk}, {k: V6[k] for k in kk})
        if src == "v7":
            R6v7 = R6
    R["v6_reproduction"] = {
        "tt": dict(rep["tt"], route="K and M means from pools_v5 (V6 code path)"),
        "v7": dict(rep["v7"], route="K and M means from perimage_v7 per-image data; "
                                    "includes hierarchical and data-only bootstraps")}
    # 'tt' route: bit-identical.  'v7' route: every rank-based value (all
    # agreements, intervals, orderings, hierarchical and data-only bootstraps)
    # bit-identical; only plain group mean Dice may differ in the last bit,
    # because per-method means are summed in a different order (per-image
    # columns of perimage_v7 vs pools_v5 means).
    assert rep["tt"]["n_mismatched"] == 0 and rep["tt"]["n_missing"] == 0, rep["tt"]
    assert rep["v7"]["n_missing"] == 0, rep["v7"]
    assert all(k.startswith("/mean_dice/") for k in rep["v7"]["mismatched"]), rep["v7"]
    assert rep["v7"]["n_mismatched"] <= 50 and rep["v7"]["max_abs_diff"] < 1e-15, rep["v7"]
    print("V6 reproduction exact:", {k: (v["n_numeric_exact"], v["n_numeric"]) for k, v in rep.items()})
    R["v6"] = R6v7

    # ---------------- 2. new method sets
    newp = {**NEWPOOLS, **SEEDPOOLS}
    R["pools"] = {p: len(names_of(p)) for p in POOLS}
    rel, tex, mat, bseed = run_mboot(newp, "v7")
    # U, F, K reused from the V6 block (same data, same RNG): copy for the V7 family
    for p in ("U", "F", "K", "U+F", "U+K", "M", "U+M"):
        rel[p], tex[p], mat[p] = R6v7["rel"][p], R6v7["texture"][p], R6v7["matched"][p]
    _, _, _, bs6 = run_mboot({p: V6POOLS[p] for p in ("M", "U+M")}, "v7")
    bseed.update(bs6)
    R["rel"], R["texture"], R["matched"], R["bench40_per_synth_seed"] = rel, tex, mat, bseed
    rh, rd, th = run_hier(HIER7)
    R["rel_hier"], R["rel_dataonly"], R["texture_hier"] = rh, rd, th
    R["texture_counts_v7family"] = texture_counts(tex, ("U", "F", "K", "M19"), V7FAMILY)
    R["gen_order_v7family"] = gen_order(mat, V7FAMILY)
    R["lomo"] = {p: lomo(p, "v7") for p in ("M19", "M11s")}
    R["mean_dice"] = mean_dice(("U", "F", "K", "M", "S", "M11s", "M19", "N8s", "P10s",
                                "P10_s0", "P10_s1", "P10_s2", "N8_s0", "N8_s1", "N8_s2",
                                "M19_s0", "M19_s1", "M19_s2"), "v7")

    # ---------------- 5. seed effect summary (amp 40)
    se = {}
    for p in ("P10s", "P10_s0", "P10_s1", "P10_s2", "N8s", "N8_s0", "N8_s1", "N8_s2",
              "M19", "M19_s0", "M19_s1", "M19_s2", "M", "M11s"):
        d = rel[p]["40"]
        se[p] = {"n_methods": len(names_of(p)), "bench": d["mean|bench"], "other": d["mean|other"],
                 "diff": d["mean|diff"], "pairs6": d["pairs6"],
                 "bench_by_ref": {r: d[f"{r}|bench"]["est"] for r in WOUND}}
    for fam, members in (("P10", ["P10_s0", "P10_s1", "P10_s2"]), ("N8", ["N8_s0", "N8_s1", "N8_s2"]),
                         ("M19", ["M19_s0", "M19_s1", "M19_s2"])):
        v = [se[m]["bench"]["est"] for m in members]
        se[f"_{fam}_single_seed_bench_range"] = [min(v), max(v)]
        se[f"_{fam}_single_seed_bench_mean"] = float(np.mean(v))
    R["seed_effect_amp40"] = se

    # ---------------- cross-check against the earlier V7 implementation
    xc = {}
    if os.path.exists(OLDV7):
        O = json.load(open(OLDV7))
        for mine, theirs in (("M19", "M19"), ("U+M19", "U+M19"), ("M19_s0", "Ms0"),
                             ("M19_s1", "Ms1"), ("M19_s2", "Ms2")):
            if theirs in O["rel"]:
                cmp = compare({"40": rel[mine]["40"], "11": rel[mine]["11"]},
                              {"40": O["rel"][theirs]["40"], "11": O["rel"][theirs]["11"]})
                xc[f"{mine}~old:{theirs}"] = {k: cmp[k] for k in ("n_numeric", "n_numeric_exact", "max_abs_diff")}
        if "M19" in O.get("rel_hier", {}):
            cmp = compare(rh["M19"], O["rel_hier"]["M19"])
            xc["hier M19~old:M19"] = {k: cmp[k] for k in ("n_numeric", "n_numeric_exact", "max_abs_diff")}
    R["crosscheck_old_v7_implementation"] = xc

    json.dump(R, open(os.path.join(OUT, "bench_v7.json"), "w"), indent=1)
    print("saved", os.path.join(OUT, "bench_v7.json"))


if __name__ == "__main__":
    main()

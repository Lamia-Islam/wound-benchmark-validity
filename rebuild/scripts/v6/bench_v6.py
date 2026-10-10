#!/usr/bin/env python3
"""V6 reanalysis of the synthetic benchmarks.

Definitions
  agreement(A, B | P)  Spearman correlation (average ranks for ties) between the
                       per-method mean Dice on set A and on set B, over the
                       methods of method set P.
  Method sets          U (18), F (14), K (14), M (11), U+F, U+K, U+M.
  bench_a(R)           mean over seeds 2024, 7, 101 of
                       agreement(genB_amp{a}_s{seed}, R).
  other(R)             mean over the three other wound datasets R' of
                       agreement(R', R).
  rel_a(R)             bench_a(R) - other(R).  Positive: the benchmark ranks
                       methods more like R than the other clinical wound
                       datasets do.
  "mean" row           average over the four wound references; mean(other)
                       equals the mean of the six wound pair correlations.

Uncertainty
  method bootstrap     methods resampled with replacement within each group
                       (mixed sets keep their composition), 2,000 draws,
                       percentile 95% intervals; RNG seeded exactly as in
                       stats_v5.boot (default_rng(0), same draw order), so
                       the stats_v5 intervals are reproduced draw for draw.
  hierarchical         (M and K only; U and F have no synthetic per-image
                       scores) each draw resamples methods within the group
                       AND, independently per dataset, clinical units
                       (Medetec photograph, AZH photograph, image otherwise;
                       see perimage_v6.unit_of) and synthetic images (each
                       synthetic image is one unit); per-method means are
                       recomputed as image-weighted means over the resampled
                       images.  default_rng(1), 2,000 draws.
  data-only            as hierarchical but methods fixed (reported for
                       context only).

Output: results/v6/bench_v6.json and results/v6/BENCH_V6.md
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage_v6 as V                                            # noqa: E402
import pools_v5 as P                                               # noqa: E402

B = int(os.environ.get("B", "2000"))
SEEDS = (2024, 7, 101)
AMPS = (11, 28, 40)
GENS = ("nrA", "nrB0", "nrC0", "nrC34")
WOUND = list(V.WOUND)
assert WOUND == ["medetec", "co2wounds", "fuseg", "azh"]
GRP = {g: list(V.GROUPS[g]) for g in ("U", "F", "K", "M", "S")}
POOLS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
         "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"]}
GB = {a: [f"genB_amp{a}_s{s}" for s in SEEDS] for a in AMPS}
NR = [f"{g}_s{s}" for g in GENS for s in SEEDS]
SYNTH = sum((GB[a] for a in AMPS), []) + NR
MAN = json.load(open("/home/claude/kaggle_kit/manifest.json"))
TT = P.load()["dice"]
OUT = os.path.join(ROOT, "results", "v6")
os.makedirs(OUT, exist_ok=True)


def sp(a, b):
    """Spearman, identical to stats_v5.sp."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def zr(M):
    """Standardised average ranks of every set at once; c(Z, a, b) then equals
    sp(M[a], M[b]) up to rounding (NaN when a vector is constant)."""
    keys = list(M)
    R = rankdata(np.vstack([M[k] for k in keys]), axis=1)
    R = R - R.mean(1, keepdims=True)
    nrm = np.sqrt((R ** 2).sum(1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(nrm > 0, R / np.where(nrm > 0, nrm, 1), np.nan)
    return dict(zip(keys, Z))


def c(Z, a, b):
    return float(Z[a] @ Z[b])


def names_of(pool):
    return sum((GRP[g] for g in POOLS[pool]), [])


def table(sets, pool):
    nm = names_of(pool)
    return {s: P.vec(TT, s, nm) for s in sets}


# ------------------------------------------------------------ statistics

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


def f_texture(M):
    """stats_v5.f_texture, unchanged."""
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
    idx = [np.array([nm.index(n) for n in GRP[g]]) for g in POOLS[pool]]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rng.choice(i, len(i)) for i in idx]) for _ in range(B)]


def mboot(pool, sets, fn, takes):
    full = table(sets, pool)
    obs = fn(zr(full))
    draws = {k: [] for k in obs}
    for t in takes:
        r = fn(zr({s: v[t] for s, v in full.items()}))
        for k in obs:
            draws[k].append(r[k])
    return ci(obs, draws)


# ------------------------------------------------------ per-image data

def synth_X(s, g):
    dirs = V.KAGGLE[g]
    mats, names = [], []
    for d in dirs:
        j = json.load(open(os.path.join(V.RES, "kaggle", "scores", d, s + ".json")))
        assert j["set"] == s and j["n"] == len(j["dice"])
        mats.append(np.asarray(j["dice"], float)); names += j["names"]
    X = np.concatenate(mats, 1)
    assert s not in V.EXCL
    return X[:, [names.index(n) for n in GRP[g]]]


def unit_struct(X, units):
    """Per-unit sums and image counts."""
    u = sorted(set(units))
    pos = {k: i for i, k in enumerate(u)}
    ix = np.array([pos[k] for k in units])
    S = np.zeros((len(u), X.shape[1])); n = np.zeros(len(u))
    np.add.at(S, ix, X); np.add.at(n, ix, 1)
    return S, n


def per_image(g, sets):
    """{set: (unit_sums, unit_counts)} for group g (K or M)."""
    out, checks = {}, []
    for s in sets:
        if s in WOUND:
            D = V.load(s)
            X = D["X"][:, V.cols(GRP[g])]
            S, n = unit_struct(X, list(D["units"]))
        else:
            X = synth_X(s, g)
            S, n = unit_struct(X, list(range(len(X))))
        checks.append(float(np.abs(X.mean(0) - P.vec(TT, s, GRP[g])).max()))
        out[s] = (S, n, X.shape[0])
    return out, max(checks)


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


def excl0(c):
    return c["lo"] > 0 or c["hi"] < 0


# ------------------------------------------------------------------ main

def main():
    R = {"definitions": __doc__, "B": B, "pools": {p: len(names_of(p)) for p in POOLS}}

    # consistency of the two loaders
    chk = {}
    for g in ("K", "M", "S"):
        sets = WOUND + SYNTH if g != "S" else SYNTH
        if g == "S":
            chk[g] = max(float(np.abs(synth_X(s, g).mean(0) - P.vec(TT, s, GRP[g])).max()) for s in sets)
        else:
            chk[g] = per_image(g, sets)[1]
    for g in ("U", "F"):
        chk[g] = max(float(np.abs(V.load(r)["X"][:, V.cols(GRP[g])].mean(0) - P.vec(TT, r, GRP[g])).max())
                     for r in WOUND)
    R["loader_max_abs_diff"] = chk
    # texture of nr sets in nonreduce32_cells vs manifest (seed 2024)
    tx = {}
    for line in open(os.path.join(V.RES, "nonreduce32_cells.jsonl")):
        c = json.loads(line)
        tx[c["name"]] = c["texture"]
    R["nr_texture_crosscheck_s2024"] = tx

    # 1-3: method bootstrap, one set of draws per pool (as stats_v5)
    R["rel"], R["texture"], R["matched"] = {}, {}, {}
    sets = WOUND + SYNTH
    for pool in POOLS:
        takes = method_takes(pool, 0)
        res = mboot(pool, sets, combine(f_rel(40), f_rel(11), f_texture, f_matched), takes)
        R["rel"][pool] = {"40": split(res, "0:"), "11": split(res, "1:")}
        R["texture"][pool] = split(res, "2:")
        R["matched"][pool] = split(res, "3:")
        print(pool, "method bootstrap done", flush=True)

    # 1b: hierarchical and data-only for M and K
    R["rel_hier"], R["rel_dataonly"], R["texture_hier"] = {}, {}, {}
    hsets = WOUND + GB[40] + GB[11]
    for g in ("M", "K"):
        fn = combine(f_rel(40), f_rel(11), f_texture)
        h = hboot(g, hsets, fn, True, 1)
        d = hboot(g, hsets, fn, False, 2)
        R["rel_hier"][g] = {"40": split(h, "0:"), "11": split(h, "1:")}
        R["texture_hier"][g] = split(h, "2:")
        R["rel_dataonly"][g] = {"40": split(d, "0:"), "11": split(d, "1:")}
        print(g, "hierarchical done", flush=True)

    # 2: texture effect, interval counts
    within = [(p, r) for p in ("U", "F", "K", "M") for r in WOUND]
    allp = [(p, r) for p in POOLS for r in WOUND]
    R["texture_counts"] = {
        "within_group_per_dataset": {
            "total": len(within),
            "exclude_zero": [f"{p}|{r}" for p, r in within if excl0(R["texture"][p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(within)},
        "all_sets_per_dataset": {
            "total": len(allp),
            "exclude_zero": [f"{p}|{r}" for p, r in allp if excl0(R["texture"][p][r])],
            "expected_by_chance_if_all_null": 0.05 * len(allp)},
        "mean_row_exclude_zero": [p for p in POOLS if excl0(R["texture"][p]["mean"])]}

    # 3: textures and orderings
    R["nr_texture"] = {k: MAN["synth"][k]["texture"] for k in NR}
    R["genB_texture"] = {k: MAN["synth"][k]["texture"] for a in AMPS for k in GB[a]}
    order = {}
    for t in ("M+A", "M+C", "mean4"):
        order[t] = {}
        for pool in POOLS:
            m = R["matched"][pool]
            o = {"seedmean": sorted(GENS, key=lambda g: -m[f"{g}_seedmean|{t}"]["est"])}
            for s in SEEDS:
                o[f"s{s}"] = sorted(GENS, key=lambda g: -m[f"{g}_s{s}|{t}"]["est"])
            order[t][pool] = {k: ">".join(v) for k, v in o.items()}
        sm = {order[t][p]["seedmean"] for p in POOLS}
        pair = {}
        for a, b in itertools.combinations(GENS, 2):
            wins = sum(R["matched"][p][f"{a}_s{s}|{t}"]["est"] > R["matched"][p][f"{b}_s{s}|{t}"]["est"]
                       for p in POOLS for s in SEEDS)
            wins_sm = sum(R["matched"][p][f"{a}_seedmean|{t}"]["est"] > R["matched"][p][f"{b}_seedmean|{t}"]["est"]
                          for p in POOLS)
            pair[f"{a}>{b}"] = {"pool_seed_cells": f"{wins}/{len(POOLS) * len(SEEDS)}",
                                "pools_seedmean": f"{wins_sm}/{len(POOLS)}"}
        order[t]["_stable_seedmean_ordering_across_pools"] = len(sm) == 1
        order[t]["_distinct_seedmean_orderings"] = sorted(sm)
        order[t]["_pairwise"] = pair
    R["gen_order"] = order

    # 4: leave-one-method-out for M
    full = table(WOUND + GB[40] + GB[11], "M")
    base = f_rel(40)(zr(full))
    lomo = {"_all": {"bench40_mean": base["mean|bench"], "pairs6": base["pairs6"],
                     "rel40_mean": base["mean|diff"]}}
    for i, n in enumerate(GRP["M"]):
        keep = np.array([j for j in range(len(GRP["M"])) if j != i])
        r = f_rel(40)(zr({s: v[keep] for s, v in full.items()}))
        lomo[n] = {"bench40_mean": r["mean|bench"], "pairs6": r["pairs6"], "rel40_mean": r["mean|diff"]}
    R["lomo_M"] = lomo

    # 5: mean Dice by group and set
    md = {}
    for g in ("U", "F", "K", "M", "S"):
        md[g] = {s: float(np.mean(P.vec(TT, s, GRP[g]))) for s in WOUND + SYNTH}
        for a in AMPS:
            md[g][f"genB_amp{a}_seedmean"] = float(np.mean([md[g][s] for s in GB[a]]))
        for gen in GENS:
            md[g][f"{gen}_seedmean"] = float(np.mean([md[g][f"{gen}_s{s}"] for s in SEEDS]))
    R["mean_dice"] = md

    # comparison with stats_v5
    S5 = json.load(open(os.path.join(V.RES, "stats_v5.json")))
    cmp = {"bench40": 0.0, "texture": 0.0, "matched": 0.0}
    mapk = {r: f"{r}|bench" for r in WOUND}
    mapk.update({"mean over references": "mean|bench", "clinical wound pairs": "pairs6"})
    for pool in POOLS:
        for k5, k6 in mapk.items():
            for f in ("est", "lo", "hi"):
                cmp["bench40"] = max(cmp["bench40"], abs(S5["bench40"][pool][k5][f] - R["rel"][pool]["40"][k6][f]))
        for r in WOUND + ["mean"]:
            for f in ("est", "lo", "hi"):
                cmp["texture"] = max(cmp["texture"], abs(S5["texture"][pool][r][f] - R["texture"][pool][r][f]))
        for key, v in S5["matched"][pool].items():
            assert abs(v["texture"] - MAN["synth"][key]["texture"]) < 1e-12
            for t in ("M+A", "M+C", "mean4"):
                cmp["matched"] = max(cmp["matched"], abs(v[t] - R["matched"][pool][f"{key}|{t}"]["est"]))
    R["max_abs_diff_vs_stats_v5"] = cmp
    # also check six individual pair estimates
    pe = 0.0
    for pool in POOLS:
        M_ = table(WOUND, pool)
        for a, b in itertools.combinations(WOUND, 2):
            pe = max(pe, abs(S5["bench40"][pool][f"{a}~{b}"]["est"] - sp(M_[a], M_[b])))
    R["max_abs_diff_vs_stats_v5"]["pairs_est"] = pe

    json.dump(R, open(os.path.join(OUT, "bench_v6.json"), "w"), indent=1)
    write_md(R)
    print("saved results/v6/bench_v6.json and BENCH_V6.md")


# ------------------------------------------------------------------- report

def f3(c):
    return f"{c['est']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def write_md(R):
    L = ["# Synthetic benchmark reanalysis (V6)", "",
         "Script: `scripts/v6/bench_v6.py`. Numbers: `results/v6/bench_v6.json`.", "",
         "## Definitions", "",
         "- Agreement: Spearman correlation (average ranks) of per-method mean Dice between two sets, over a method set.",
         "- Method sets: " + ", ".join(f"{p} ({n})" for p, n in R["pools"].items()) + ". S (SAM, MedSAM) is never pooled.",
         "- bench(R): agreement of generator B at the given amplitude with wound reference R, averaged over seeds 2024, 7, 101.",
         "- other(R): mean agreement of the other three wound datasets with R. Averaged over the 4 references it equals the mean of the 6 wound pairs.",
         "- diff(R) = bench(R) - other(R). The 'mean' row averages over Medetec, CO2Wounds-V2, FUSeg, AZH.",
         "- Method bootstrap: methods resampled with replacement within groups, 2,000 draws, percentile 95% intervals, RNG as in stats_v5 (draw for draw).",
         "- Hierarchical bootstrap (M and K only): methods within group plus, independently per dataset, clinical units (Medetec photograph, AZH photograph, otherwise image) and synthetic images; 2,000 draws.",
         "- Data-only bootstrap: units and images resampled, methods fixed (context only).", ""]

    def rel_table(amp):
        L.append(f"## 1. Relative agreement, generator B amplitude {amp}")
        L.append("")
        L.append("Method bootstrap intervals. Each cell: estimate [95% interval].")
        L.append("")
        L.append("| Set | Ref | bench | other | diff |")
        L.append("|---|---|---|---|---|")
        for p in POOLS:
            d = R["rel"][p][str(amp)]
            for r in WOUND + ["mean"]:
                L.append(f"| {p} | {r} | {f3(d[r + '|bench'])} | {f3(d[r + '|other'])} | {f3(d[r + '|diff'])} |")
        L.append("")
        L.append("Hierarchical (methods + units + synthetic images) and data-only intervals, M and K:")
        L.append("")
        L.append("| Set | Ref | bench (hier) | other (hier) | diff (hier) | diff (data-only) |")
        L.append("|---|---|---|---|---|---|")
        for g in ("M", "K"):
            h = R["rel_hier"][g][str(amp)]; d = R["rel_dataonly"][g][str(amp)]
            for r in WOUND + ["mean"]:
                L.append(f"| {g} | {r} | {f3(h[r + '|bench'])} | {f3(h[r + '|other'])} | {f3(h[r + '|diff'])} | {f3(d[r + '|diff'])} |")
        L.append("")
        L.append("Verdict per method set (mean over the 4 references):")
        L.append("")
        for p in POOLS:
            c = R["rel"][p][str(amp)]["mean|diff"]
            s = "distinguishable" if excl0(c) else "not distinguishable"
            refs = [r for r in WOUND if excl0(R["rel"][p][str(amp)][r + "|diff"])]
            extra = ""
            if p in ("M", "K"):
                ch = R["rel_hier"][p][str(amp)]["mean|diff"]
                extra = f"; hierarchical {f3(ch)}, {'excludes' if excl0(ch) else 'includes'} zero"
            L.append(f"- {p}: diff {f3(c)}, {s} from a clinical wound dataset as a proxy for the others"
                     f" (method bootstrap){extra}. Per-reference intervals excluding zero: {', '.join(refs) or 'none'}.")
        L.append("")

    rel_table(40)
    rel_table(11)

    L += ["## 2. Texture effect (amplitude 40 minus 11)", "",
          "Mean over seeds of agreement at amplitude 40 minus the same at 11; method bootstrap.", "",
          "| Set | " + " | ".join(WOUND) + " | mean |", "|---" * 6 + "|"]
    for p in POOLS:
        L.append(f"| {p} | " + " | ".join(f3(R["texture"][p][r]) for r in WOUND + ["mean"]) + " |")
    tc = R["texture_counts"]
    w = tc["within_group_per_dataset"]; a = tc["all_sets_per_dataset"]
    L += ["", f"- Within-group (U, F, K, M) per-dataset intervals excluding zero: {len(w['exclude_zero'])} of {w['total']} "
          f"({', '.join(w['exclude_zero']) or 'none'}); expected by chance at 95% if every effect were zero: {w['expected_by_chance_if_all_null']:.1f}.",
          f"- All 7 sets (mixtures overlap with groups, so not independent): {len(a['exclude_zero'])} of {a['total']}; chance {a['expected_by_chance_if_all_null']:.1f}.",
          f"- 'mean' intervals excluding zero: {', '.join(tc['mean_row_exclude_zero']) or 'none'}.",
          "- Hierarchical texture effect: " + "; ".join(f"{g} mean {f3(R['texture_hier'][g]['mean'])}" for g in ("M", "K")) + ".", ""]

    L += ["## 3. Generator families", "", "Texture (manifest) per benchmark and seed:", "",
          "| Benchmark | s2024 | s7 | s101 |", "|---|---|---|---|"]
    for g in GENS:
        L.append(f"| {g} | " + " | ".join(f"{R['nr_texture'][f'{g}_s{s}']:.3f}" for s in SEEDS) + " |")
    for t, lab in (("M+A", "mean of Medetec and AZH"), ("M+C", "mean of Medetec and CO2Wounds-V2"),
                   ("mean4", "mean of all 4 wound datasets")):
        L += ["", f"### Agreement with {lab}", "",
              "| Set | seed | nrA | nrB0 | nrC0 | nrC34 | B0 - C34 [95%] |", "|---|---|---|---|---|---|---|"]
        for p in POOLS:
            m = R["matched"][p]
            for s in list(SEEDS) + ["seedmean"]:
                ss = f"s{s}" if s != "seedmean" else "seedmean"
                L.append(f"| {p} | {s} | " + " | ".join(f"{m[f'{g}_{ss}|{t}']['est']:+.3f}" for g in GENS)
                         + f" | {f3(m[f'B0-C34_{ss}|{t}'])} |")
        o = R["gen_order"][t]
        L += ["", f"Seed-mean orderings: " + "; ".join(f"{p}: {o[p]['seedmean']}" for p in POOLS) + ".",
              f"Same ordering in all 7 sets: {'yes' if o['_stable_seedmean_ordering_across_pools'] else 'no'}.",
              "Pairwise (cells where first beats second, out of 21 set x seed cells; and out of 7 seed means): "
              + "; ".join(f"{k} {v['pool_seed_cells']}, {v['pools_seedmean']}" for k, v in o["_pairwise"].items()) + "."]
    L.append("")

    L += ["## 4. Leave-one-method-out, M", "",
          "| Dropped | bench40 mean | 6-pair wound | diff40 mean |", "|---|---|---|---|"]
    for n, v in R["lomo_M"].items():
        L.append(f"| {n} | {v['bench40_mean']:+.3f} | {v['pairs6']:+.3f} | {v['rel40_mean']:+.3f} |")
    L.append("")

    L += ["## 5. Mean Dice by group (how easy the benchmarks are)", "",
          "| Set | U | F | K | M | S |", "|---|---|---|---|---|---|"]
    md = R["mean_dice"]
    rows = WOUND + SYNTH + [f"genB_amp{a}_seedmean" for a in AMPS] + [f"{g}_seedmean" for g in GENS]
    for s in rows:
        L.append(f"| {s} | " + " | ".join(f"{md[g][s]:.3f}" for g in ("U", "F", "K", "M", "S")) + " |")
    L += ["", "## Checks", "",
          f"- Max abs difference, per-image loader means vs pools_v5 means: " + ", ".join(f"{k} {v:.1e}" for k, v in R["loader_max_abs_diff"].items()) + ".",
          f"- Max abs difference vs stats_v5.json (est, lo, hi): " + ", ".join(f"{k} {v:.1e}" for k, v in R["max_abs_diff_vs_stats_v5"].items()) + ".",
          "", "## Limitations", "",
          "- U and F have only per-method mean Dice on synthetic sets, so no hierarchical bootstrap for U, F or the mixtures.",
          "- That the U and F synthetic means were computed on the same 45 images per set as the Kaggle scores cannot be checked from per-method means; textures for seed 2024 match the manifest.",
          "- The four reference-level diffs share data (each wound pair appears in two 'other' terms), so per-reference intervals are not independent.",
          "- Floor effect: M scores near zero on nrA (group mean Dice about 0.01; 67 to 100% of images below Dice 0.01 per method on seed 7), so M agreements involving nrA rank near-zero values and carry little information. K scores about 0.84 there, so in U+M and U+K the nrA ranking is driven by the gap between groups.",
          "- In mixed sets (U+F, U+K, U+M) a Spearman over pooled methods mixes within-group ordering with between-group difficulty offsets; differences between mixtures and single groups partly reflect those offsets.",
          "- Data-only intervals sit asymmetrically around the estimate because resampling images adds noise that pulls Spearman toward zero; they are context, not the primary interval.", ""]
    open(os.path.join(OUT, "BENCH_V6.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()

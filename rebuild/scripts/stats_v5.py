#!/usr/bin/env python3
"""Every number reported in manuscript V5, with stratified method-bootstrap
95% intervals, written to results/stats_v5.json.

Pools (all scored on the same images, duplicated evaluation images removed):
  U  18 untrained classical methods
  F  14 methods trained from scratch on FUSeg
  K  the same 14, trained on Kvasir-SEG
  M  11 modern methods trained on FUSeg: 10 ImageNet-pretrained architectures
     and nnU-Net
  mixtures U+F, U+K, U+M
Bootstrap: methods resampled with replacement within each group (so a mixed
pool keeps its composition), 2,000 draws, percentile intervals.
"""
import itertools, json, os, sys
import numpy as np
from scipy.stats import rankdata
from sympy.utilities.iterables import multiset_permutations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
from hierarchy_v4 import REFS as HREFS, level_of, LEVELS

B = int(os.environ.get("B", "2000"))
G = P.groups(); TT = P.load()
GRP = {"U": G["U"], "F": G["F"], "K": G["K"], "M": G["P"] + G["N"]}
POOLS = {"U": ["U"], "F": ["F"], "K": ["K"], "M": ["M"],
         "U+F": ["U", "F"], "U+K": ["U", "K"], "U+M": ["U", "M"]}
KEYS = list(HREFS)
WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
NONW = [k for k in KEYS if HREFS[k][0] != "wound"]
COLON = [k for k in KEYS if HREFS[k][0] == "colonoscopy"]
SEEDS = (2024, 7, 101)
MAN = json.load(open("/home/claude/kaggle_kit/manifest.json"))


def sp(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def names_of(pool):
    return sum((GRP[g] for g in POOLS[pool]), [])


def table(metric, sets, pool):
    nm = names_of(pool)
    return {s: P.vec(TT[metric], s, nm) for s in sets}


def boot(pool, metric, sets, fn, b=B, seed=0):
    """fn(dict set->vector) -> dict of named scalars. Returns obs and CIs."""
    full = table(metric, sets, pool)
    obs = fn(full)
    nm = names_of(pool)
    idx = [np.array([nm.index(n) for n in GRP[g]]) for g in POOLS[pool]]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in obs}
    for _ in range(b):
        take = np.concatenate([rng.choice(i, len(i)) for i in idx])
        r = fn({s: v[take] for s, v in full.items()})
        for k in obs:
            draws[k].append(r[k])
    return {k: {"est": obs[k],
                "lo": float(np.nanpercentile(draws[k], 2.5)),
                "hi": float(np.nanpercentile(draws[k], 97.5))} for k in obs}


# ------------------------------------------------------------- statistics

def rmat(M):
    return {(a, b): sp(M[a], M[b]) for a, b in itertools.combinations(KEYS, 2)}


def mod_diff(R, labels):
    same = [R[p] for p in R if labels[p[0]] == labels[p[1]]]
    diff = [R[p] for p in R if labels[p[0]] != labels[p[1]]]
    return float(np.nanmean(same) - np.nanmean(diff))


LAB = {k: HREFS[k][0] for k in KEYS}


def f_hier(M):
    R = rmat(M)
    out = {"modality_diff": mod_diff(R, LAB)}
    for L in LEVELS[1:]:
        out[L] = float(np.nanmean([R[(a, b)] for (a, b) in R if level_of(a, b) == L]))
    out["same dataset"] = sp(M["azh"], M["azh_train_slice"])
    return out


def exact_tests(M):
    """Exact relabeling test of the modality difference, and block tests for
    whether the wound (or colonoscopy) references cluster."""
    R = rmat(M)
    lab = [LAB[k] for k in KEYS]
    obs = mod_diff(R, LAB)
    vals = [mod_diff(R, dict(zip(KEYS, l))) for l in multiset_permutations(lab)]
    out = {"modality": {"diff": obs, "p": float(np.mean(np.array(vals) >= obs - 1e-12)),
                        "n_relabelings": len(vals), "min_p": 1 / len(vals)}}
    for name, block in (("wound", WOUND), ("colonoscopy", COLON)):
        def stat(sub):
            s = set(sub)
            inn = [R[p] for p in R if p[0] in s and p[1] in s]
            cross = [R[p] for p in R if (p[0] in s) != (p[1] in s)]
            return np.mean(inn) - np.mean(cross)
        o = stat(block)
        null = [stat(c) for c in itertools.combinations(KEYS, len(block))]
        out[name] = {"diff": float(o), "p": float(np.mean(np.array(null) >= o - 1e-12)),
                     "n_subsets": len(null), "min_p": 1 / len(null)}
    return out


def f_comp(domain_refs):
    dom = set(domain_refs)

    def fn(M):
        R = rmat(M)
        cat = {"in x in": [], "in x out": [], "out, same modality": [],
               "out, across modality": []}
        for (a, b), r in R.items():
            ia, ib = a in dom, b in dom
            if ia and ib: cat["in x in"].append(r)
            elif ia != ib: cat["in x out"].append(r)
            elif LAB[a] == LAB[b]: cat["out, same modality"].append(r)
            else: cat["out, across modality"].append(r)
        return {k: float(np.nanmean(v)) for k, v in cat.items()}
    return fn


def f_bench(amp):
    def fn(M):
        out = {}
        for r in WOUND:
            out[r] = float(np.nanmean([sp(M[f"genB_amp{amp}_s{s}"], M[r]) for s in SEEDS]))
        out["mean over references"] = float(np.mean([out[r] for r in WOUND]))
        pr = [sp(M[a], M[b]) for a, b in itertools.combinations(WOUND, 2)]
        out["clinical wound pairs"] = float(np.nanmean(pr))
        for a, b in itertools.combinations(WOUND, 2):
            out[f"{a}~{b}"] = sp(M[a], M[b])
        return out
    return fn


def f_texture(M):
    out = {}
    for r in WOUND + ["mean"]:
        def ag(amp):
            rs = WOUND if r == "mean" else [r]
            return np.mean([sp(M[f"genB_amp{amp}_s{s}"], M[x]) for s in SEEDS for x in rs])
        out[r] = float(ag(40) - ag(11))
    return out


def f_stratified(M_groups):
    """Agreement computed on ranks taken within each group, then pooled."""
    pass


def stratified(pool, metric):
    """Ranks within each group, pooled; then the modality test and categories."""
    nm = names_of(pool)
    Mv = table(metric, KEYS, pool)
    idx = [np.array([nm.index(n) for n in GRP[g]]) for g in POOLS[pool]]
    W = {}
    for k, v in Mv.items():
        w = np.empty_like(v)
        for i in idx:
            w[i] = (rankdata(v[i]) - 0.5) / len(i)       # within-group percentile
        W[k] = w
    R = rmat(W)
    lab = [LAB[k] for k in KEYS]
    obs = mod_diff(R, LAB)
    vals = [mod_diff(R, dict(zip(KEYS, l))) for l in multiset_permutations(lab)]
    comp = f_comp(WOUND)(W)
    return {"modality_diff": obs, "p": float(np.mean(np.array(vals) >= obs - 1e-12)),
            "categories_wound": comp}


def reliability():
    """Split-half reliability of each reference's ranking over all 57 methods."""
    import glob
    rng = np.random.default_rng(1)
    out = {}
    for r in KEYS + ["azh_train_slice"]:
        mats, empt = [], None
        d = json.load(open(os.path.join(P.RES, "perimage32", r + ".json")))
        mats.append(np.asarray(d["dice"]))
        for f in glob.glob(os.path.join(P.RES, "kaggle", "scores", "*", r + ".json")):
            g = json.load(open(f))
            if g["group"] != "sam":
                mats.append(np.asarray(g["dice"]))
        X = np.concatenate(mats, axis=1)
        if r in P.EXCL:
            drop = {e[0] for e in P.EXCL[r]}
            X = X[[i for i in range(len(X)) if i not in drop]]
        rs = []
        for _ in range(200):
            p = rng.permutation(len(X)); h = len(X) // 2
            c = sp(X[p[:h]].mean(0), X[p[h:]].mean(0))
            rs.append(2 * c / (1 + c))
        out[r] = {"n_images": int(len(X)), "n_methods": int(X.shape[1]),
                  "spearman_brown": float(np.median(rs))}
    return out


def matched(pool, metric="dice"):
    """Table 8 benchmarks at three seeds: texture and agreement per pair."""
    nm = names_of(pool)
    out = {}
    for s in SEEDS:
        for b in ("nrA", "nrB0", "nrC0", "nrC34"):
            key = f"{b}_s{s}"
            v = P.vec(TT[metric], key, nm)
            out[key] = {"texture": MAN["synth"][key]["texture"],
                        "M+A": float(np.mean([sp(v, P.vec(TT[metric], r, nm)) for r in ("medetec", "azh")])),
                        "M+C": float(np.mean([sp(v, P.vec(TT[metric], r, nm)) for r in ("medetec", "co2wounds")])),
                        "mean4": float(np.mean([sp(v, P.vec(TT[metric], r, nm)) for r in WOUND]))}
    return out


def mean_dice():
    out = {}
    for g in ("U", "F", "K", "M"):
        out[g] = {r: float(np.mean(P.vec(TT["dice"], r, GRP[g]))) for r in KEYS + ["azh_train_slice"]}
        out[g]["genB_amp40"] = float(np.mean([np.mean(P.vec(TT["dice"], f"genB_amp40_s{s}", GRP[g])) for s in SEEDS]))
    out["S"] = {r: float(np.mean(P.vec(TT["dice"], r, G["S"]))) for r in KEYS}
    out["M_members"] = {n: {r: TT["dice"][r][n] for r in ("fuseg", "medetec", "co2wounds", "azh")} for n in GRP["M"]}
    return out


if __name__ == "__main__":
    S = {"pools": {k: {"groups": v, "n": len(names_of(k))} for k, v in POOLS.items()},
         "hier": {}, "exact": {}, "comp": {}, "bench40": {}, "bench_avg": {},
         "texture": {}, "stratified": {}, "matched": {}}
    for metric in ("dice", "nsd", "dice_noempty"):
        for pool in POOLS:
            S["hier"][f"{metric}|{pool}"] = boot(pool, metric, KEYS + ["azh_train_slice"], f_hier, b=B // 2)
            S["exact"][f"{metric}|{pool}"] = exact_tests(table(metric, KEYS, pool))
            print(metric, pool, round(S["exact"][f"{metric}|{pool}"]["modality"]["diff"], 3), flush=True)
    for metric in ("dice", "nsd"):
        for pool in POOLS:
            dom = COLON if "K" in POOLS[pool] else WOUND
            S["comp"][f"{metric}|{pool}|{'colonoscopy' if dom is COLON else 'wound'}"] = \
                boot(pool, metric, KEYS, f_comp(dom))
            if pool == "U":
                S["comp"][f"{metric}|U|colonoscopy"] = boot(pool, metric, KEYS, f_comp(COLON))
            if pool in ("U", "F", "U+F", "M", "U+M"):
                S["comp"][f"{metric}|{pool}|wound"] = boot(pool, metric, KEYS, f_comp(WOUND))
    synth = [f"genB_amp{a}_s{s}" for a in (11, 28, 40) for s in SEEDS]
    for pool in POOLS:
        S["bench40"][pool] = boot(pool, "dice", WOUND + synth, f_bench(40))
        S["texture"][pool] = boot(pool, "dice", WOUND + synth, f_texture)
        S["matched"][pool] = matched(pool)
    for pool in ("U+F", "U+K", "U+M"):
        S["stratified"][pool] = {m: stratified(pool, m) for m in ("dice", "nsd")}
    S["reliability"] = reliability()
    S["mean_dice"] = mean_dice()
    json.dump(S, open(os.path.join(P.RES, "stats_v5.json"), "w"), indent=1)
    print("saved results/stats_v5.json")

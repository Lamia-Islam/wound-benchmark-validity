#!/usr/bin/env python3
"""Every number in manuscript V5, with stratified method-bootstrap intervals.

Pools: U (18 classical), F (14 trained from scratch on FUSeg), M (10
ImageNet-pretrained architectures plus nnU-Net, trained on FUSeg), K (the F
recipe trained on Kvasir-SEG), and mixtures U+F, U+M, U+K. SAM and MedSAM are
box-prompted with the ground truth and are reported only as mean Dice.

Writes results/v5_numbers.json.
"""
import itertools, json, os, sys
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
from hierarchy_v4 import REFS as HREFS, level_of
from sympy.utilities.iterables import multiset_permutations

B = int(os.environ.get("BOOT", "2000"))
G = P.groups(); TAB = P.load()
G["M"] = G["P"] + G["N"]
KEYS = list(HREFS)
WOUND = [k for k in KEYS if HREFS[k][0] == "wound"]
COLON = [k for k in KEYS if HREFS[k][0] == "colonoscopy"]
GENB = [f"genB_amp{a}_s{s}" for a in (11, 28, 40) for s in (2024, 7, 101)]
POOLS = {"U": ["U"], "F": ["F"], "M": ["M"], "K": ["K"],
         "U+F": ["U", "F"], "U+M": ["U", "M"], "U+K": ["U", "K"]}
DOMAIN = {"F": "wound", "M": "wound", "K": "colonoscopy", "U+F": "wound",
          "U+M": "wound", "U+K": "colonoscopy", "U": "wound"}
LEVELS = ["same dataset", "same source", "different source, same modality",
          "across modality"]


def sp(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def strat_sp(a, b, gid):
    """Agreement with ranks taken within each method group, then pooled."""
    ra = np.empty(len(a)); rb = np.empty(len(b))
    for g in np.unique(gid):
        m = gid == g
        x, y = rankdata(a[m]), rankdata(b[m])
        ra[m] = x - x.mean(); rb[m] = y - y.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def matrix(V, f=sp):
    return {(a, b): f(V[a], V[b]) for a, b in itertools.combinations(KEYS, 2)}


def stats(V, dom, gid=None):
    R = matrix(V)
    out = {}
    for L in LEVELS[1:]:
        out["level:" + L] = np.nanmean([r for (a, b), r in R.items() if level_of(a, b) == L])
    out["control"] = sp(V["azh"], V["azh_train_slice"])
    same = [r for (a, b), r in R.items() if HREFS[a][0] == HREFS[b][0]]
    diff = [r for (a, b), r in R.items() if HREFS[a][0] != HREFS[b][0]]
    out["modality_diff"] = np.nanmean(same) - np.nanmean(diff)
    ww = [r for (a, b), r in R.items() if a in WOUND and b in WOUND]
    ww_ind = [r for (a, b), r in R.items() if a in WOUND and b in WOUND
              and level_of(a, b) != "same source"]
    cc = [r for (a, b), r in R.items() if a in COLON and b in COLON]
    out["wound_x_wound"] = np.nanmean(ww)
    out["wound_x_wound_indep"] = np.nanmean(ww_ind)
    out["colon_x_colon"] = np.nanmean(cc)
    for k in (dom,):
        ins = [a for a in KEYS if HREFS[a][0] == k]
        cat = {"in_in": [], "in_out": [], "out_same": [], "out_across": []}
        for (a, b), r in R.items():
            ia, ib = a in ins, b in ins
            key = ("in_in" if ia and ib else "in_out" if ia != ib else
                   "out_same" if HREFS[a][0] == HREFS[b][0] else "out_across")
            cat[key].append(r)
        for c, v in cat.items():
            out["dom:" + c] = np.nanmean(v)
    bench = [np.mean([sp(V[c], V[r]) for c in GENB]) for r in WOUND]
    for r, v in zip(WOUND, bench):
        out["bench:" + r] = v
    out["bench_wound"] = np.mean(bench)
    for a in (11, 28, 40):
        out[f"bench_amp{a}"] = np.mean([sp(V[f"genB_amp{a}_s{s}"], V[r])
                                        for s in (2024, 7, 101) for r in WOUND])
    if gid is not None and len(np.unique(gid)) > 1:
        RS = matrix(V, lambda a, b: strat_sp(a, b, gid))
        same = [r for (a, b), r in RS.items() if HREFS[a][0] == HREFS[b][0]]
        diff = [r for (a, b), r in RS.items() if HREFS[a][0] != HREFS[b][0]]
        out["strat:modality_diff"] = np.mean(same) - np.mean(diff)
        out["strat:wound_x_wound"] = np.mean([r for (a, b), r in RS.items() if a in WOUND and b in WOUND])
        ins = [a for a in KEYS if HREFS[a][0] == dom]
        out["strat:in_out"] = np.mean([r for (a, b), r in RS.items() if (a in ins) != (b in ins)])
        out["strat:bench_wound"] = np.mean([np.mean([strat_sp(V[c], V[r], gid) for c in GENB]) for r in WOUND])
    return out


def exact_modality_p(V):
    R = matrix(V)
    lab = [HREFS[k][0] for k in KEYS]
    M = np.zeros((len(KEYS), len(KEYS)))
    for (a, b), r in R.items():
        i, j = KEYS.index(a), KEYS.index(b); M[i, j] = M[j, i] = r
    iu = np.triu_indices(len(KEYS), 1)
    def st(l):
        l = np.asarray(l); s = (l[:, None] == l[None, :])[iu]
        return M[iu][s].mean() - M[iu][~s].mean()
    obs = st(lab)
    vals = np.array([st(l) for l in multiset_permutations(lab)])
    # wounds alone: do the four wound references cluster among the 11?
    wl = ["w" if HREFS[k][0] == "wound" else "o" for k in KEYS]
    def stw(l):
        l = np.asarray(l); idx = np.where(l == "w")[0]
        return np.mean([M[i, j] for i, j in itertools.combinations(idx, 2)])
    wobs = stw(wl)
    wvals = np.array([stw(l) for l in multiset_permutations(wl)])
    return {"p": float((vals >= obs - 1e-12).mean()), "n_relabel": len(vals),
            "p_min": float(1 / len(vals)),
            "wound_cluster_p": float((wvals >= wobs - 1e-12).mean()),
            "wound_cluster_n": len(wvals)}


def run(metric, gs, rng):
    names = sum((G[g] for g in gs), [])
    gid = np.concatenate([[i] * len(G[g]) for i, g in enumerate(gs)])
    sets = KEYS + ["azh_train_slice"] + (GENB if metric == "dice" else [])
    full = {s: P.vec(TAB[metric], s, names) for s in sets}
    if metric != "dice":
        for c in GENB:
            full[c] = P.vec(TAB["dice"], c, names)
    dom = DOMAIN.get("+".join(gs), "wound")
    obs = stats(full, dom, gid)
    idx = [np.where(gid == i)[0] for i in range(len(gs))]
    bs = []
    for _ in range(B):
        take = np.concatenate([rng.choice(i, len(i)) for i in idx])
        bs.append(stats({s: v[take] for s, v in full.items()}, dom, gid[take]))
    ci = {k: [float(np.nanpercentile([b[k] for b in bs], 2.5)),
              float(np.nanpercentile([b[k] for b in bs], 97.5))] for k in obs}
    return {"n": len(names), "obs": {k: float(v) for k, v in obs.items()}, "ci": ci,
            "exact": exact_modality_p(full)}


def main():
    rng = np.random.default_rng(0)
    res = {"groups": {g: v for g, v in G.items()}, "pools": {}}
    for metric in ("dice", "nsd", "dice_noempty"):
        for pn, gs in POOLS.items():
            r = run(metric, gs, rng)
            res["pools"][f"{metric}|{pn}"] = r
            o, c = r["obs"], r["ci"]
            print(f"{metric:<12} {pn:<4} n={r['n']:>2} modality {o['modality_diff']:+.2f} "
                  f"[{c['modality_diff'][0]:+.2f},{c['modality_diff'][1]:+.2f}] p {r['exact']['p']:.4f} "
                  f"woundclust p {r['exact']['wound_cluster_p']:.3f} | ww {o['wound_x_wound']:+.2f} "
                  f"[{c['wound_x_wound'][0]:+.2f},{c['wound_x_wound'][1]:+.2f}] | in_out {o['dom:in_out']:+.2f} "
                  f"[{c['dom:in_out'][0]:+.2f},{c['dom:in_out'][1]:+.2f}] | bench {o['bench_wound']:+.2f} "
                  f"[{c['bench_wound'][0]:+.2f},{c['bench_wound'][1]:+.2f}] | ctrl {o['control']:+.2f}", flush=True)
    # mean Dice per group and reference (SAM included here only)
    res["mean_dice"] = {g: {s: float(np.mean(P.vec(TAB["dice"], s, G[g])))
                            for s in KEYS + ["azh_train_slice"] + GENB}
                        for g in ("U", "F", "M", "K", "S", "P", "N")}
    json.dump(res, open(os.path.join(ROOT, "results", "v5_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

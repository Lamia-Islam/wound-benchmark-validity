#!/usr/bin/env python3
"""Stratified method bootstrap for the headline pool comparisons."""
import itertools, os, sys
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
from hierarchy_v4 import REFS as HREFS
G = P.groups(); T = P.load()["dice"]
WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
KEYS = list(HREFS)
SETS = KEYS + [f"genB_amp40_s{s}" for s in (2024, 7, 101)]

def sp(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (ra * rb).sum() / d if d > 0 else np.nan

def stats(M):                      # M: dict set -> vector over the sampled methods
    ww = np.nanmean([sp(M[a], M[b]) for a, b in itertools.combinations(WOUND, 2)])
    gb = np.nanmean([sp(M[f"genB_amp40_s{s}"], M[r]) for s in (2024, 7, 101) for r in WOUND])
    wn = np.nanmean([sp(M[a], M[b]) for a in WOUND for b in KEYS if HREFS[b][0] != "wound"])
    return ww, gb, wn

rng = np.random.default_rng(0)
def run(groups, B=2000):
    names = sum((G[g] for g in groups), [])
    full = {s: P.vec(T, s, names) for s in SETS}
    obs = stats(full)
    idx = [np.array([names.index(n) for n in G[g]]) for g in groups]
    bs = []
    for _ in range(B):
        take = np.concatenate([rng.choice(i, len(i)) for i in idx])
        bs.append(stats({s: v[take] for s, v in full.items()}))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return obs, lo, hi

lab = ["wound x wound", "benchmark x wound (amp 40)", "wound x non-wound"]
for groups in (["U"], ["F"], ["P"], ["P", "N"], ["K"], ["U", "F"], ["U", "P"], ["U", "P", "N"]):
    obs, lo, hi = run(groups)
    print("+".join(groups).ljust(5), "  ".join(f"{l}: {o:+.2f} [{a:+.2f}, {b:+.2f}]" for l, o, a, b in zip(lab, obs, lo, hi)))

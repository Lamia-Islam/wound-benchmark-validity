#!/usr/bin/env python3
"""Hierarchy, composition and benchmark validity across all method groups."""
import itertools, json, os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
from hierarchy_v4 import REFS as HREFS, level_of, LEVELS
from sympy.utilities.iterables import multiset_permutations

G = P.groups(); T = P.load()
KEYS = list(HREFS)                      # 11 non-control references
rho = lambda a, b: float(spearmanr(a, b).statistic)

def pairs_table(metric, names):
    v = {k: P.vec(T[metric], k, names) for k in KEYS + ["azh_train_slice"]}
    rows = [(a, b, level_of(a, b), rho(v[a], v[b])) for a, b in itertools.combinations(KEYS, 2)]
    rows.append(("azh", "azh_train_slice", "same dataset", rho(v["azh"], v["azh_train_slice"])))
    return rows

def levels(rows):
    return {L: float(np.mean([r[3] for r in rows if r[2] == L])) for L in LEVELS}

def modality_test(metric, names):
    v = {k: P.vec(T[metric], k, names) for k in KEYS}
    lab = [HREFS[k][0] for k in KEYS]
    R = np.array([[rho(v[a], v[b]) if a != b else 0 for b in KEYS] for a in KEYS])
    iu = np.triu_indices(len(KEYS), 1)
    def stat(l):
        l = np.asarray(l); same = (l[:, None] == l[None, :])[iu]
        return R[iu][same].mean() - R[iu][~same].mean()
    obs = stat(lab)
    vals = np.array([stat(l) for l in multiset_permutations(lab)])
    return float(obs), float((vals >= obs - 1e-12).mean())

def category(a, b, domain):
    ia, ib = HREFS[a][0] == domain, HREFS[b][0] == domain
    if ia and ib: return "in x in"
    if ia != ib: return "in x out"
    return "out, same modality" if HREFS[a][0] == HREFS[b][0] else "out, across modality"

def domain_split(metric, names, domain):
    v = {k: P.vec(T[metric], k, names) for k in KEYS}
    acc = {}
    for a, b in itertools.combinations(KEYS, 2):
        acc.setdefault(category(a, b, domain), []).append(rho(v[a], v[b]))
    return {c: (len(x), float(np.mean(x))) for c, x in acc.items()}

res = {}
pools = {"U18": G["U"], "F14": G["F"], "K14": G["K"], "P10": G["P"],
         "U18+F14": G["U"] + G["F"], "U18+K14": G["U"] + G["K"],
         "U18+P10": G["U"] + G["P"]}
if "N" in G:
    pools["P10+N"] = G["P"] + G["N"]
    pools["U18+P10+N"] = G["U"] + G["P"] + G["N"]
print("== mean Dice per pool on each reference ==")
for k in KEYS:
    print(f"  {k:<16}" + "".join(f"{p}:{np.mean(P.vec(T['dice'], k, G[p])):.3f}  " for p in ("U", "F", "P", "K", "S") if p in G))
for metric in ("dice", "nsd"):
    print(f"\n== {metric}: level means and modality test per pool ==")
    for pn, names in pools.items():
        lv = levels(pairs_table(metric, names)); mt = modality_test(metric, names)
        res[(metric, pn)] = {"levels": lv, "modality": mt}
        print(f"  {pn:<11} n={len(names):>2} | " + " ".join(f"{L[:14]}:{lv[L]:+.3f}" for L in LEVELS) + f" | test {mt[0]:+.3f} p {mt[1]:.4f}")
    print(f"\n== {metric}: composition, split by training domain ==")
    for pn, dom in (("U18", "wound"), ("F14", "wound"), ("U18+F14", "wound"),
                    ("U18", "colonoscopy"), ("K14", "colonoscopy"), ("U18+K14", "colonoscopy"),
                    ("P10", "wound"), ("U18+P10", "wound")):
        ds = domain_split(metric, pools[pn], dom)
        print(f"  {pn:<9} domain={dom:<12} " + "  ".join(f"{c}({n}):{m:+.3f}" for c, (n, m) in sorted(ds.items())))

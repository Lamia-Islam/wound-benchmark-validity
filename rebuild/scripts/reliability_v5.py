#!/usr/bin/env python3
"""Split-half reliability of each reference's method ranking, per pool.

For each reference, images are split at random into halves (100 splits), the
pool is ranked on each half, and the Spearman between halves is stepped up to
the full set with Spearman-Brown. Exclusions in exclusions_v5.json apply.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
G = P.groups(); G["M"] = G["P"] + G["N"]
RES = os.path.join(ROOT, "results")
EX = json.load(open(os.path.join(RES, "exclusions_v5.json")))

def per_image(ref, metric="dice"):
    cols = {}
    d = json.load(open(os.path.join(RES, "perimage32", ref + ".json")))
    A = [np.asarray(d[metric], float)]; names = list(d["names"])
    for f in glob.glob(os.path.join(RES, "kaggle", "scores", "*", ref + ".json")):
        e = json.load(open(f)); A.append(np.asarray(e[metric], float)); names += e["names"]
    A = np.concatenate(A, axis=1)
    drop = {x[0] for x in EX.get(ref, [])}
    keep = np.array([i not in drop for i in range(A.shape[0])])
    return A[keep], names

def reliability(A, cols, rng, n=100):
    X = A[:, cols]; m = X.shape[0]; out = []
    for _ in range(n):
        p = rng.permutation(m); h = m // 2
        r = spearmanr(X[p[:h]].mean(0), X[p[h:]].mean(0)).statistic
        out.append(2 * r / (1 + r))
    return float(np.nanmean(out))

rng = np.random.default_rng(0)
res = {}
for metric in ("dice", "nsd"):
    for ref in P.REFS:
        A, names = per_image(ref, metric)
        for pn, gs in {"U": ["U"], "F": ["F"], "M": ["M"], "K": ["K"], "U+F": ["U", "F"],
                       "U+M": ["U", "M"], "U+K": ["U", "K"]}.items():
            cols = [names.index(x) for g in gs for x in G[g]]
            res[f"{metric}|{pn}|{ref}"] = reliability(A, cols, rng)
json.dump(res, open(os.path.join(RES, "reliability_v5.json"), "w"), indent=1)
for metric in ("dice",):
    print("ref".ljust(16) + "".join(p.rjust(7) for p in ("U", "F", "M", "K", "U+F", "U+M", "U+K")))
    for ref in P.REFS:
        print(ref.ljust(16) + "".join(f"{res[f'{metric}|{p}|{ref}']:7.2f}" for p in ("U", "F", "M", "K", "U+F", "U+M", "U+K")))

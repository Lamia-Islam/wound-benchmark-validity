#!/usr/bin/env python3
"""Generator B agreement with each wound reference, per method pool."""
import os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
G = P.groups(); T = P.load()["dice"]
WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
pools = {"U18": G["U"], "F14": G["F"], "P10": G["P"], "K14": G["K"],
         "U18+F14": G["U"] + G["F"], "U18+P10": G["U"] + G["P"], "U18+K14": G["U"] + G["K"]}
if "N" in G:
    pools["P10+N"] = G["P"] + G["N"]
rho = lambda a, b: float(spearmanr(a, b).statistic)
print("clinical-to-clinical agreement among wound references, per pool")
for pn, names in pools.items():
    r = {f"{a[:3]}-{b[:3]}": rho(P.vec(T, a, names), P.vec(T, b, names))
         for i, a in enumerate(WOUND) for b in WOUND[i + 1:]}
    print(f"  {pn:<9} " + "  ".join(f"{k}:{v:+.2f}" for k, v in r.items()))
print("\ngenerator B agreement with each reference (mean over 3 seeds), amp 11 / 28 / 40")
for pn, names in pools.items():
    s = f"  {pn:<9}"
    for ref in WOUND:
        vals = []
        for amp in (11, 28, 40):
            vals.append(np.mean([rho(P.vec(T, f"genB_amp{amp}_s{sd}", names),
                                     P.vec(T, ref, names)) for sd in (2024, 7, 101)]))
        s += f" {ref[:6]} " + "/".join(f"{v:+.2f}" for v in vals)
    print(s)
print("\nmean Dice on generator B amp 40 per group: " +
      "  ".join(f"{g}:{np.mean([P.vec(T, f'genB_amp40_s{sd}', G[g]).mean() for sd in (2024, 7, 101)]):.3f}"
                for g in G if g != 'N' or True))

#!/usr/bin/env python3
"""Flatten every number the V5 manuscript quotes into manuscript_v5/facts.json.

Keys: <metric>.<pool>.<stat> -> {"v": "0.13", "ci": "−0.13 to 0.38",
"vci": "0.13 (95% CI −0.13 to 0.38)"}, plus exact-test p values, mean Dice per
group, reliability ranges, training reports and the matched-benchmark table.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "..", "manuscript_v5", "facts.json")

def f(x, sign=False):
    s = f"{x:+.2f}" if sign else f"{x:.2f}"
    return s.replace("-", "−")

def fp(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

N = json.load(open(os.path.join(RES, "v5_numbers.json")))
F = {}
for key, r in N["pools"].items():
    metric, pool = key.split("|")
    for st, v in r["obs"].items():
        lo, hi = r["ci"][st]
        k = f"{metric}.{pool}.{st}"
        F[k] = {"v": f(v), "ci": f"{f(lo)} to {f(hi)}", "vci": f"{f(v)} (95% CI {f(lo)} to {f(hi)})",
                "raw": [v, lo, hi]}
    e = r["exact"]
    F[f"{metric}.{pool}.p"] = fp(e["p"]); F[f"{metric}.{pool}.p_raw"] = e["p"]
    F[f"{metric}.{pool}.pw"] = fp(e["wound_cluster_p"]); F[f"{metric}.{pool}.pw_raw"] = e["wound_cluster_p"]
    F["n_relabel"] = e["n_relabel"]; F["p_min"] = f"{e['p_min']:.5f}"
    F["n_wound_relabel"] = e["wound_cluster_n"]
    F[f"n.{pool}"] = r["n"]
md = N["mean_dice"]
for g, d in md.items():
    for s, v in d.items():
        F[f"md.{g}.{s}"] = f(v)
# ranges quoted in the text
G0 = P.groups(); G0["M"] = G0["P"] + G0["N"]
T0 = P.load()["dice"]
COLON = ["kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib"]
NONW = [r for r in P.REFS if r not in ("medetec", "co2wounds", "fuseg", "azh", "azh_train_slice")]
def mrange(g, sets):
    v = [np.mean(P.vec(T0, s, G0[g])) for s in sets]
    return f"{f(min(v))} to {f(max(v))}"
F["mdr.M.colon"] = mrange("M", COLON)
F["mdr.S.all"] = mrange("S", [r for r in P.REFS if r != "azh_train_slice"])
F["mdr.K.wound"] = mrange("K", ["medetec", "co2wounds", "fuseg", "azh"])
F["mdr.U.wound"] = mrange("U", ["medetec", "co2wounds", "fuseg", "azh"])
for g in ("U", "M", "F"):
    F[f"md.{g}.genB40"] = f(np.mean([np.mean(P.vec(T0, f"genB_amp40_s{s}", G0[g])) for s in (2024, 7, 101)]))
cc = [N["pools"][f"dice|{g}"]["obs"]["colon_x_colon"] for g in ("U", "F", "M", "K")]
F["colon_range"] = f"{f(min(cc))} to {f(max(cc))}"
# mirror test: colonoscopy against other modalities, within each group and pooled
import itertools
from hierarchy_v4 import REFS as HR
def colon_out(names):
    v = {k: P.vec(T0, k, names) for k in HR}
    return np.mean([spearmanr(v[a], v[b]).statistic for a, b in itertools.combinations(HR, 2)
                    if (HR[a][0] == "colonoscopy") != (HR[b][0] == "colonoscopy")])
F["mirror.U"] = f(colon_out(G0["U"])); F["mirror.K"] = f(colon_out(G0["K"]))
F["mirror.UK"] = f(colon_out(G0["U"] + G0["K"]))
man0 = json.load(open("/home/claude/kaggle_kit/manifest.json"))["synth"]
spread = []
for base in ["genB_amp11", "genB_amp28", "genB_amp40", "nrA", "nrB0", "nrC0", "nrC34"]:
    t = [man0[f"{base}_s{s}"]["texture"] for s in (2024, 7, 101)]
    spread.append(max(t) - min(t))
F["tex_seed_spread"] = f(max(spread))
REL = json.load(open(os.path.join(RES, "reliability_v5.json")))
WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
for pool in ("U", "F", "M", "K"):
    for metric in ("dice",):
        vals = [REL[f"{metric}|{pool}|{r}"] for r in P.REFS if r != "azh_train_slice"]
        w = [REL[f"{metric}|{pool}|{r}"] for r in WOUND]
        F[f"rel.{pool}.all"] = f"{f(min(vals))} to {f(max(vals))}"
        F[f"rel.{pool}.wound"] = f"{f(min(w))} to {f(max(w))}"
tr = json.load(open(os.path.join(RES, "kaggle", "train_report_pretrained.json")))
v = [x["sweep_dice"] for x in tr.values()]
F["pt_heldout"] = f"{f(min(v))} to {f(max(v))}"
EX = json.load(open(os.path.join(RES, "exclusions_v5.json")))
F["n_excl_fuseg"] = len(EX["fuseg"]); F["n_excl_ctrl"] = len(EX["azh_train_slice"])
# matched-texture benchmarks over three seeds (exploratory)
G = P.groups(); G["M"] = G["P"] + G["N"]; T = P.load()["dice"]
man = json.load(open("/home/claude/kaggle_kit/manifest.json"))["synth"]
rows = {}
for b in ("nrA", "nrC0", "nrC34", "nrB0"):
    tex = np.mean([man[f"{b}_s{s}"]["texture"] for s in (2024, 7, 101)])
    texr = [man[f"{b}_s{s}"]["texture"] for s in (2024, 7, 101)]
    row = {"texture": f(tex), "texture_range": f"{f(min(texr))} to {f(max(texr))}"}
    for pool, gs in (("U", ["U"]), ("M", ["M"]), ("U+M", ["U", "M"]), ("U+F", ["U", "F"])):
        names = sum((G[g] for g in gs), [])
        ok = all(all(n in T.get(f"{b}_s{s}", {}) for n in names) for s in (2024, 7, 101))
        if not ok:
            row[pool] = None; continue
        vals = [np.mean([spearmanr(P.vec(T, f"{b}_s{s}", names), P.vec(T, r, names)).statistic
                         for r in WOUND]) for s in (2024, 7, 101)]
        row[pool] = f"{f(np.mean(vals))} (sd {np.std(vals, ddof=1):.2f})"
    rows[b] = row
F["matched"] = rows
mv = [float(rows[k]["U+M"].split()[0].replace("\u2212", "-")) for k in rows if rows[k]["U+M"]]
F["matched_mixed_range"] = f"{f(min(mv))} to {f(max(mv))}"
json.dump(F, open(OUT, "w"), indent=1, ensure_ascii=False)
print("facts:", len(F)); print(json.dumps(rows, indent=1, ensure_ascii=False))

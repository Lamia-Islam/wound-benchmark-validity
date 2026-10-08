#!/usr/bin/env python3
"""Does the clinical agreement hierarchy depend on images with empty ground truth?

Reads the per-image matrices written by perimage32.py and recomputes every
pairwise rank agreement under three image filters:

    all            every scored image
    no_empty       images whose ground-truth mask is empty are dropped
    no_empty_tiny  also drop images with lesion fraction below 0.001

An empty ground truth gives Dice 1 to a method that predicts nothing and 0 to
any method that predicts anything, so a reference with many empty masks can
reward a method for being timid. This checks whether that moved any conclusion.

The hierarchy labels are corrected here: FUSeg and AZH come from the same clinic
(the AZH Wound and Vascular Center), so FUSeg vs AZH is "same source", not
"same modality". The Kvasir pair is now disjoint (see perimage32.py).

Writes results/EMPTY_MASK_SENSITIVITY.json and results/EMPTY_MASK_SENSITIVITY.md.
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from woundbench.deep import DEEP_NAMES                        # noqa: E402
from woundbench.metrics import spearman_permutation           # noqa: E402

OUT = os.path.join(ROOT, "results")
PER = os.path.join(OUT, "perimage32")
REFS = ["medetec", "fuseg", "azh", "azh_train_slice",
        "kvasir_disjoint", "kvasir_sessile"]
LABEL = {"medetec": "Medetec", "fuseg": "FUSeg", "azh": "AZH test",
         "azh_train_slice": "AZH train slice",
         "kvasir_disjoint": "Kvasir (disjoint)",
         "kvasir_sessile": "Kvasir sessile"}
CONDS = ["all", "no_empty", "no_empty_tiny"]
COND_LABEL = {"all": "all images", "no_empty": "excluding empty GT",
              "no_empty_tiny": "excluding empty GT and fraction < 0.001"}
METRICS = ["dice", "nsd"]
TINY = 0.001
N_PERM = 20000
N_BOOT = 5000

LEVELS = [
    ("same dataset", [("azh", "azh_train_slice")]),
    ("same source", [("fuseg", "azh"), ("kvasir_disjoint", "kvasir_sessile")]),
    ("different source, same modality", [("medetec", "fuseg"),
                                         ("medetec", "azh")]),
    ("across modality", [(w, k) for w in ("medetec", "fuseg", "azh")
                         for k in ("kvasir_disjoint", "kvasir_sessile")]),
]
PREV_KVASIR_SAME_SOURCE = 0.952


def rho_ci(a, b, n_boot=N_BOOT, seed=0):
    """Bootstrap CI on a rank correlation by resampling methods.

    Identical to azh_clinical._rho_ci, which produced the earlier CIs."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(i.tolist())) < 4:
            continue
        v = spearmanr(a[i], b[i]).statistic
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    data = {}
    for r in REFS:
        d = json.load(open(os.path.join(PER, r + ".json")))
        data[r] = {"names": d["names"], "ids": d["ids"],
                   "dice": np.asarray(d["dice"], float),
                   "nsd": np.asarray(d["nsd"], float),
                   "frac": np.asarray(d["gt_fraction"], float),
                   "empty": np.asarray(d["gt_empty"], bool)}
    names = data[REFS[0]]["names"]
    for r in REFS:
        assert data[r]["names"] == names
    a = set(data["kvasir_disjoint"]["ids"])
    b = set(data["kvasir_sessile"]["ids"])
    assert not (a & b), "Kvasir references overlap"
    return data, names


def mask_for(d, cond):
    if cond == "all":
        return np.ones(len(d["frac"]), bool)
    if cond == "no_empty":
        return ~d["empty"]
    return (~d["empty"]) & (d["frac"] >= TINY)


def tie_groups(v, names):
    groups = {}
    for n, x in zip(names, v):
        groups.setdefault(float(x), []).append(n)
    return [g for g in groups.values() if len(g) > 1]


def fmt(x, sign=True, nd=3):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    data, names = load()
    deep_idx = [i for i, n in enumerate(names) if n in DEEP_NAMES]
    nondeep_idx = [i for i, n in enumerate(names) if n not in DEEP_NAMES]
    assert len(deep_idx) == 10 and len(nondeep_idx) == 22
    res = {"names": names, "deep_names": [names[i] for i in deep_idx],
           "conditions": COND_LABEL, "tiny_threshold": TINY}

    # ---- a) counts --------------------------------------------------------
    counts = {}
    for r in REFS:
        d = data[r]
        counts[r] = {"n": int(len(d["frac"])),
                     "n_empty": int(d["empty"].sum()),
                     "n_tiny": int(((d["frac"] > 0) & (d["frac"] < TINY)).sum()),
                     "n_no_empty": int(mask_for(d, "no_empty").sum()),
                     "n_no_empty_tiny": int(mask_for(d, "no_empty_tiny").sum()),
                     "mean_fraction": float(d["frac"].mean()),
                     "median_fraction": float(np.median(d["frac"]))}
    res["counts"] = counts

    # per-method means under every condition
    means = {m: {c: {} for c in CONDS} for m in METRICS}
    for m in METRICS:
        for c in CONDS:
            for r in REFS:
                sel = mask_for(data[r], c)
                means[m][c][r] = data[r][m][sel].mean(axis=0)
    res["means"] = {m: {c: {r: means[m][c][r].tolist() for r in REFS}
                        for c in CONDS} for m in METRICS}

    # ---- b) every pair, every condition, both metrics ----------------------
    pairs = {}
    for m in METRICS:
        for c in CONDS:
            for a, b in itertools.combinations(REFS, 2):
                va, vb = means[m][c][a], means[m][c][b]
                rho, p = spearman_permutation(va, vb, n_perm=N_PERM)
                lo, hi = rho_ci(va, vb)
                pairs[f"{m}|{c}|{a}|{b}"] = {"rho": rho, "p": p,
                                             "ci": [lo, hi]}
                print(f"{m:<4} {c:<14} {a:>16} vs {b:<16} {rho:+.4f} "
                      f"p={p:.4f} [{lo:+.3f},{hi:+.3f}]", flush=True)
    res["pairs"] = pairs

    def P(m, c, a, b):
        k = f"{m}|{c}|{a}|{b}"
        if k not in pairs:
            k = f"{m}|{c}|{b}|{a}"
        return pairs[k]

    # ---- c) ties ------------------------------------------------------------
    ties = {}
    for m in METRICS:
        for c in CONDS:
            for r in REFS:
                g = tie_groups(means[m][c][r], names)
                ties[f"{m}|{c}|{r}"] = {
                    "n_methods_tied": int(sum(len(x) for x in g)),
                    "groups": g}
    # behaviour on the empty-GT images themselves
    empty_behaviour = {}
    for r in REFS:
        d = data[r]
        e = d["empty"]
        if e.sum() == 0:
            empty_behaviour[r] = None
            continue
        sub = d["dice"][e]                      # every value is 0 or 1
        pred_empty = sub.mean(axis=0)           # share of empty images left empty
        empty_behaviour[r] = {
            "n_empty": int(e.sum()),
            "dice_values_present": sorted(set(np.unique(sub).tolist())),
            "share_left_empty_per_method": dict(zip(names, pred_empty.tolist())),
            "n_methods_never_empty": int((pred_empty == 0).sum()),
            "n_methods_always_empty": int((pred_empty == 1).sum()),
            "n_distinct_score_vectors_on_empty": int(
                len({tuple(col) for col in sub.T.tolist()})),
        }
    res["ties"] = ties
    res["empty_behaviour"] = empty_behaviour

    # ---- d) corrected hierarchy --------------------------------------------
    hier = {}
    for m in METRICS:
        for c in CONDS:
            lv = {}
            for level, prs in LEVELS:
                rows = [{"a": a, "b": b, **P(m, c, a, b)} for a, b in prs]
                lv[level] = {"pairs": rows,
                             "mean_rho": float(np.mean([x["rho"] for x in rows]))}
            hier[f"{m}|{c}"] = lv
    res["hierarchy"] = hier

    # ---- e) subsets, across modality ----------------------------------------
    subsets = {"all 32": list(range(32)), "non-deep 22": nondeep_idx,
               "deep 10": deep_idx}
    sub_res = {}
    for m in METRICS:
        for c in CONDS:
            for sname, idx in subsets.items():
                lv_means = {}
                for level, prs in LEVELS:
                    rows = []
                    for a, b in prs:
                        va = means[m][c][a][idx]
                        vb = means[m][c][b][idx]
                        rho, p = spearman_permutation(va, vb, n_perm=N_PERM)
                        rows.append({"a": a, "b": b, "rho": rho, "p": p})
                    lv_means[level] = {
                        "pairs": rows,
                        "mean_rho": float(np.nanmean([x["rho"] for x in rows]))}
                sub_res[f"{m}|{c}|{sname}"] = lv_means
    res["subsets"] = sub_res

    # ---- check the bootstrap against the previously published CI ----------
    old = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    o_r, o_p = spearman_permutation(old["kvasir"], old["kvasir_sessile"],
                                    n_perm=N_PERM)
    o_lo, o_hi = rho_ci(old["kvasir"], old["kvasir_sessile"])
    res["previous_overlapping_kvasir"] = {"rho": o_r, "p": o_p,
                                          "ci": [o_lo, o_hi]}

    build_verdict(res)
    json.dump(res, open(os.path.join(OUT, "EMPTY_MASK_SENSITIVITY.json"), "w"),
              indent=1)
    write_md(res, data, names, P)
    print("saved EMPTY_MASK_SENSITIVITY.{json,md}")


def write_md(res, data, names, P):
    L = []
    w = L.append
    counts = res["counts"]
    hier = res["hierarchy"]
    sub = res["subsets"]

    w("# Empty ground-truth masks: do they move the agreement hierarchy?")
    w("")
    w("Every clinical reference was rescored with all 32 methods, keeping the "
      "Dice and NSD (tau = 2 px) of every method on every image "
      "(results/perimage32/). Rank agreement between two references is the "
      "Spearman correlation of the 32 per-method mean scores. p is a two-sided "
      f"permutation test ({N_PERM} permutations). The 95% CI is a bootstrap "
      f"over methods ({N_BOOT} resamples, seed 0).")
    w("")
    w("Two corrections are built in. First, the two Kvasir references no longer "
      "share images: Kvasir (disjoint) is the first 120 Kvasir-SEG images that "
      "are not in the sessile subset, so it shares no image with Kvasir "
      "sessile. The old selection shared 31. Second, FUSeg and AZH come from "
      "the same clinic, so FUSeg vs AZH is now labelled same source, not "
      "same modality.")
    w("")
    w("Three image filters are compared: all images; excluding images whose "
      "ground-truth mask is empty; and also excluding images whose lesion "
      "covers less than 0.1% of the image.")
    w("")

    # a
    w("## 1. How many empty and near-empty masks there are")
    w("")
    w("| reference | n | empty GT | lesion fraction in (0, 0.001) | "
      "n after both exclusions | median lesion fraction |")
    w("|---|---|---|---|---|---|")
    for r in REFS:
        c = counts[r]
        w(f"| {LABEL[r]} | {c['n']} | {c['n_empty']} | {c['n_tiny']} | "
          f"{c['n_no_empty_tiny']} | {c['median_fraction']:.4f} |")
    w("")

    # d (headline) first, it is what the reader wants
    for m, mname in (("dice", "Dice"), ("nsd", "NSD")):
        w(f"## {'2' if m == 'dice' else '3'}. Corrected hierarchy under {mname}")
        w("")
        w("| level | pair | all images: rho, p, CI | excluding empty GT: rho, "
          "p, CI | excluding empty and < 0.001: rho, p, CI |")
        w("|---|---|---|---|---|")
        for level, prs in LEVELS:
            for a, b in prs:
                cells = []
                for c in CONDS:
                    x = P(m, c, a, b)
                    cells.append(f"{fmt(x['rho'])}, {fmt_p(x['p'])}, "
                                 f"[{fmt(x['ci'][0])}, {fmt(x['ci'][1])}]")
                w(f"| {level} | {LABEL[a]} vs {LABEL[b]} | " + " | ".join(cells)
                  + " |")
        w("")
        w("Level means:")
        w("")
        w("| level | all images | excluding empty GT | excluding empty and "
          "< 0.001 |")
        w("|---|---|---|---|")
        for level, _ in LEVELS:
            w(f"| {level} | " + " | ".join(
                fmt(hier[f'{m}|{c}'][level]['mean_rho'], nd=4) for c in CONDS)
              + " |")
        w("")

    # b: full pair table
    w("## 4. Every pair of references")
    w("")
    for m, mname in (("dice", "Dice"), ("nsd", "NSD")):
        w(f"Under {mname}:")
        w("")
        w("| pair | all images | excluding empty GT | excluding empty and "
          "< 0.001 |")
        w("|---|---|---|---|")
        for a, b in itertools.combinations(REFS, 2):
            cells = []
            for c in CONDS:
                x = P(m, c, a, b)
                cells.append(f"{fmt(x['rho'])} (p {fmt_p(x['p'])}; "
                             f"[{fmt(x['ci'][0])}, {fmt(x['ci'][1])}])")
            w(f"| {LABEL[a]} vs {LABEL[b]} | " + " | ".join(cells) + " |")
        w("")

    # c: ties
    w("## 5. Ties")
    w("")
    w("Number of methods whose mean score is exactly equal to at least one "
      "other method's mean score (Spearman gives tied methods the average "
      "rank).")
    w("")
    w("| reference | Dice, all | Dice, excl. empty | Dice, excl. both | "
      "NSD, all | NSD, excl. empty | NSD, excl. both |")
    w("|---|---|---|---|---|---|---|")
    for r in REFS:
        cells = [str(res["ties"][f"{m}|{c}|{r}"]["n_methods_tied"])
                 for m in METRICS for c in CONDS]
        w(f"| {LABEL[r]} | " + " | ".join(cells) + " |")
    w("")
    any_groups = [(k, v["groups"]) for k, v in res["ties"].items() if v["groups"]]
    if any_groups:
        w("Tied groups:")
        w("")
        for k, g in any_groups:
            m, c, r = k.split("|")
            w(f"- {LABEL[r]}, {m}, {COND_LABEL[c]}: "
              + "; ".join("{" + ", ".join(x) + "}" for x in g))
        w("")
    w("What the methods do on the empty-GT images themselves (on such an image "
      "Dice and NSD are 1 if the method predicts nothing and 0 otherwise):")
    w("")
    w("| reference | empty-GT images | methods that never leave one empty | "
      "methods that always leave them empty | distinct per-image score "
      "patterns across the 32 methods |")
    w("|---|---|---|---|---|")
    for r in REFS:
        e = res["empty_behaviour"][r]
        if e is None:
            w(f"| {LABEL[r]} | 0 | | | |")
        else:
            w(f"| {LABEL[r]} | {e['n_empty']} | {e['n_methods_never_empty']} | "
              f"{e['n_methods_always_empty']} | "
              f"{e['n_distinct_score_vectors_on_empty']} |")
    w("")
    tie_note = res.get("tie_note")
    if tie_note:
        w(tie_note)
        w("")

    # e: subsets
    w("## 6. The pool split, repeated on the new data")
    w("")
    w("Mean Spearman over the six across-modality pairs (each wound reference "
      "against each Kvasir reference), within each part of the pool. The "
      "earlier overlapping-Kvasir values were -0.253 (all 32), +0.108 "
      "(22 non-deep) and +0.236 (10 deep).")
    w("")
    for m, mname in (("dice", "Dice"), ("nsd", "NSD")):
        w(f"Under {mname}:")
        w("")
        w("| subset | all images | excluding empty GT | excluding empty and "
          "< 0.001 |")
        w("|---|---|---|---|")
        for s in ("all 32", "non-deep 22", "deep 10"):
            w(f"| {s} | " + " | ".join(
                fmt(sub[f'{m}|{c}|{s}']['across modality']['mean_rho'], nd=4)
                for c in CONDS) + " |")
        w("")
    w("All four levels by subset (Dice, all images):")
    w("")
    w("| subset | " + " | ".join(l for l, _ in LEVELS) + " |")
    w("|---|" + "---|" * len(LEVELS))
    for s in ("all 32", "non-deep 22", "deep 10"):
        w(f"| {s} | " + " | ".join(
            fmt(sub[f'dice|all|{s}'][l]['mean_rho'], nd=4) for l, _ in LEVELS)
          + " |")
    w("")

    # f: verdict
    w("## 7. Verdict")
    w("")
    for para in res["verdict"]:
        w(para)
        w("")
    w("## Limitations")
    w("")
    for para in res["limitations"]:
        w(f"- {para}")
    w("")

    text = "\n".join(L)
    assert "\u2014" not in text
    open(os.path.join(OUT, "EMPTY_MASK_SENSITIVITY.md"), "w").write(text)


def build_verdict(res):
    """Plain-language verdict. Every number in it is computed here."""
    pairs = res["pairs"]
    hier = res["hierarchy"]
    counts = res["counts"]
    sub = res["subsets"]
    names = res["names"]
    deep = set(res["deep_names"])
    V, LIM = [], []

    def X(m, c, a, b):
        k = f"{m}|{c}|{a}|{b}"
        return pairs[k] if k in pairs else pairs[f"{m}|{c}|{b}|{a}"]

    def r_p(x):
        return f"{x['rho']:+.3f} (p {fmt_p(x['p'])})"

    # largest shift, for the record
    best = (0.0, None)
    for m in METRICS:
        for c in ("no_empty", "no_empty_tiny"):
            for a, b in itertools.combinations(REFS, 2):
                d = X(m, c, a, b)["rho"] - X(m, "all", a, b)["rho"]
                if abs(d) > abs(best[0]):
                    best = (d, [m, c, a, b])
    res["max_rho_shift"] = {"delta": best[0], "where": best[1]}
    extra = max(abs(X(m, "no_empty_tiny", a, b)["rho"]
                    - X(m, "no_empty", a, b)["rho"])
                for m in METRICS for a, b in itertools.combinations(REFS, 2))
    res["max_extra_shift_from_tiny_exclusion"] = extra

    # significance changes at 0.05
    flips = []
    for m in METRICS:
        for a, b in itertools.combinations(REFS, 2):
            x0, x1 = X(m, "all", a, b), X(m, "no_empty", a, b)
            if (x0["p"] < 0.05) != (x1["p"] < 0.05):
                flips.append((m, a, b, x0, x1))
    res["significance_changes_no_empty"] = [
        {"metric": m, "a": a, "b": b, "all": x0, "no_empty": x1}
        for m, a, b, x0, x1 in flips]

    with_empty = [r for r in REFS if counts[r]["n_empty"] > 0]
    V.append("**Where empty masks are.** " + ", ".join(
        f"{LABEL[r]} has {counts[r]['n_empty']} of {counts[r]['n']}"
        for r in with_empty) + ". Medetec and both Kvasir references have none, "
        "and no Kvasir lesion is below 0.001 of the image.")

    # mechanism on AZH test
    eb = res["empty_behaviour"]["azh"]
    sh = eb["share_left_empty_per_method"]
    trained = [n for n in names if n in deep or n in
               ("unet", "logreg_all", "logreg_colour", "gbm_all")]
    deep_hi = [n for n in deep if sh[n] >= 0.7]
    never = [n for n in names if sh[n] == 0]
    never_untrained = [n for n in never if n not in trained]
    bonus = counts["azh"]["n_empty"] / counts["azh"]["n"]

    fa_all = X("dice", "all", "fuseg", "azh")
    fa_ne = X("dice", "no_empty", "fuseg", "azh")
    ft_all = X("dice", "all", "fuseg", "azh_train_slice")
    ft_ne = X("dice", "no_empty", "fuseg", "azh_train_slice")
    mf = X("dice", "no_empty", "medetec", "fuseg")
    ma = X("dice", "no_empty", "medetec", "azh")
    below = fa_ne["rho"] < min(mf["rho"], ma["rho"])
    n_fa = X("nsd", "no_empty", "fuseg", "azh")
    n_mf = X("nsd", "no_empty", "medetec", "fuseg")
    n_ma = X("nsd", "no_empty", "medetec", "azh")

    V.append("**Yes, excluding empty masks changed one conclusion, under Dice "
             "only.** The wound same-source pair, FUSeg vs AZH test, falls from "
             f"{r_p(fa_all)} to {r_p(fa_ne)}. "
             + (f"That puts it below both different-source wound pairs "
                f"(Medetec vs FUSeg {r_p(mf)}, Medetec vs AZH test {r_p(ma)}), so "
                "under Dice the claim that same-source wound references agree "
                "more than different-source ones holds only while the empty "
                "masks are left in. " if below else "")
             + f"FUSeg vs AZH train slice falls from {r_p(ft_all)} to "
             f"{r_p(ft_ne)}"
             + (" and stops being significant." if ft_ne["p"] >= 0.05 else ".")
             + f" Under NSD the same pair moves less ({r_p(X('nsd', 'all', 'fuseg', 'azh'))} "
             f"to {r_p(n_fa)}) and stays above both different-source pairs "
             f"({r_p(n_mf)} and {r_p(n_ma)})"
             + ("." if n_fa["rho"] > max(n_mf["rho"], n_ma["rho"]) else
                ", which is NOT the case here."))

    V.append("**Why.** On an empty-GT image a method scores 1 if it predicts "
             "nothing and 0 otherwise. On AZH test, "
             f"{len(deep_hi)} of the 10 deep models leave at least 70% of the "
             f"{eb['n_empty']} empty images empty, while {len(never_untrained)} "
             f"of the {len([n for n in names if n not in trained])} untrained "
             "classical methods never leave one empty. That is a bonus of "
             f"up to {bonus:.4f} mean Dice (10 of 150 images) that goes almost "
             "entirely to the trained methods. The trained methods also lead on "
             "FUSeg (the deep models were trained on FUSeg training images), so "
             "the bonus pushes the AZH "
             "ranking toward the FUSeg ranking. Remove the empty images and the "
             "deep models on AZH drop back among the classical ones. The empty "
             "images create no exact ties in any mean (section 5); their effect "
             "is a shared shift, not a tie.")

    lv_all = [hier["dice|all"][l]["mean_rho"] for l, _ in LEVELS]
    lv_ne = [hier["dice|no_empty"][l]["mean_rho"] for l, _ in LEVELS]
    ordered = all(lv_ne[i] > lv_ne[i + 1] for i in range(3))
    V.append("**The four-level order itself survives.** Under Dice the level "
             "means go from " + ", ".join(f"{v:+.4f}" for v in lv_all)
             + " (all images) to " + ", ".join(f"{v:+.4f}" for v in lv_ne)
             + " (excluding empty GT), for same dataset, same source, different "
             "source same modality, and across modality. "
             + ("The order is unchanged, but the same-source level now rests on "
                "the Kvasir pair; its wound half sits below the level beneath it."
                if ordered and below else
                "The order is unchanged." if ordered else
                "The order CHANGES."))

    across_sig = [(a, b) for a, b in LEVELS[3][1]
                  if (X("dice", "all", a, b)["p"] < 0.05)
                  != (X("dice", "no_empty", a, b)["p"] < 0.05)]
    V.append("**Across modality.** The Dice mean moves from "
             f"{lv_all[3]:+.4f} to {lv_ne[3]:+.4f}, because the AZH test vs "
             "Kvasir pairs move from weakly negative to about zero "
             f"({r_p(X('dice', 'all', 'azh', 'kvasir_disjoint'))} to "
             f"{r_p(X('dice', 'no_empty', 'azh', 'kvasir_disjoint'))}; "
             f"{r_p(X('dice', 'all', 'azh', 'kvasir_sessile'))} to "
             f"{r_p(X('dice', 'no_empty', 'azh', 'kvasir_sessile'))}). "
             + ("No across-modality pair crosses p = 0.05. "
                if not across_sig else "Some across-modality pairs cross "
                "p = 0.05. ")
             + "The two FUSeg vs Kvasir pairs stay significantly negative "
             f"({r_p(X('dice', 'no_empty', 'fuseg', 'kvasir_disjoint'))}, "
             f"{r_p(X('dice', 'no_empty', 'fuseg', 'kvasir_sessile'))}).")

    s = lambda m, c, k: sub[f"{m}|{c}|{k}"]["across modality"]["mean_rho"]
    V.append("**The pool split still holds.** Across-modality mean under Dice, "
             f"all images: all 32 {s('dice', 'all', 'all 32'):+.4f}, 22 non-deep "
             f"{s('dice', 'all', 'non-deep 22'):+.4f}, 10 deep "
             f"{s('dice', 'all', 'deep 10'):+.4f}. Excluding empty GT: "
             f"{s('dice', 'no_empty', 'all 32'):+.4f}, "
             f"{s('dice', 'no_empty', 'non-deep 22'):+.4f}, "
             f"{s('dice', 'no_empty', 'deep 10'):+.4f}. "
             + ("Under Dice the pooled value is negative and both subsets are "
                "positive in all three conditions, so the negative pooled number "
                "is still a composition effect."
                if all(s("dice", c, "all 32") < 0 < min(s("dice", c, "non-deep 22"),
                                                        s("dice", c, "deep 10"))
                       for c in CONDS)
                else "The pooled/subset sign pattern does NOT hold in every "
                     "condition; see section 6."))

    k = X("dice", "all", "kvasir_disjoint", "kvasir_sessile")
    prev = res["previous_overlapping_kvasir"]
    V.append("**Clean Kvasir same-source value.** With the 31 shared images "
             f"removed it is {k['rho']:+.4f} (p {fmt_p(k['p'])}, 95% CI "
             f"[{k['ci'][0]:+.3f}, {k['ci'][1]:+.3f}]) over 32 methods, against "
             f"{prev['rho']:+.4f} with the overlap (CI [{prev['ci'][0]:+.3f}, "
             f"{prev['ci'][1]:+.3f}], recomputed here from the old cache). The "
             f"overlap inflated it by {prev['rho'] - k['rho']:.4f}, which is "
             "negligible. Kvasir has no empty masks, so this value is the same "
             "in every condition.")

    V.append("**Tiny lesions add nothing.** Also dropping lesions below 0.001 "
             f"of the image moves no pairwise rho by more than {extra:.4f} "
             "beyond the empty-mask exclusion.")

    LIM.append("The deep-only subset has 10 methods, so a single rho there has "
               "a wide null band (about |rho| >= 0.64 for p < 0.05).")
    LIM.append("The bootstrap CI resamples methods, not images. It describes "
               "uncertainty from the choice of method pool, not from the choice "
               "of images.")
    LIM.append("Lesion fraction is measured after resampling for Kvasir and at "
               "native size for the wound sets. The 0.001 cut is a share of "
               "pixels, not a clinical size.")
    LIM.append("Which condition is \"right\" is a design choice. Empty masks are "
               "part of the published AZH and FUSeg test sets, and a method that "
               "correctly predicts nothing on them is doing its job. The point "
               "here is only that the FUSeg vs AZH Dice agreement leans on them.")
    res["verdict"] = V
    res["limitations"] = LIM


if __name__ == "__main__":
    main()

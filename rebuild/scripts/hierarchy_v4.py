#!/usr/bin/env python3
"""The agreement hierarchy over eleven clinical references in four modalities.

Reads results/perimage32/<ref>.json (per-image Dice and NSD for all 32 methods)
and computes, for every pair of references, the Spearman rank agreement of
per-method mean scores.

References, sources and modalities:

  wound photography   medetec (Medetec archive)
                      fuseg, azh (AZH Wound and Vascular Center; one source)
                      azh_train_slice (disjoint slice of azh; control only)
  colonoscopy         kvasir_disjoint, kvasir_sessile (Kvasir; one source,
                      no shared images)
                      cvc_clinicdb, cvc_colondb (CVC Barcelona; one group,
                      different collections)
                      etis_larib (ETIS, Lariboisiere Paris)
  dermoscopy          ph2 (Porto)
  thyroid ultrasound  tn3k_test

Pair levels:
  same dataset                  azh vs azh_train_slice
  same source                   two references from one clinic or group
  different source, same mod.   same modality, different source
  across modality               different modality

The level comparison is tested with a reference-level permutation test: the
modality labels of the ten non-control references are permuted over all
distinct relabelings, and the statistic is the mean same-modality agreement
minus the mean across-modality agreement. Pairs share references, so testing
at the pair level would overstate the evidence; permuting labels respects that.

Writes results/hierarchy_v4.json and results/HIERARCHY_V4.md.
"""

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402
from woundbench.metrics import spearman_permutation              # noqa: E402

PER = os.path.join(ROOT, "results", "perimage32")
OUT = os.path.join(ROOT, "results")

REFS = {
    # name: (modality, source)
    "medetec": ("wound", "Medetec"),
    "co2wounds": ("wound", "CO2Wounds"),
    "fuseg": ("wound", "AZH"),
    "azh": ("wound", "AZH"),
    "kvasir_disjoint": ("colonoscopy", "Kvasir"),
    "kvasir_sessile": ("colonoscopy", "Kvasir"),
    "cvc_clinicdb": ("colonoscopy", "CVC"),
    "cvc_colondb": ("colonoscopy", "CVC"),
    "etis_larib": ("colonoscopy", "ETIS"),
    "ph2": ("dermoscopy", "PH2"),
    "tn3k_test": ("ultrasound", "TN3K"),
}
CONTROL = ("azh", "azh_train_slice")
LEVELS = ["same dataset", "same source", "different source, same modality",
          "across modality"]


def load():
    data = {}
    for name in list(REFS) + ["azh_train_slice"]:
        d = json.load(open(os.path.join(PER, name + ".json")))
        data[name] = d
    names = data["medetec"]["names"]
    for d in data.values():
        assert d["names"] == names
    return data, names


def means(d, metric, exclude_empty=False):
    m = np.asarray(d[metric], float)
    if exclude_empty:
        keep = ~np.asarray(d["gt_empty"], bool)
        m = m[keep]
    return m.mean(axis=0)


def boot_ci(a, b, B=5000, seed=0):
    """Bootstrap CI of Spearman over methods, vectorised."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    ra = rankdata(a[idx], axis=1)
    rb = rankdata(b[idx], axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    r = (ra * rb).sum(1)[ok] / den[ok]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def level_of(a, b):
    if {a, b} == set(CONTROL):
        return "same dataset"
    ma, sa = REFS[a]
    mb, sb = REFS[b]
    if ma != mb:
        return "across modality"
    if sa == sb:
        return "same source"
    return "different source, same modality"


def pair_table(vec, names_idx=None, perm=True, ci=True):
    out = []
    keys = list(REFS)
    pairs = list(itertools.combinations(keys, 2)) + [CONTROL]
    for a, b in pairs:
        va, vb = vec[a], vec[b]
        if names_idx is not None:
            va, vb = va[names_idx], vb[names_idx]
        if perm:
            r, p = spearman_permutation(va, vb, n_perm=20000)
        else:
            r, p = float(spearmanr(va, vb).statistic), None
        row = {"a": a, "b": b, "level": level_of(a, b), "rho": float(r),
               "p": None if p is None else float(p)}
        if ci:
            row["ci"] = boot_ci(va, vb)
        out.append(row)
    return out


def level_means(rows):
    return {L: (float(np.mean([r["rho"] for r in rows if r["level"] == L]))
                if any(r["level"] == L for r in rows) else None,
                sum(1 for r in rows if r["level"] == L))
            for L in LEVELS}


def label_permutation_test(vec, names_idx=None):
    """Exact test over all distinct relabelings of modality across references.

    Statistic: mean agreement over same-modality pairs minus mean agreement
    over across-modality pairs, control pair excluded.
    """
    keys = list(REFS)
    labels = [REFS[k][0] for k in keys]
    R = np.zeros((len(keys), len(keys)))
    for i, j in itertools.combinations(range(len(keys)), 2):
        va, vb = vec[keys[i]], vec[keys[j]]
        if names_idx is not None:
            va, vb = va[names_idx], vb[names_idx]
        R[i, j] = R[j, i] = spearmanr(va, vb).statistic
    iu = np.triu_indices(len(keys), 1)

    def stat(lab):
        lab = np.asarray(lab)
        same = (lab[:, None] == lab[None, :])[iu]
        return R[iu][same].mean() - R[iu][~same].mean()

    obs = stat(labels)
    # every distinct assignment of the observed labels to references
    from sympy.utilities.iterables import multiset_permutations
    vals = np.asarray([stat(lab) for lab in multiset_permutations(labels)])
    p = float((np.sum(vals >= obs - 1e-12)) / len(vals))
    return float(obs), p, len(vals)


def main():
    data, names = load()
    # the confound is training, not architecture: the three pixel classifiers
    # and the small U-Net are trained on FUSeg too, so they belong with the
    # ten deep models
    trained = lambda n: (ALL_METHOD_FAMILY[n] in ("learned", "CNN")
                         or ALL_METHOD_FAMILY[n].startswith("deep"))
    deep_idx = np.array([i for i, n in enumerate(names) if trained(n)])
    other_idx = np.array([i for i, n in enumerate(names) if not trained(n)])
    assert len(deep_idx) == 14 and len(other_idx) == 18

    results = {}
    md = []
    md.append("# The agreement hierarchy over twelve references\n")
    md.append("Eleven references in four modalities plus a within-dataset control, "
              "32 methods. Spearman of per-method mean score.\n")

    # reference summary
    md.append("## References\n")
    md.append("| reference | modality | source | n | empty GT | median lesion fraction |")
    md.append("|---|---|---|---|---|---|")
    for k in list(REFS) + ["azh_train_slice"]:
        d = data[k]
        mod, src = REFS.get(k, ("wound", "AZH"))
        md.append(f"| {k} | {mod} | {src} | {len(d['ids'])} | {sum(d['gt_empty'])} | "
                  f"{np.median(d['gt_fraction']):.3f} |")
    md.append("")

    for metric in ("dice", "nsd"):
        for excl in (False, True):
            tag = f"{metric}_{'noempty' if excl else 'all'}"
            vec = {k: means(data[k], metric, excl) for k in data}
            rows = pair_table(vec, perm=True, ci=(not excl))
            lm = level_means(rows)
            sub = {}
            for sname, idx in (("trained", deep_idx), ("untrained", other_idx)):
                srows = pair_table(vec, names_idx=idx, perm=False, ci=False)
                sub[sname] = level_means(srows)
            test = label_permutation_test(vec)
            test_other = label_permutation_test(vec, other_idx)    # untrained 18
            test_deep = label_permutation_test(vec, deep_idx)      # trained 14
            results[tag] = {"pairs": rows, "levels": lm, "subsets": sub,
                            "modality_test": {"all": test, "untrained": test_other,
                                              "trained": test_deep}}

            md.append(f"## {metric.upper()}, "
                      f"{'excluding empty ground truth' if excl else 'all images'}\n")
            md.append("| level | pairs | mean rho, all 32 | 18 untrained | 14 trained |")
            md.append("|---|---|---|---|---|")
            for L in LEVELS:
                m, n = lm[L]
                mo = sub["untrained"][L][0]
                mdp = sub["trained"][L][0]
                fmt = lambda x: "n/a" if x is None else f"{x:+.3f}"
                md.append(f"| {L} | {n} | {fmt(m)} | {fmt(mo)} | {fmt(mdp)} |")
            md.append("")
            md.append(f"Modality permutation test (same-modality minus across-modality "
                      f"mean agreement, exact over {test[2]} relabelings): "
                      f"all 32 {test[0]:+.3f} (p {test[1]:.4f}); "
                      f"18 untrained {test_other[0]:+.3f} (p {test_other[1]:.4f}); "
                      f"14 trained {test_deep[0]:+.3f} (p {test_deep[1]:.4f}).\n")

            if not excl:
                md.append("| pair | level | rho | p | 95% CI |")
                md.append("|---|---|---|---|---|")
                for r in sorted(rows, key=lambda r: (LEVELS.index(r["level"]),
                                                     -r["rho"])):
                    md.append(f"| {r['a']} vs {r['b']} | {r['level']} | "
                              f"{r['rho']:+.3f} | {r['p']:.4f} | "
                              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] |")
                md.append("")

            print(f"\n=== {tag} ===", flush=True)
            for L in LEVELS:
                m, n = lm[L]
                print(f"  {L:<34} n={n:>2}  all {m:+.3f}  "
                      f"untrained {sub['untrained'][L][0]:+.3f}  trained {sub['trained'][L][0]:+.3f}",
                      flush=True)
            print(f"  modality test: all {test[0]:+.3f} p {test[1]:.4f} | "
                  f"untrained {test_other[0]:+.3f} p {test_other[1]:.4f} | "
                  f"trained {test_deep[0]:+.3f} p {test_deep[1]:.4f}", flush=True)

    # where the across-modality structure comes from: split pairs by whether
    # they cross the training-domain boundary (wound photography)
    def tcat(a, b):
        wa, wb = REFS[a][0] == "wound", REFS[b][0] == "wound"
        if wa and wb:
            return "wound x wound"
        if wa != wb:
            return "wound x non-wound"
        if REFS[a][0] == REFS[b][0]:
            return "non-wound, same modality"
        return "non-wound, across modality"
    TC = ["wound x wound", "non-wound, same modality",
          "non-wound, across modality", "wound x non-wound"]
    results["training_domain_split"] = {}
    for metric in ("dice", "nsd"):
        for excl in (False, True):
            tag = f"{metric}_{'noempty' if excl else 'all'}"
            vec = {k: means(data[k], metric, excl) for k in REFS}
            md.append(f"## Pairs split by the training-domain boundary, {metric.upper()}, "
                      f"{'excluding empty ground truth' if excl else 'all images'}\n")
            md.append("| category | pairs | all 32 | 18 untrained | 14 trained |")
            md.append("|---|---|---|---|---|")
            res = {}
            for C in TC:
                pr = [(a, b) for a, b in itertools.combinations(list(REFS), 2)
                      if tcat(a, b) == C]
                v = {}
                for g, idx in (("all", None), ("untrained", other_idx),
                               ("trained", deep_idx)):
                    rs = []
                    for a, b in pr:
                        va, vb = vec[a], vec[b]
                        if idx is not None:
                            va, vb = va[idx], vb[idx]
                        rs.append(spearmanr(va, vb).statistic)
                    v[g] = float(np.mean(rs))
                res[C] = {"n": len(pr), **v}
                md.append(f"| {C} | {len(pr)} | {v['all']:+.3f} | "
                          f"{v['untrained']:+.3f} | {v['trained']:+.3f} |")
            md.append("")
            results["training_domain_split"][tag] = res

    # mean Dice by subset per reference, the composition mechanism
    md.append("## Mean Dice by subset on each reference\n")
    md.append("| reference | modality | 14 trained | 18 untrained | gap |")
    md.append("|---|---|---|---|---|")
    comp = {}
    for k in list(REFS):
        v = means(data[k], "dice")
        dmn, omn = float(v[deep_idx].mean()), float(v[other_idx].mean())
        comp[k] = {"trained": dmn, "untrained": omn}
        md.append(f"| {k} | {REFS[k][0]} | {dmn:.3f} | {omn:.3f} | {dmn - omn:+.3f} |")
    md.append("")
    results["composition"] = comp

    json.dump(results, open(os.path.join(OUT, "hierarchy_v4.json"), "w"), indent=1)
    open(os.path.join(OUT, "HIERARCHY_V4.md"), "w").write("\n".join(md))
    print("\nsaved results/hierarchy_v4.json and results/HIERARCHY_V4.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Writes novelty/RESULTS_P1.md from novelty/results/p1.json (no hand-typed numbers)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(ROOT, "novelty", "results", "p1.json")))
M = "−"


def f(x, d=2):
    if x is None or x != x:
        return "n/a"
    s = f"{x:.{d}f}"
    return s.replace("-", M) if s.startswith("-") else s


def ci(c, d=2):
    return f"{f(c[0], d)} to {f(c[1], d)}"


L = ["# Phase 1 results: competence score (generated from novelty/results/p1.json)", ""]
L.append(f"Fit on {R['n_fit']} FUSeg training images (class sizes {', '.join(map(str, R['class_counts']))}); "
         f"threshold from {R['n_threshold_images']} held-out FUSeg training images: {f(R['threshold'])}. "
         f"ResNet-50 embedding: {R['resnet50_embedding']}. CO2Wounds-V2: {R['co2wounds']}.")
L += ["", "## Score by evaluation set", "", "| Set | Images | Median score | Share below threshold |", "|---|---|---|---|"]
for s, v in R["score_summary"].items():
    L.append(f"| {s} | {v['n']} | {f(v['median'])} | {f(100 * v['share_below_threshold'], 1)}% |")
L += ["", "## H1: failure prediction (95% unit-bootstrap intervals)", "",
      "| Set | Non-empty | Majority misses | Score AUROC | Black share AUROC | Lesion fraction AUROC | Score Spearman | Partial Spearman (score given black share) |",
      "|---|---|---|---|---|---|---|---|"]
for g in ("pooled", "medetec", "fuseg", "azh", "azh_train_slice"):
    v = R["H1"][g]
    st = v["stats"]

    def a(k):
        return f"{f(st[k]['est'])} ({ci(st[k]['ci'])})" if v["auroc_reported"] else "not reported (too few misses)"
    L.append(f"| {g} | {v['n_nonempty']} | {v['n_majority_miss']} | {a('score|auroc')} | {a('black_share|auroc')} | "
             f"{a('lesion_fraction|auroc')} | {f(st['score|spearman']['est'])} ({ci(st['score|spearman']['ci'])}) | "
             f"{f(st['partial_spearman_score_given_black|est']['est'])} ({ci(st['partial_spearman_score_given_black|est']['ci'])}) |")
h = R["H1"]["verdict"]
L += ["", f"H1 verdict by the pre-registered rule: {'supported' if h['supported'] else 'not supported'} "
      f"(pooled AUROC lower bound {f(h['auroc_lower_ci'])}; partial Spearman interval {ci(h['partial_ci'])})."]
H2 = R["H2"]
L += ["", "## H2: competence-filtered agreement (M19, wound pairs among the three available wound datasets)", "",
      f"Images removed by the threshold: {', '.join(f'{k} {v} of {H2['n_images'][k]}' for k, v in H2['n_removed'].items())}.",
      f"Agreement with all images {f(H2.get('agree_all'))}; filtered {f(H2.get('agree_filtered'))}; change {f(H2.get('change'))} "
      f"({ci(H2.get('change_ci', [None, None]))}); random-removal control 95th percentile {f(H2.get('control_p95'))}.",
      f"H2 verdict by the pre-registered rule: {'supported' if H2['verdict']['supported'] else 'not supported'}."]
if any(H2["n_removed"][k] == H2["n_images"][k] for k in H2["n_removed"]):
    L.append("The threshold removes every image of at least one wound dataset, so filtered agreement is undefined.")
H3 = R["H3"]
L += ["", "## H3: benchmark shift (exploratory)", "",
      f"{H3['M19']['n_sets']} synthetic sets regenerated (texture of the regenerated sets differs from the stored values by at most "
      f"{f(H3['regeneration_check_texture_max_abs_diff'], 4)}).", "",
      "| Method set | Sets | Spearman(mean score, benchmark agreement) | Spearman(Frechet distance, benchmark agreement) |", "|---|---|---|---|"]
for p, lab in (("M19", "M19"), ("UM19", "U+M19")):
    L.append(f"| {lab} | {H3[p]['n_sets']} | {f(H3[p]['spearman_mean_score_vs_bench'])} | {f(H3[p]['spearman_frechet_vs_bench'])} |")
L += ["", "## Note on the score (observation, not a change to the pre-registered analysis)", "",
      "The FUSeg training images all have the same height and width, so the size features have zero variance in the fit. "
      "After standardisation and the variance floor, any image of another size receives a very low score. The score therefore "
      "flags every Medetec, AZH and control image, and its pooled ranking mixes image size with the other features."]
txt = "\n".join(L) + "\n"
assert "—" not in txt
open(os.path.join(ROOT, "novelty", "RESULTS_P1.md"), "w").write(txt)
print("written")

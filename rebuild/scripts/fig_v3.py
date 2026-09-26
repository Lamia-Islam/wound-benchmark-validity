#!/usr/bin/env python3
"""Figures 1, 2, 3 and 5 for manuscript V3.

Monochrome throughout. Identity is carried by position, marker shape and direct
labels, never by colour alone, so the figures survive greyscale printing and
colour-vision deficiency without a palette check.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from woundbench.deep import DEEP_NAMES                            # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
INK, SEC, MUT, BAND, SURF = "#111111", "#555555", "#8A8A8A", "#EDEDED", "#FFFFFF"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUT)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=SEC, labelsize=8.5, length=3)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1
def fig1():
    d = json.load(open(os.path.join(OUT, "hierarchy32_pairs.json")))
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=SEC, lw=1.4,
                solid_capstyle="butt")
        ax.plot([r["lo"], r["lo"]], [yi - .14, yi + .14], color=SEC, lw=1.4)
        ax.plot([r["hi"], r["hi"]], [yi - .14, yi + .14], color=SEC, lw=1.4)
        filled = r["p"] < 0.05
        ax.plot(r["rho"], yi, "o", ms=7, color=INK if filled else SURF,
                markeredgecolor=INK, markeredgewidth=1.5, zorder=3)
    ax.axvline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5, color=INK)
    # group bands
    prev, start = rows[0]["cls"], 0
    for i, r in enumerate(rows + [{"cls": None}]):
        if r["cls"] != prev:
            if prev == "across modality":
                ax.axhspan(y[i - 1] - .5, y[start] + .5, color=BAND, zorder=0,
                           lw=0)
            ax.text(1.02, (y[start] + y[i - 1]) / 2, prev, fontsize=8,
                    color=SEC, va="center", ha="left")
            prev, start = r["cls"], i
    ax.set_xlim(-0.85, 1.0)
    ax.set_xlabel("Spearman rank agreement over 32 methods", fontsize=9.5,
                  color=INK, labelpad=7)
    ax.set_title("Agreement between clinical references falls in steps",
                 fontsize=11, color=INK, loc="left", pad=22)
    ax.text(0, 1.02, "filled marker = p < 0.05;  bars are 95% bootstrap "
            "intervals over methods", transform=ax.transAxes, fontsize=8.5,
            color=SEC, va="bottom")
    style(ax); ax.grid(axis="x", color="#F1F1F1", lw=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_hierarchy.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight"); print("saved " + p)


# ---------------------------------------------------------------- Figure 2
def fig2():
    c = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    names = c["__method_names"]
    dp = [i for i, n in enumerate(names) if n in DEEP_NAMES]
    cl = [i for i, n in enumerate(names) if n not in DEEP_NAMES]
    sub = json.load(open(os.path.join(OUT, "subset_agreement.json")))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURF)

    ax = axes[0]; ax.set_facecolor(SURF)
    labels = ["all 32", "22 others", "10 deep"]
    same = [sub["all"]["same"], sub["classical"]["same"], sub["deep"]["same"]]
    acr = [sub["all"]["across"], sub["classical"]["across"], sub["deep"]["across"]]
    x = np.arange(3)
    ax.bar(x - 0.19, same, 0.34, color=INK, zorder=3)
    ax.bar(x + 0.19, acr, 0.34, color=SURF, edgecolor=INK, hatch="////",
           linewidth=1.2, zorder=3)
    ax.axhline(0, color=MUT, lw=1.0)
    for xi, (s, a) in enumerate(zip(same, acr)):
        ax.text(xi - 0.19, s + 0.02, f"{s:+.3f}", ha="center", fontsize=8,
                color=INK)
        ax.text(xi + 0.19, a + (0.02 if a > 0 else -0.05), f"{a:+.3f}",
                ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_ylabel("mean Spearman", fontsize=9.5, color=INK)
    ax.set_ylim(-0.36, 0.95)
    ax.text(0.02, 0.95, "solid = same modality\nhatched = across modality",
            transform=ax.transAxes, fontsize=8, color=SEC, va="top",
            linespacing=1.4)
    ax.set_title("Only the mixed pool goes negative", fontsize=10, color=INK,
                 loc="left", pad=10)
    style(ax); ax.grid(axis="y", color="#F1F1F1", lw=0.9)

    ax = axes[1]; ax.set_facecolor(SURF)
    refs = ["medetec", "fuseg", "azh", "kvasir", "kvasir_sessile"]
    disp = ["Medetec", "FUSeg", "AZH", "Kvasir", "Kvasir\nsessile"]
    md = [np.mean([c[r][i] for i in dp]) for r in refs]
    mc = [np.mean([c[r][i] for i in cl]) for r in refs]
    x = np.arange(5)
    ax.axvspan(2.5, 4.5, color=BAND, zorder=0, lw=0)
    ax.plot(x, md, "o-", color=INK, lw=2, ms=7, zorder=3, label="10 deep")
    ax.plot(x, mc, "s--", color=SEC, lw=1.6, ms=6, zorder=3,
            markerfacecolor=SURF, markeredgewidth=1.4, label="22 others")
    ax.set_xticks(x); ax.set_xticklabels(disp, fontsize=8.5, color=INK)
    ax.set_ylabel("mean Dice", fontsize=9.5, color=INK)
    ax.text(3.5, 0.60, "colonoscopy", fontsize=8.5, color=SEC, ha="center")
    ax.text(1.0, 0.60, "wound photography", fontsize=8.5, color=SEC,
            ha="center")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_ylim(-0.02, 0.68)
    ax.set_title("The two groups exchange places between modalities",
                 fontsize=10, color=INK, loc="left", pad=10)
    style(ax); ax.grid(axis="y", color="#F1F1F1", lw=0.9)

    fig.tight_layout()
    p = os.path.join(OUT, "fig2_composition.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight"); print("saved " + p)


# ---------------------------------------------------------------- Figure 3
def fig3():
    d = json.load(open(os.path.join(OUT, "benchmark_quality32.json")))
    ceil = d["ceiling"]["rho"]
    rows = sorted(d["rows"], key=lambda r: r["texture"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    x = [r["texture"] for r in rows]
    y = [r["mean_rho"] for r in rows]
    ax.axhline(ceil, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax.text(x[0], ceil + 0.014, f"clinical-to-clinical ceiling  {ceil:+.3f}",
            fontsize=8.5, color=INK, ha="left")
    ax.axhspan(0, ceil, color="#F7F7F7", zorder=0, lw=0)
    ax.plot(x, y, "o-", color=INK, lw=2.2, ms=7, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.5)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi / ceil:.0%}", (xi, yi), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=8.5, color=SEC)
    ax.plot([x[-1]], [0.98 * ceil], "^", ms=9, color=SURF,
            markeredgecolor=SEC, markeredgewidth=1.5, zorder=3)
    ax.annotate("same generator, 22-method pool,\nFUSeg included: 98%",
                (x[-1], 0.98 * ceil), textcoords="offset points",
                xytext=(-14, -8), ha="right", va="top", fontsize=8.5, color=SEC,
                linespacing=1.35)
    ax.set_xlabel("measured background texture of the benchmark", fontsize=9.5,
                  color=INK, labelpad=7)
    ax.set_ylabel("agreement with Medetec and AZH", fontsize=9.5, color=INK)
    ax.set_ylim(0, 0.70)
    ax.set_title("Benchmark validity depends on the method pool it is measured "
                 "over", fontsize=10.5, color=INK, loc="left", pad=12)
    style(ax); ax.grid(axis="y", color="#F1F1F1", lw=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_pool_dependence.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight"); print("saved " + p)


# ---------------------------------------------------------------- Figure 5
def fig5():
    rows = [json.loads(l) for l in
            open(os.path.join(OUT, "nonreduce32_cells.jsonl")) if l.strip()]
    rows.sort(key=lambda r: r["texture"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    short = {"generator A, uniform drape": "A, uniform",
             "generator C, no noise": "C, no noise",
             "generator C, noise 3.4": "C, noise 3.4",
             "generator B, texture ablated": "B, ablated"}
    for r in rows:
        ax.plot(r["texture"], r["mean_rho"], "o", ms=9, color=INK,
                markeredgecolor=SURF, markeredgewidth=1.5, zorder=4)
        ax.annotate(short.get(r["name"], r["name"]),
                    (r["texture"], r["mean_rho"]), textcoords="offset points",
                    xytext=(9, -3), fontsize=8.5, color=INK)
    pairs = [(("generator C, noise 3.4", "generator B, texture ablated"),
              "matched to 0.03,\ndiffer by 0.115"),
             (("generator A, uniform drape", "generator C, no noise"),
              "matched to 0.24,\ndiffer by 0.160")]
    by = {r["name"]: r for r in rows}
    for (a, b), note in pairs:
        if a not in by or b not in by:
            continue
        xa, ya = by[a]["texture"], by[a]["mean_rho"]
        xb, yb = by[b]["texture"], by[b]["mean_rho"]
        xm = (xa + xb) / 2
        ax.plot([xm, xm], [ya, yb], color=SEC, lw=1.2, zorder=2)
        ax.plot([xa, xm], [ya, ya], color=SEC, lw=1.0, ls=":", zorder=2)
        ax.plot([xb, xm], [yb, yb], color=SEC, lw=1.0, ls=":", zorder=2)
        ax.annotate(note, (xm, (ya + yb) / 2), textcoords="offset points",
                    xytext=(-10, 0), ha="right", va="center", fontsize=8,
                    color=SEC, linespacing=1.35)
    ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("measured background texture statistic", fontsize=9.5,
                  color=INK, labelpad=7)
    ax.set_ylabel("agreement with Medetec and AZH", fontsize=9.5, color=INK)
    ax.set_xlim(1.85, 3.75)
    ax.set_title("Benchmarks matched on the statistic do not agree",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    style(ax); ax.grid(axis="y", color="#F1F1F1", lw=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_nonreducibility.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight"); print("saved " + p)


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "5"]
    for w in want:
        {"1": fig1, "2": fig2, "3": fig3, "5": fig5}[w]()

#!/usr/bin/env python3
"""Figure 4: the framing curve from two references, under two metrics.

Two panels, one per metric. Each shows the AZH-built curve (the reference no
method was trained on) and the FUSeg-built curve, with the held-out clinical
datasets as open predicted and filled measured markers on the AZH curve.
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
INK, SEC, MUT, SURF = "#111111", "#555555", "#8A8A8A", "#FFFFFF"

az = json.load(open(os.path.join(OUT, "framing_azh.json")))
fu = json.load(open(os.path.join(OUT, "framing32_slope.json")))

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200, sharey=True)
fig.patch.set_facecolor(SURF)
for ax, metric, title in zip(axes, ("dice", "nsd"),
                             ("Dice", "normalised surface Dice, 2 px")):
    ax.set_facecolor(SURF)
    a, f = az[metric], fu[metric]
    ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
    ax.plot(f["fractions"], f["curve"], "s--", color=SEC, lw=1.4, ms=5,
            markerfacecolor=SURF, markeredgewidth=1.3, zorder=2,
            label="built from FUSeg crops")
    ax.plot(a["fractions"], a["curve"], "o-", color=INK, lw=2.2, ms=6,
            markeredgecolor=SURF, markeredgewidth=1.4, zorder=3,
            label="built from AZH crops")
    o = a["overlay"]["medetec"]
    ax.plot([o["fraction"]] * 2, [o["predicted"], o["measured"]], color=SEC,
            lw=1.0, zorder=4)
    ax.plot(o["fraction"], o["predicted"], "D", ms=8, mfc=SURF, mec=SEC,
            mew=1.4, zorder=5)
    ax.plot(o["fraction"], o["measured"], "D", ms=8, mfc=INK, mec=SURF,
            mew=1.5, zorder=6)
    ax.annotate("Medetec\nheld out", (o["fraction"], o["measured"]),
                textcoords="offset points", xytext=(10, 2), fontsize=8,
                color=INK, linespacing=1.35)
    ax.set_xscale("log")
    ax.set_xlim(0.006, 0.42)
    ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.20, 0.40])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10", "0.20", "0.40"])
    ax.set_xlabel("lesion fraction of the clinical reference", fontsize=9.5,
                  color=INK, labelpad=6)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    mono = min(a["mono"])
    ax.text(0.02, 0.06, f"AZH curve: Spearman {mono:+.3f}\nin all four seeds",
            transform=ax.transAxes, fontsize=8, color=SEC, linespacing=1.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUT); ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=SEC, labelsize=8.5, length=3)
    ax.grid(axis="y", color="#F1F1F1", lw=0.9); ax.set_axisbelow(True)

axes[0].set_ylabel("slope of rank agreement\nagainst contrast, per 10 sat units",
                   fontsize=9.5, color=INK)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("A reference's design verdict is set by how much of the frame its "
             "lesions fill", fontsize=11, color=INK, x=0.012, ha="left", y=1.02)
fig.tight_layout()
p = os.path.join(OUT, "fig4_framing_both.png")
fig.savefig(p, facecolor=SURF, bbox_inches="tight"); print("saved " + p)

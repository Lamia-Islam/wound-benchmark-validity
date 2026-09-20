#!/usr/bin/env python3
"""Figure: the clinical reference's verdict as a function of its framing.

One panel. X is the reference's lesion fraction on a log axis, because the
clinical range spans more than an order of magnitude. Y is the contrast verdict,
a Spearman bounded at +/-1, so the axis is fixed to that range and zero is drawn
as the polarity midpoint rather than as an ordinary gridline.

Encoding is monochrome and identity is carried by marker shape and direct
labels, never by colour alone. The eight crop levels are the curve. The three
real clinical datasets are overlaid: an open marker where the curve predicts
them, a filled marker where they were measured, joined by a thin residual stem.

Writes results/fig_framing_law.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")

INK = "#111111"
SECONDARY = "#555555"
MUTED = "#8A8A8A"
BAND = "#EDEDED"
SURFACE = "#FFFFFF"

MARKERS = {"fuseg": "s", "azh": "^", "medetec": "D"}
LABELS = {"fuseg": "FUSeg", "azh": "AZH", "medetec": "Medetec"}
NOTE = {"fuseg": "curve built from this", "azh": "held out", "medetec": "held out"}
OFFSETS = {"fuseg": ((-6, -34), "center"),
           "azh": ((10, 6), "left"),
           "medetec": ((6, 26), "left")}


def main():
    d = json.load(open(os.path.join(OUT, "framing_law.json")))
    curve = sorted(d["curve"], key=lambda r: r["achieved"])
    x = np.array([r["achieved"] for r in curve])
    y = np.array([r["trend"] for r in curve])
    overlay = d["overlay"]

    fig, ax = plt.subplots(figsize=(7.2, 4.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # the band within which the verdict changes sign
    sgn = None
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            sgn = (x[i], x[i + 1])
            break
    if sgn:
        ax.axvspan(sgn[0], sgn[1], color=BAND, zorder=0, lw=0)
        ax.text(np.sqrt(sgn[0] * sgn[1]), -1.14,
                f"verdict changes sign\n{sgn[0]:.3f} to {sgn[1]:.3f}",
                ha="center", va="top", fontsize=7.5, color=SECONDARY)

    ax.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

    # the curve: eight crops of one dataset, identical images and annotations
    ax.plot(x, y, color=INK, lw=2.0, zorder=3, solid_capstyle="round")
    ax.plot(x, y, "o", color=INK, ms=5.5, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.6)

    # the three real clinical datasets
    for k, v in overlay.items():
        xi = v["lesion_fraction"]
        pred, meas = v["predicted"], v["measured"]
        ax.plot([xi, xi], [pred, meas], color=SECONDARY, lw=1.0, zorder=5)
        ax.plot(xi, pred, MARKERS[k], ms=8, mfc=SURFACE, mec=SECONDARY,
                mew=1.4, zorder=6)
        ax.plot(xi, meas, MARKERS[k], ms=8, mfc=INK, mec=SURFACE,
                mew=1.6, zorder=7)
        off, ha = OFFSETS[k]
        ax.annotate(f"{LABELS[k]}\n{NOTE[k]}", (xi, meas),
                    textcoords="offset points", xytext=off,
                    fontsize=8, color=INK, ha=ha, linespacing=1.35)

    ax.set_xscale("log")
    ax.set_xlim(0.006, 0.22)
    ax.set_ylim(-1.32, 1.18)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.20])
    ax.set_xticklabels(["0.01", "0.02", "0.05", "0.10", "0.20"])
    ax.set_xlabel("lesion fraction of the clinical reference", fontsize=9.5,
                  color=INK, labelpad=7)
    ax.set_ylabel("contrast verdict\n(trend of rank agreement vs contrast)",
                  fontsize=9.5, color=INK, labelpad=7)

    ax.text(0.208, 0.62, "rewards\nhigh contrast", fontsize=8, color=SECONDARY,
            va="center", ha="right", linespacing=1.35)
    ax.text(0.0072, -0.48, "rewards\nlow contrast", fontsize=8, color=SECONDARY,
            va="center", ha="left", linespacing=1.35)

    mono = d["monotonicity"]
    ax.set_title("The verdict is set by how much of the frame the lesion fills",
                 fontsize=11, color=INK, loc="left", pad=26)
    ax.text(0, 1.035,
            f"eight crops of one dataset, Spearman {mono['rho']:+.3f}, "
            f"p = {mono['p']:.4f}",
            transform=ax.transAxes, fontsize=8.5, color=SECONDARY, va="bottom")
    ax.text(0, 1.005,
            "open marker = predicted by the curve, filled = measured",
            transform=ax.transAxes, fontsize=8.5, color=SECONDARY, va="bottom")

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=SECONDARY, labelsize=8.5, length=3)
    ax.grid(axis="y", color="#F1F1F1", lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_framing_law.png")
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    print("saved " + p)


if __name__ == "__main__":
    main()

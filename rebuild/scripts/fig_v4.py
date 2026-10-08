#!/usr/bin/env python3
"""Figures for manuscript V4.

fig1_matrix.png       agreement matrix over ten references, three method pools
fig2_composition.png  where pooled structure comes from
fig3_pool_dependence  benchmark validity against two clearly labelled ceilings

Diverging scale: burnt orange (negative) to near-white to muted purple
(positive), validated for colour-vision deficiency. Every matrix cell carries
its value, so sign never depends on colour alone.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402
import hierarchy_v4 as H                                          # noqa: E402

OUT = os.path.join(ROOT, "results")
INK, SEC, MUT, SURF = "#111111", "#444444", "#7A7A7A", "#FFFFFF"
NEG, MID, POS = "#B35806", "#F4F4F2", "#6A4A9C"
CMAP = LinearSegmentedColormap.from_list("po", [NEG, MID, POS])

ORDER = ["medetec", "fuseg", "azh",
         "kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb",
         "etis_larib", "ph2", "tn3k_test"]
LABEL = {"medetec": "Medetec", "fuseg": "FUSeg", "azh": "AZH",
         "kvasir_disjoint": "Kvasir", "kvasir_sessile": "Kvasir-sessile",
         "cvc_clinicdb": "CVC-ClinicDB", "cvc_colondb": "CVC-ColonDB",
         "etis_larib": "ETIS-Larib", "ph2": "PH2", "tn3k_test": "TN3K"}
BLOCKS = [("wound", 0, 3), ("colonoscopy", 3, 8), ("derm.", 8, 9), ("US", 9, 10)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUT)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=SEC, labelsize=8.5, length=3)
    ax.set_axisbelow(True)


def pools(names):
    trained = lambda n: (ALL_METHOD_FAMILY[n] in ("learned", "CNN")
                         or ALL_METHOD_FAMILY[n].startswith("deep"))
    tr = np.array([i for i, n in enumerate(names) if trained(n)])
    un = np.array([i for i, n in enumerate(names) if not trained(n)])
    return {"all 32 methods": np.arange(len(names)),
            "18 untrained methods": un, "14 trained methods": tr}


def matrix(vec, idx):
    n = len(ORDER)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = spearmanr(vec[ORDER[i]][idx],
                                          vec[ORDER[j]][idx]).statistic
    return M


def fig1(data, names, metric="dice"):
    vec = {k: H.means(data[k], metric) for k in ORDER}
    P = pools(names)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.6), dpi=200)
    fig.patch.set_facecolor(SURF)
    for ax, (title, idx) in zip(axes, P.items()):
        M = matrix(vec, idx)
        ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                if i == j:
                    continue
                v = M[i, j]
                ax.text(j, i, f"{v:+.2f}".replace("+", ""), ha="center",
                        va="center", fontsize=6.3,
                        color=SURF if abs(v) > 0.62 else INK)
        for _, a, b in BLOCKS[1:]:
            ax.axhline(a - 0.5, color=INK, lw=1.1)
            ax.axvline(a - 0.5, color=INK, lw=1.1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_yticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=55, ha="right",
                           fontsize=7.8, color=INK)
        ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=7.8, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    cax = fig.add_axes([0.93, 0.22, 0.011, 0.56])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=SEC, length=2)
    cb.set_label("Spearman rank agreement", fontsize=8.5, color=SEC)
    fig.suptitle("Agreement between clinical references, by method pool "
                 "(Dice, per-method means)", fontsize=11.5, color=INK,
                 x=0.012, ha="left", y=0.99)
    fig.text(0.012, 0.935, "Black lines separate modalities: wound "
             "photography, colonoscopy, dermoscopy, thyroid ultrasound.",
             fontsize=8.8, color=SEC)
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.2, top=0.86,
                        wspace=0.42)
    p = os.path.join(OUT, f"fig1_matrix_{metric}.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


def fig2(data, names):
    res = json.load(open(os.path.join(OUT, "hierarchy_v4.json")))
    split = res["training_domain_split"]["dice_all"]
    comp = res["composition"]
    cats = ["wound x wound", "non-wound, same modality",
            "non-wound, across modality", "wound x non-wound"]
    disp = ["wound vs\nwound", "non-wound,\nsame modality",
            "non-wound,\nacross modality", "wound vs\nnon-wound"]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), dpi=200,
                             gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(SURF)

    ax = axes[0]
    ax.set_facecolor(SURF)
    x = np.arange(len(cats))
    marks = [("all", "o", INK, INK, "all 32 (mixed)"),
             ("untrained", "s", SURF, SEC, "18 untrained"),
             ("trained", "D", SURF, SEC, "14 trained")]
    off = {"all": -0.2, "untrained": 0.0, "trained": 0.2}
    ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
    for key, m, fc, ec, lab in marks:
        y = [split[c][key] for c in cats]
        ax.plot(x + off[key], y, m, ms=8 if key == "all" else 7, mfc=fc,
                mec=ec, mew=1.5, ls="none", label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=8.3, color=INK)
    ax.set_ylabel("mean Spearman over pairs", fontsize=9.5, color=INK)
    ax.set_ylim(-0.55, 1.02)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")
    ax.set_title("Mixing pools manufactures agreement and disagreement",
                 fontsize=10, color=INK, loc="left", pad=10)
    style(ax)
    ax.grid(axis="y", color="#EFEFEF", lw=0.9)

    ax = axes[1]
    ax.set_facecolor(SURF)
    xs = np.arange(len(ORDER))
    tr = [comp[k]["trained"] for k in ORDER]
    un = [comp[k]["untrained"] for k in ORDER]
    ax.axvspan(-0.5, 2.5, color="#EDEDED", zorder=0, lw=0)
    ax.plot(xs, tr, "D-", color=INK, lw=1.8, ms=6.5, mec=SURF, mew=1.2,
            label="14 trained on wound photographs", zorder=3)
    ax.plot(xs, un, "s--", color=SEC, lw=1.4, ms=6, mfc=SURF, mew=1.4,
            label="18 untrained", zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=50, ha="right",
                       fontsize=8, color=INK)
    ax.set_ylabel("mean Dice", fontsize=9.5, color=INK)
    ax.text(1.0, 0.62, "training\ndomain", ha="center", fontsize=8.3,
            color=SEC, linespacing=1.3)
    ax.set_ylim(-0.02, 0.68)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right")
    ax.set_title("Trained methods lead in their domain and collapse outside it",
                 fontsize=10, color=INK, loc="left", pad=10)
    style(ax)
    ax.grid(axis="y", color="#EFEFEF", lw=0.9)

    fig.tight_layout()
    p = os.path.join(OUT, "fig2_composition_v4.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


def fig3():
    d = json.load(open(os.path.join(OUT, "benchmark_quality32.json")))
    ceil_new = d["ceiling"]["rho"]                  # Medetec vs AZH, 32 methods
    ceil_old = 0.567                                # Medetec vs FUSeg, 22 methods
    old_best = 0.554                                # 22-method best, 98% of 0.567
    rows = sorted(d["rows"], key=lambda r: r["texture"])
    fig, ax = plt.subplots(figsize=(7.4, 4.7), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    x = [r["texture"] for r in rows]
    y = [r["mean_rho"] for r in rows]

    ax.axhline(ceil_new, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax.text(x[0], ceil_new + 0.035,
            "clinical-to-clinical ceiling, one line because the two coincide:",
            fontsize=8.2, color=INK, ha="left")
    ax.text(x[0], ceil_new + 0.012,
            f"Medetec vs AZH over 32 methods {ceil_new:+.3f};  "
            f"Medetec vs FUSeg over 22 methods {ceil_old:+.3f}",
            fontsize=8.2, color=SEC, ha="left")

    ax.plot(x, y, "o-", color=INK, lw=2.2, ms=7, zorder=3,
            markeredgecolor=SURF, markeredgewidth=1.5,
            label="32 methods, against Medetec and AZH")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi / ceil_new:.0%}", (xi, yi), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=8.5, color=SEC)
    ax.plot([x[1]], [old_best], "^", ms=9, color=SURF, markeredgecolor=SEC,
            markeredgewidth=1.5, zorder=3,
            label="22 methods, against Medetec and FUSeg")
    ax.annotate(f"{old_best / ceil_old:.0%}", (x[1], old_best),
                textcoords="offset points", xytext=(11, -12),
                fontsize=8.5, color=SEC)

    ax.set_xlabel("measured background texture of the benchmark", fontsize=9.5,
                  color=INK, labelpad=7)
    ax.set_ylabel("agreement with clinical references", fontsize=9.5,
                  color=INK)
    ax.set_ylim(0, 0.70)
    ax.legend(frameon=False, fontsize=8.3, loc="lower right")
    ax.set_title("The same generator, measured over two method pools",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    style(ax)
    ax.grid(axis="y", color="#F1F1F1", lw=0.9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_pool_dependence_v4.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


def fig4():
    """Framing curve built from AZH crops, within each method pool."""
    d = json.load(open(os.path.join(OUT, "framing_azh_pools.json")))
    frac = d["fractions"]
    R = d["results"]
    cache = json.load(open(os.path.join(OUT, "clinical_ranking_cache32.json")))
    f0 = cache["medetec__lesion_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURF)
    spec = [("all 32", "o-", INK, INK, 2.0, "all 32 (mixed)"),
            ("18 untrained", "s--", SEC, SURF, 1.5, "18 untrained"),
            ("14 trained", "D:", SEC, SURF, 1.5, "14 trained")]
    for ax, metric, title in zip(axes, ("dice", "nsd"),
                                 ("Dice", "normalised surface Dice, 2 px")):
        ax.set_facecolor(SURF)
        ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
        for pool, fmt, col, mfc, lw, lab in spec:
            r = R[f"{metric}|{pool}"]
            ax.plot(frac, r["curve"], fmt, color=col, lw=lw, ms=6, mfc=mfc,
                    mec=col, mew=1.3, label=lab, zorder=3)
            ax.plot([f0], [r["medetec_meas"]], "*", ms=10,
                    color=col if pool != "all 32" else INK, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([0.035, 0.05, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0.035", "0.05", "0.10", "0.20", "0.30"])
        ax.set_xlabel("lesion fraction of the reference (AZH crops)",
                      fontsize=9.5, color=INK, labelpad=6)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        style(ax)
        ax.grid(axis="y", color="#EFEFEF", lw=0.9)
    axes[0].set_ylabel("slope of agreement against contrast\n"
                       "(per 10 saturation units)", fontsize=9.5, color=INK)
    axes[0].legend(frameon=False, fontsize=8.3, loc="lower left")
    axes[1].text(0.98, 0.04, "stars: held-out Medetec, measured",
                 transform=axes[1].transAxes, ha="right", fontsize=8,
                 color=SEC)
    fig.suptitle("The reversal exists only in the mixed pool; the attenuation "
                 "survives within pools under NSD", fontsize=11, color=INK,
                 x=0.012, ha="left", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_framing_pools.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


def fig1_print(data, names, metric="dice"):
    """Page-width version: no in-cell numbers (they are in Appendix Table A1)."""
    vec = {k: H.means(data[k], metric) for k in ORDER}
    P = pools(names)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), dpi=220)
    fig.patch.set_facecolor(SURF)
    for ax, (title, idx) in zip(axes, P.items()):
        M = matrix(vec, idx)
        ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest")
        for _, a, b in BLOCKS[1:]:
            ax.axhline(a - 0.5, color=INK, lw=1.2)
            ax.axvline(a - 0.5, color=INK, lw=1.2)
        ax.set_xticks(range(len(ORDER)))
        ax.set_yticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=60, ha="right",
                           fontsize=8.2, color=INK)
        ax.set_yticklabels([LABEL[k] for k in ORDER] if ax is axes[0] else [],
                           fontsize=8.2, color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)
    cax = fig.add_axes([0.925, 0.3, 0.012, 0.52])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=SEC, length=2)
    cb.set_label("Spearman", fontsize=8.5, color=SEC)
    fig.subplots_adjust(left=0.1, right=0.91, bottom=0.27, top=0.9, wspace=0.08)
    p = os.path.join(OUT, f"fig1_matrix_print_{metric}.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


def fig5():
    d = json.load(open(os.path.join(OUT, "nonreduce32_pools.json")))
    short = {"generator A, uniform drape": "A, uniform",
             "generator C, no noise": "C, no noise",
             "generator C, noise 3.4": "C, noise 3.4",
             "generator B, texture ablated": "B, ablated"}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURF)
    for ax, pool in zip(axes, ("all 32", "18 untrained")):
        ax.set_facecolor(SURF)
        rows = {n: (t, v) for n, t, v in d[pool]["rows"]}
        for n, (t, v) in rows.items():
            ax.plot(t, v, "o", ms=9, color=INK, mec=SURF, mew=1.5, zorder=4)
            ax.annotate(short[n], (t, v), textcoords="offset points",
                        xytext=(9, -3), fontsize=8.3, color=INK)
        for a, b, note in (("generator C, noise 3.4",
                            "generator B, texture ablated", "0.03 apart"),
                           ("generator A, uniform drape",
                            "generator C, no noise", "0.24 apart")):
            (xa, ya), (xb, yb) = rows[a], rows[b]
            xm = (xa + xb) / 2
            ax.plot([xm, xm], [ya, yb], color=SEC, lw=1.2, zorder=2)
            ax.plot([xa, xm], [ya, ya], color=SEC, lw=0.9, ls=":", zorder=2)
            ax.plot([xb, xm], [yb, yb], color=SEC, lw=0.9, ls=":", zorder=2)
            ax.annotate(f"{note},\ndiffer by {abs(ya - yb):.3f}",
                        (xm, (ya + yb) / 2), textcoords="offset points",
                        xytext=(-10, 0), ha="right", va="center", fontsize=7.8,
                        color=SEC, linespacing=1.3)
        ax.axhline(0, color=MUT, lw=1.0, ls=(0, (4, 3)))
        ax.set_xlim(1.75, 3.75)
        ax.set_xlabel("measured background texture statistic", fontsize=9.5,
                      color=INK)
        ax.set_title(pool + " methods", fontsize=10, color=INK, loc="left",
                     pad=8)
        style(ax)
        ax.grid(axis="y", color="#EFEFEF", lw=0.9)
    axes[0].set_ylabel("agreement with Medetec and AZH", fontsize=9.5,
                       color=INK)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_nonreducibility_v4.png")
    fig.savefig(p, facecolor=SURF, bbox_inches="tight")
    print("saved " + p)


if __name__ == "__main__":
    data, names = H.load()
    want = sys.argv[1:] or ["1", "2", "3"]
    if "1" in want:
        fig1(data, names, "dice")
        fig1(data, names, "nsd")
    if "2" in want:
        fig2(data, names)
    if "3" in want:
        fig3()
    if "4" in want:
        fig4()
    if "1p" in want:
        fig1_print(data, names, "dice")
    if "5" in want:
        fig5()

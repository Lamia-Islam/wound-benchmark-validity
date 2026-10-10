#!/usr/bin/env python3
"""Figures for manuscript V5, drawn at print size (174 mm wide), greyscale,
lettering 8 to 9 pt. Writes TIFF (600 dpi, LZW) for submission and PNG for
the review copy into ../manuscript_v5/figures/."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "results", "stats_v5.json")))
OUT = os.path.join(ROOT, "..", "manuscript_v5", "figures")
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4
INK, SEC, MUT, GRID = "#111111", "#444444", "#6E6E6E", "#E2E2E2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.edgecolor": SEC, "axes.labelcolor": INK,
                     "xtick.color": SEC, "ytick.color": SEC,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})
STYLE = {  # pool kind -> marker, face
    "untrained": dict(marker="s", mfc="white", mec=INK, color=INK),
    "trained": dict(marker="D", mfc="white", mec=SEC, color=SEC),
    "mixed": dict(marker="o", mfc=INK, mec=INK, color=INK),
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".tif"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, facecolor="white")
    plt.close(fig)
    print("saved", name)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvline(0, color=MUT, lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def fig1():
    """Pair categories, three panels: modern, from-scratch, Kvasir-trained."""
    panels = [
        ("a  Untrained with modern\n    (domain: wound photography)", "wound", ["U", "M", "U+M"]),
        ("b  Untrained with from-scratch\n    (domain: wound photography)", "wound", ["U", "F", "U+F"]),
        ("c  Untrained with Kvasir-trained\n    (domain: colonoscopy)", "colonoscopy", ["U", "K", "U+K"]),
    ]
    cats = [("in x in", "domain with domain"), ("in x out", "domain with other"),
            ("out, same modality", "other, same modality"),
            ("out, across modality", "other, across modality")]
    kinds = ["untrained", "trained", "mixed"]
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 64 * MM), sharex=True, sharey=True)
    y0 = np.arange(len(cats))[::-1]
    for ax, (title, dom, pools) in zip(axes, panels):
        for j, pool in enumerate(pools):
            st = STYLE[kinds[j]]
            d = S["comp"][f"dice|{pool}|{dom}"]
            for i, (c, _) in enumerate(cats):
                v = d[c]
                y = y0[i] + (1 - j) * 0.25
                ax.plot([v["lo"], v["hi"]], [y, y], color=st["color"], lw=0.9)
                ax.plot(v["est"], y, ls="none", ms=4.6, mew=0.9,
                        **{k: st[k] for k in ("marker", "mfc", "mec")})
        ax.set_xlim(-0.8, 1.05)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-0.5", "0", "0.5", "1"])
        ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=4)
        style_ax(ax)
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels([c[1] for c in cats])
    axes[1].set_xlabel("mean Spearman agreement between clinical datasets (Dice, 95% interval)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="-", lw=0.9, color=STYLE[k]["color"], marker=STYLE[k]["marker"],
                mfc=STYLE[k]["mfc"], mec=STYLE[k]["mec"], ms=4.6) for k in kinds]
    fig.legend(h, ["untrained methods alone", "trained methods alone",
                   "both pooled"], loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    save(fig, "Fig1_pooling")


def fig2():
    """Benchmark-to-clinical agreement next to clinical-to-clinical agreement."""
    rows = [("U", "18 untrained", "untrained"), ("F", "14 from scratch", "trained"),
            ("M", "11 modern", "trained"), ("K", "14 Kvasir-trained", "trained"),
            ("U+F", "32 pooled: untrained + from scratch", "mixed"),
            ("U+K", "32 pooled: untrained + Kvasir-trained", "mixed"),
            ("U+M", "29 pooled: untrained + modern", "mixed")]
    fig, ax = plt.subplots(figsize=(140 * MM, 72 * MM))
    y = np.arange(len(rows))[::-1]
    for yi, (pool, label, kind) in zip(y, rows):
        b = S["bench40"][pool]["mean over references"]
        c = S["bench40"][pool]["clinical wound pairs"]
        ax.plot([b["lo"], b["hi"]], [yi + 0.14] * 2, color=INK, lw=0.9)
        ax.plot(b["est"], yi + 0.14, "o", ms=4.8, mfc=INK, mec=INK,
                label="synthetic benchmark with clinical wound data" if yi == y[0] else None)
        ax.plot([c["lo"], c["hi"]], [yi - 0.14] * 2, color=MUT, lw=0.9)
        ax.plot(c["est"], yi - 0.14, "^", ms=5, mfc="white", mec=SEC, mew=0.9,
                label="clinical wound data with each other" if yi == y[0] else None)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlim(-0.4, 1.0)
    ax.set_xlabel("mean Spearman agreement (Dice, 95% interval)")
    style_ax(ax)
    ax.axhline(2.5, color=GRID, lw=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=1, fontsize=7.5,
              frameon=False)
    fig.tight_layout()
    save(fig, "Fig2_benchmark_validity")


def fig_matrix():
    """Online Resource: agreement matrices for the modern, untrained and pooled sets."""
    import itertools, sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import pools_v5 as P
    from scipy.stats import spearmanr
    from matplotlib.colors import LinearSegmentedColormap
    G = P.groups(); T = P.load()["dice"]
    order = ["medetec", "co2wounds", "fuseg", "azh", "kvasir_disjoint", "kvasir_sessile",
             "cvc_clinicdb", "cvc_colondb", "etis_larib", "ph2", "tn3k_test"]
    lab = ["Medetec", "CO2Wounds-V2", "FUSeg", "AZH", "Kvasir", "Kvasir-sessile",
           "CVC-ClinicDB", "CVC-ColonDB", "ETIS-Larib", "PH2", "TN3K"]
    pools = [("18 untrained", G["U"]), ("11 modern", G["P"] + G["N"]),
             ("29 pooled", G["U"] + G["P"] + G["N"])]
    cmap = LinearSegmentedColormap.from_list("bw", ["#3A3A3A", "#FFFFFF", "#3A3A3A"])
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 70 * MM))
    for ax, (t, nm) in zip(axes, pools):
        M = np.array([[spearmanr(P.vec(T, a, nm), P.vec(T, b, nm)).statistic for b in order] for a in order])
        ax.imshow(np.abs(M), cmap="Greys", vmin=0, vmax=1.6)
        for i, j in itertools.product(range(11), range(11)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:+.1f}".replace("+", "") if M[i, j] >= 0 else f"{M[i, j]:.1f}",
                        ha="center", va="center", fontsize=4.6,
                        color="white" if abs(M[i, j]) > 0.8 else INK,
                        fontweight="bold" if M[i, j] < 0 else "normal")
        ax.set_xticks(range(11)); ax.set_yticks(range(11))
        ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(lab if ax is axes[0] else [], fontsize=6)
        for b in (3.5, 8.5, 9.5):
            ax.axhline(b, color=INK, lw=0.8); ax.axvline(b, color=INK, lw=0.8)
        ax.set_title(t, fontsize=8, loc="left")
    fig.tight_layout(w_pad=0.5)
    save(fig, "OR_Fig1_matrices")


if __name__ == "__main__":
    fig1(); fig2(); fig_matrix()

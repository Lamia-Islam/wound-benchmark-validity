#!/usr/bin/env python3
"""Figures for manuscript V5. No titles inside figures; lettering 8 pt or more
at 174 mm print width; values printed so sign survives greyscale printing."""
import itertools, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pools_v5 as P
OUT = os.path.join(ROOT, "..", "manuscript_v5")
os.makedirs(OUT, exist_ok=True)
INK, SEC, MUT = "#111111", "#3a3a3a", "#6b6b6b"
CMAP = LinearSegmentedColormap.from_list("po", ["#B35806", "#F4F4F2", "#5E3C99"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.edgecolor": SEC, "axes.labelcolor": INK,
                     "xtick.color": SEC, "ytick.color": SEC})
ORDER = ["medetec", "co2wounds", "fuseg", "azh", "kvasir_disjoint", "kvasir_sessile",
         "cvc_clinicdb", "cvc_colondb", "etis_larib", "ph2", "tn3k_test"]
LABEL = {"medetec": "Medetec", "co2wounds": "CO2Wounds-V2", "fuseg": "FUSeg", "azh": "AZH",
         "kvasir_disjoint": "Kvasir", "kvasir_sessile": "Kvasir-sessile",
         "cvc_clinicdb": "CVC-ClinicDB", "cvc_colondb": "CVC-ColonDB",
         "etis_larib": "ETIS-Larib", "ph2": "PH2", "tn3k_test": "TN3K"}
MM = 1 / 25.4
G = P.groups(); G["M"] = G["P"] + G["N"]
T = P.load()


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    for ext in ("png", "tiff"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=600 if ext == "tiff" else 300,
                    bbox_inches="tight", facecolor="white",
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}))
    plt.close(fig)


def fig1():
    """Mean Dice of each method group on each clinical dataset."""
    fig, ax = plt.subplots(figsize=(174 * MM, 72 * MM))
    x = np.arange(len(ORDER))
    spec = [("U", "Classical (18)", "o", MUT, "white", -0.24),
            ("F", "Scratch-FUSeg (14)", "s", SEC, SEC, -0.08),
            ("M", "Modern: pretrained and nnU-Net, FUSeg (11)", "D", INK, INK, 0.08),
            ("K", "Scratch-Kvasir (14)", "^", SEC, "white", 0.24)]
    ax.axvspan(-0.5, 3.5, color="#ECECEC", lw=0, zorder=0)
    ax.text(1.5, 0.86, "wound photography", ha="center", fontsize=8, color=SEC)
    for g, lab, mk, c, fc, dx in spec:
        y = [np.mean(P.vec(T["dice"], r, G[g])) for r in ORDER]
        ax.plot(x + dx, y, ls="none", marker=mk, ms=5.5, mfc=fc, mec=c, mew=1.1, label=lab)
    for xb in np.arange(0.5, len(ORDER) - 1, 1):
        ax.axvline(xb, color="#F0F0F0", lw=0.6, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[r] for r in ORDER], rotation=35, ha="right")
    ax.set_ylabel("mean Dice of the group"); ax.set_ylim(0, 0.92); ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", color="#E2E2E2", lw=0.6); style(ax)
    save(fig, "fig1_group_dice")


def matrix(names, metric="dice"):
    v = {r: P.vec(T[metric], r, names) for r in ORDER}
    n = len(ORDER); M = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2):
        M[i, j] = M[j, i] = spearmanr(v[ORDER[i]], v[ORDER[j]]).statistic
    return M


def fig2():
    """Agreement matrices: modern methods alone, and mixed with classical."""
    panels = [("a", "11 pretrained and nnU-Net models", G["M"]),
              ("b", "the same 11 plus 18 classical methods", G["U"] + G["M"])]
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 92 * MM))
    for ax, (tag, lab, names) in zip(axes, panels):
        M = matrix(names)
        im = ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1)
        n = len(ORDER)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.1f}".replace("-0.0", "0.0").lstrip("0").replace("-0", "-"),
                            ha="center", va="center", fontsize=6.3,
                            color="white" if abs(M[i, j]) > 0.6 else INK)
        for b in (3.5, 8.5, 9.5):
            ax.axhline(b, color=INK, lw=0.8); ax.axvline(b, color=INK, lw=0.8)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([LABEL[r] for r in ORDER], rotation=60, ha="right", fontsize=7.5)
        ax.set_yticklabels([LABEL[r] for r in ORDER] if tag == "a" else [], fontsize=7.5)
        ax.set_xlabel(f"({tag}) {lab}", fontsize=8.5, labelpad=6)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Spearman agreement", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    save(fig, "fig2_matrices")


def fig3():
    """Key agreements with 95% method-bootstrap intervals, per pool."""
    R = json.load(open(os.path.join(ROOT, "results", "v5_numbers.json")))["pools"]
    pools = ["U", "F", "M", "K", "U+F", "U+M", "U+K"]
    lab = {"U": "Classical (18)", "F": "Scratch-FUSeg (14)", "M": "Modern (11)",
           "K": "Scratch-Kvasir (14)", "U+F": "Classical + Scratch-FUSeg",
           "U+M": "Classical + Modern", "U+K": "Classical + Scratch-Kvasir"}
    stats = [("wound_x_wound", "(a) wound dataset\nwith wound dataset"),
             ("dom:in_out", "(b) training domain\nwith other modalities"),
             ("bench_wound", "(c) synthetic benchmark\nwith wound datasets")]
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM), sharey=True)
    y = np.arange(len(pools))[::-1]
    for ax, (k, t) in zip(axes, stats):
        for yi, p in zip(y, pools):
            r = R[f"dice|{p}"]; o = r["obs"][k]; lo, hi = r["ci"][k]
            mixed = "+" in p
            ax.plot([lo, hi], [yi, yi], color=INK if mixed else SEC, lw=1.2)
            ax.plot(o, yi, "o", ms=5, mfc=INK if mixed else "white", mec=INK)
        ax.axvline(0, color=MUT, lw=0.8, ls=(0, (3, 2)))
        ax.axhline(2.5, color="#BDBDBD", lw=0.7)
        ax.set_xlim(-0.8, 1.0); ax.set_xlabel(t, fontsize=8)
        ax.grid(axis="x", color="#E6E6E6", lw=0.5); style(ax)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lab[p] for p in pools])
    save(fig, "fig3_forest")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3"]
    if "1" in want: fig1()
    if "2" in want: fig2()
    if "3" in want: fig3()

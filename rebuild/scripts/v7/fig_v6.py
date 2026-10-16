#!/usr/bin/env python3
"""Figures 1 to 3 for manuscript V6 (JIIM), drawn at print size, greyscale.

Style follows scripts/fig_v5.py (greyscale, open and filled markers, no colour),
adapted to JIIM: Liberation Sans (metric-compatible with Arial; Arial and Helvetica
are not installed), lettering 8 pt at print size, widths 174 mm (Fig 1, 3) and
129 mm (Fig 2). Writes, into manuscript_v6/figures/, TIFF (line art, 1200 dpi,
LZW, RGB without alpha), PNG (300 dpi, embedded by the manuscript build), vector
PDF and EPS; submission copies Fig1/2/3.{tif,eps,pdf} go to figures/submission/.

Every plotted number is read from results/v7/{stats_v7,bench_v7,robust_v7}.json
through `take()`, which records the JSON path. After drawing, the full list of
plotted values with their paths is written to results/v7/FIGURE_VALUES_V6.md.

Fig 1 and the V5-comparable rows are checked against results/stats_v5.json
(the source of the V5 manuscript figures) before anything is drawn.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # rebuild/
RES = os.path.join(ROOT, "results", "v7")
OUT = os.path.normpath(os.path.join(ROOT, "..", "manuscript_v6", "figures"))
os.makedirs(OUT, exist_ok=True)
SUB = os.path.join(OUT, "submission")
os.makedirs(SUB, exist_ok=True)

J = {n: json.load(open(os.path.join(RES, f"{n}.json"))) for n in ("stats_v7", "bench_v7", "robust_v7")}
V5 = json.load(open(os.path.join(ROOT, "results", "stats_v5.json")))
# stats_v5.json was computed under exclusions_v5.json. The value checks against it
# apply only when the V7 results use the same exclusions; with exclusions_v6.json
# the V5-comparable values differ by design (AZH and the control lost images).
V5_SAME_EXCL = json.load(open(os.path.join(RES, "stats_v7.json"))).get("exclusions", "exclusions_v5.json") == "exclusions_v5.json"

MM = 1 / 25.4
INK, SEC, MUT, GRID, BAND = "#111111", "#444444", "#6E6E6E", "#E2E2E2", "#C8C8C8"
MINUS = "−"
from matplotlib import font_manager as _fm
FONT = "Liberation Sans"  # Arial/Helvetica not installed (fc-list); Liberation Sans is metric-compatible with Arial
_fm.findfont(_fm.FontProperties(family=FONT), fallback_to_default=False)  # fail rather than silently fall back
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 8,
                     "legend.fontsize": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.edgecolor": SEC, "axes.labelcolor": INK,
                     "xtick.color": SEC, "ytick.color": SEC,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.unicode_minus": True,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
STYLE = {  # pool kind -> marker, face (as fig_v5)
    "untrained": dict(marker="s", mfc="white", mec=INK, color=INK),
    "trained": dict(marker="D", mfc="white", mec=SEC, color=SEC),
    "mixed": dict(marker="o", mfc=INK, mec=INK, color=INK),
}
TOL = 1e-12

# ------------------------------------------------------------------ registry
RECORDS = []  # (figure, panel, row, quantity, value, json path)


def get(src, path):
    o = J[src]
    for k in path:
        o = o[k]
    return o


def fmt_path(src, path):
    return f"{src}.json" + "".join(f"[{k!r}]" if isinstance(k, str) else f"[{k}]" for k in path)


def take(fig, panel, row, src, path, kind):
    """Read one plotted quantity and record it with its JSON path.
    kind: 'est_ci' (dict with est and a 2-list under key), 'est_lohi', 'scalar'."""
    if kind[0] == "est_ci":
        node = get(src, path)
        est, (lo, hi) = node["est"], node[kind[1]]
        RECORDS.append((fig, panel, row, "estimate", est, fmt_path(src, path + ["est"])))
        RECORDS.append((fig, panel, row, "95% lower", lo, fmt_path(src, path + [kind[1], 0])))
        RECORDS.append((fig, panel, row, "95% upper", hi, fmt_path(src, path + [kind[1], 1])))
        return float(est), float(lo), float(hi)
    if kind[0] == "est_lohi":
        node = get(src, path)
        for q, k in (("estimate", "est"), ("95% lower", "lo"), ("95% upper", "hi")):
            RECORDS.append((fig, panel, row, q, node[k], fmt_path(src, path + [k])))
        return float(node["est"]), float(node["lo"]), float(node["hi"])
    if kind[0] == "pair":  # estimate and interval stored under two separate paths
        est = get(src, path)
        lo, hi = get(src, kind[1])
        RECORDS.append((fig, panel, row, "estimate", est, fmt_path(src, path)))
        RECORDS.append((fig, panel, row, "95% lower", lo, fmt_path(src, kind[1] + [0])))
        RECORDS.append((fig, panel, row, "95% upper", hi, fmt_path(src, kind[1] + [1])))
        return float(est), float(lo), float(hi)
    v = get(src, path)
    RECORDS.append((fig, panel, row, kind[1] if len(kind) > 1 else "value", v, fmt_path(src, path)))
    return v


CHECKS = []


def check(name, ok, detail=""):
    assert ok, f"CHECK FAILED: {name} {detail}"
    CHECKS.append(f"{name}: passed{(' (' + detail + ')') if detail else ''}")


def nfmt(v, nd=1):
    """Tick label with the Unicode minus sign."""
    s = f"{v:.{nd}f}" if nd is not None else f"{v:g}"
    if s.startswith("-"):
        s = MINUS + s[1:]
    if s in (MINUS + "0", MINUS + "0.0", MINUS + "0.00"):
        s = s[1:]
    return s


def set_ticks(axis, ticks, nd):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _p: nfmt(v, nd)))


TIF_DPI = 1200
SIZES = {}


def min_font_pt(fig):
    """Smallest font size of any visible text artist in the figure."""
    from matplotlib.text import Text
    sz = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sz), max(sz)


def save(fig, name, num):
    import shutil
    from PIL import Image
    lo_pt, hi_pt = min_font_pt(fig)
    assert lo_pt >= 7, f"{name}: text below 7 pt ({lo_pt})"
    w_mm, h_mm = (x * 25.4 for x in fig.get_size_inches())
    SIZES[name] = (w_mm, h_mm, lo_pt, hi_pt)
    for ext, kw in (("tif", dict(dpi=TIF_DPI)), ("png", dict(dpi=300)), ("pdf", dict()), ("eps", dict())):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), facecolor="white", **kw)
    plt.close(fig)
    # TIFF: flatten to RGB on white (no alpha channel), 1200 dpi, LZW
    tif = os.path.join(OUT, f"{name}.tif")
    im = Image.open(tif)
    bg = Image.new("RGB", im.size, "white")
    bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    bg.save(tif, compression="tiff_lzw", dpi=(TIF_DPI, TIF_DPI))
    for ext in ("tif", "eps", "pdf"):
        shutil.copyfile(os.path.join(OUT, f"{name}.{ext}"), os.path.join(SUB, f"Fig{num}.{ext}"))
    print("saved", name, f"{w_mm:.1f} x {h_mm:.1f} mm, text {lo_pt} to {hi_pt} pt")


def style_ax(ax, zero="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if zero == "x":
        ax.axvline(0, color=MUT, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    else:
        ax.axhline(0, color=MUT, lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


N_SET = {k: v["n"] for k, v in J["stats_v7"]["definitions"]["method_sets"].items()}
for p, n in J["bench_v7"]["pools"].items():
    if p in N_SET:
        check(f"method-set size {p} equal in stats_v7 and bench_v7", N_SET[p] == n, f"n {n}")

# ------------------------------------------------------------------ Figure 1
CATS = [("in x in", "domain with domain"), ("in x out", "domain with other"),
        ("out, same modality", "other, same modality"),
        ("out, across modality", "other, across modality")]
BKEY = "method_only_repro_B2000"


def fig1():
    panels = [
        ("a", "Untrained and modern\n    domain: wound", "wound", ["U", "M19", "U+M19"]),
        ("b", "Untrained and from scratch\n    domain: wound", "wound", ["U", "F", "U+F"]),
        ("c", "Untrained and Kvasir-trained\n    domain: colonoscopy", "colonoscopy", ["U", "K", "U+K"]),
    ]
    kinds = ["untrained", "trained", "mixed"]
    # extraction check: every V5-comparable value equals stats_v5.json "comp" (V5 Figure 1 source)
    nchk, dmax = 0, 0.0
    for _l, _t, dom, pools in panels:
        for pool in pools:
            v5key = f"dice|{pool}|{dom}"
            if v5key not in V5["comp"] or not V5_SAME_EXCL:
                continue
            for c, _ in CATS:
                b = get("stats_v7", ["bootstrap", f"dice|{pool}", f"comp|{dom}|{c}"])
                e = V5["comp"][v5key][c]
                d = max(abs(b["est"] - e["est"]), abs(b[BKEY][0] - e["lo"]), abs(b[BKEY][1] - e["hi"]))
                dmax = max(dmax, d)
                nchk += 1
                check(f"Fig1 {pool}|{dom}|{c} equals stats_v5 comp", d < TOL, f"max abs diff {d:.1e}")
    if not V5_SAME_EXCL:
        CHECKS.append("Fig1 values vs stats_v5: not applicable (exclusions_v6.json); the same extraction passed under exclusions_v5.json")
    check("Fig1 number of V5-comparable cells", nchk == (28 if V5_SAME_EXCL else 0), f"{nchk} cell checks: U wound (drawn in panels a and b, checked twice), F, U+F, U colonoscopy, K, U+K; max abs diff {dmax:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 66 * MM), sharex=True, sharey=True)
    y0 = np.arange(len(CATS))[::-1]
    for ax, (letter, title, dom, pools) in zip(axes, panels):
        for j, pool in enumerate(pools):
            st = STYLE[kinds[j]]
            for i, (c, lab) in enumerate(CATS):
                est, lo, hi = take("Figure 1", letter, f"{pool} ({N_SET[pool]}), {lab}", "stats_v7",
                                   ["bootstrap", f"dice|{pool}", f"comp|{dom}|{c}"], ("est_ci", BKEY))
                y = y0[i] + (1 - j) * 0.25
                ax.plot([lo, hi], [y, y], color=st["color"], lw=0.9, solid_capstyle="butt")
                ax.plot(est, y, ls="none", ms=4.6, mew=0.9,
                        **{k: st[k] for k in ("marker", "mfc", "mec")})
        ax.set_xlim(-0.8, 1.05)
        set_ticks(ax.xaxis, [-0.5, 0, 0.5, 1], 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: nfmt(v, None)))
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=8, color=INK, pad=4)
        style_ax(ax)
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels([c[1] for c in CATS], fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    axes[1].set_xlabel("mean Spearman agreement between clinical datasets (Dice, 95% interval)", fontsize=8)
    h = [Line2D([], [], ls="-", lw=0.9, color=STYLE[k]["color"], marker=STYLE[k]["marker"],
                mfc=STYLE[k]["mfc"], mec=STYLE[k]["mec"], ms=4.6) for k in kinds]
    fig.legend(h, ["untrained methods alone", "trained methods alone", "both pooled"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    save(fig, "Fig1_pooling", 1)


# ------------------------------------------------------------------ Figure 2
def fig2():
    rows = [("U", "untrained", "untrained"), ("F", "from scratch", "trained"),
            ("K", "Kvasir-trained", "trained"), ("M19", "modern", "trained"),
            ("U+F", "pooled: untrained + from scratch", "mixed"),
            ("U+K", "pooled: untrained + Kvasir-trained", "mixed"),
            ("U+M19", "pooled: untrained + modern", "mixed")]
    # checks: wound-to-wound = mean of the 6 wound pairs (pairs6) equals the 'mean|other' row;
    # V5-comparable rows equal stats_v5 bench40 (V5 Figure 2 source); pairs6 equals stats_v7 comp|wound|in x in
    for pool, _l, _k in rows:
        r = get("bench_v7", ["rel", pool, "40"])
        d = max(abs(r["pairs6"][k] - r["mean|other"][k]) for k in ("est", "lo", "hi"))
        check(f"Fig2 {pool} pairs6 equals mean|other", d < TOL, f"{d:.1e}")
        s = get("stats_v7", ["bootstrap", f"dice|{pool}", "comp|wound|in x in"])
        d2 = max(abs(r["pairs6"]["est"] - s["est"]), abs(r["pairs6"]["lo"] - s[BKEY][0]),
                 abs(r["pairs6"]["hi"] - s[BKEY][1]))
        check(f"Fig2 {pool} wound-to-wound equals stats_v7 method-only B2000", d2 < 1e-9, f"{d2:.1e}")
        if pool in V5["bench40"] and V5_SAME_EXCL:
            e = V5["bench40"][pool]
            d3 = max(max(abs(r["mean|bench"][k] - e["mean over references"][k]) for k in ("est", "lo", "hi")),
                     max(abs(r["pairs6"][k] - e["clinical wound pairs"][k]) for k in ("est", "lo", "hi")))
            check(f"Fig2 {pool} equals stats_v5 bench40", d3 < TOL, f"{d3:.1e}")

    fig, ax = plt.subplots(figsize=(129 * MM, 78 * MM))
    y = np.arange(len(rows))[::-1]
    for yi, (pool, label, kind) in zip(y, rows):
        lab = f"{N_SET[pool]} {label}"
        b = take("Figure 2", "-", f"{lab} [{pool}]: benchmark with wound data", "bench_v7",
                 ["rel", pool, "40", "mean|bench"], ("est_lohi",))
        c = take("Figure 2", "-", f"{lab} [{pool}]: wound data with each other", "bench_v7",
                 ["rel", pool, "40", "pairs6"], ("est_lohi",))
        ax.plot([b[1], b[2]], [yi + 0.14] * 2, color=INK, lw=0.9, solid_capstyle="butt")
        ax.plot(b[0], yi + 0.14, "o", ms=4.8, mfc=INK, mec=INK)
        ax.plot([c[1], c[2]], [yi - 0.14] * 2, color=MUT, lw=0.9, solid_capstyle="butt")
        ax.plot(c[0], yi - 0.14, "^", ms=5, mfc="white", mec=SEC, mew=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{N_SET[r[0]]} {r[1]}" for r in rows], fontsize=8)
    ax.set_xlim(-0.4, 1.0)
    set_ticks(ax.xaxis, [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("mean Spearman agreement (Dice, 95% interval)", fontsize=8)
    style_ax(ax)
    ax.axhline(2.5, color=MUT, lw=0.6)
    h = [Line2D([], [], color=INK, lw=0.9, marker="o", ms=4.8, mfc=INK, mec=INK),
         Line2D([], [], color=MUT, lw=0.9, marker="^", ms=5, mfc="white", mec=SEC, mew=0.9)]
    fig.legend(h, ["synthetic benchmark (generator B, texture amplitude 40)\nwith each wound dataset",
                   "wound datasets with each other"],
               loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.81))
    save(fig, "Fig2_benchmark_validity", 2)


# ------------------------------------------------------------------ Figure 3
WNAME = {"medetec": "Medetec", "co2wounds": "CO2Wounds-V2", "fuseg": "FUSeg", "azh": "AZH"}


def fig3():
    sd = ["seeds", "dice|C16"]
    seeds = get("stats_v7", sd)
    check("Fig3a C16 has 16 architectures", seeds["n"] == 16 == get("stats_v7", ["seeds", "sets", "C16", "n"]))
    # per-dataset r-bar: the mean over the 4 wound datasets equals rbar_wound; control r-bar equals rbar_control
    rb = [seeds["per_dataset"][d]["rbar"] for d in WNAME]
    check("Fig3a mean of 4 per-dataset r-bar equals rbar_wound",
          abs(np.mean(rb) - seeds["summary"]["rbar_wound"]["est"]) < TOL)
    check("Fig3a per_dataset azh_train_slice r-bar equals rbar_control",
          abs(seeds["per_dataset"]["azh_train_slice"]["rbar"] - seeds["summary"]["rbar_control"]["est"]) < TOL)
    wp = get("stats_v7", ["seeds", "disattenuation", "dice|C16", "wound_pairs"])
    check("Fig3a 6 wound pairs", len(wp) == 6)
    check("Fig3a mean of per-pair single-seed values equals single|wound pairs",
          abs(np.mean([v["observed_single_mean"] for v in wp.values()])
              - seeds["summary"]["single|wound pairs"]["est"]) < TOL)

    FH = 96
    fig = plt.figure(figsize=(174 * MM, FH * MM))
    ax = fig.add_axes((0.262, 0.105, 0.375, 0.815))   # panel a
    bx = fig.add_axes((0.755, 0.395, 0.23, 0.525))    # panel b, legend below it

    # ---- panel a: rows from top
    rows = []  # (label, kind, payload)
    rows.append(("Same dataset, different seed", "head", None))
    for d, nm in WNAME.items():
        rows.append((nm, "single", ("pair", ["seeds", "dice|C16", "per_dataset", d, "rbar"],
                                    ["seeds", "dice|C16", "per_dataset", d, "rbar_method_boot"])))
    rows.append(("mean of 4 datasets", "mean", ["seeds", "dice|C16", "summary", "rbar_wound"]))
    rows.append(("Different dataset, same seed", "head", None))
    rows.append(("each of 6 pairs (no interval)", "pairs", None))
    rows.append(("mean of 6 pairs", "mean", ["seeds", "dice|C16", "summary", "single|wound pairs"]))
    rows.append(("Different dataset, different seed", "head", None))
    rows.append(("mean of 6 pairs", "mean", ["seeds", "dice|C16", "summary", "crossseed|wound pairs"]))
    rows.append(("Control: AZH train slice", "head", None))
    rows.append(("Control images, different seed", "control",
                 ["seeds", "dice|C16", "summary", "rbar_control"]))
    n = len(rows)
    ys = np.arange(n)[::-1].astype(float)
    labels = []
    for yv, (lab, kind, pl) in zip(ys, rows):
        if kind == "head":
            labels.append((yv, lab, True))
            continue
        labels.append((yv, lab, False))
        if kind == "single":
            est, lo, hi = take("Figure 3", "a", f"same dataset, different seed: {lab}", "stats_v7",
                               pl[1], ("pair", pl[2]))
            ax.plot([lo, hi], [yv, yv], color=SEC, lw=0.9, solid_capstyle="butt")
            ax.plot(est, yv, "o", ms=4.6, mfc="white", mec=INK, mew=0.9)
        elif kind == "mean":
            head = [r[0] for r in rows[:rows.index((lab, kind, pl))] if r[1] == "head"][-1]
            est, lo, hi = take("Figure 3", "a", f"{head.lower()}: {lab}", "stats_v7", pl,
                               ("est_ci", "method_boot"))
            ax.plot([lo, hi], [yv, yv], color=INK, lw=1.2, solid_capstyle="butt")
            ax.plot(est, yv, "o", ms=5.2, mfc=INK, mec=INK)
        elif kind == "pairs":
            for pname in wp:
                a, b = pname.split("~")
                v = take("Figure 3", "a", f"different dataset, same seed: {WNAME[a]} vs {WNAME[b]} "
                         f"(mean over 3 seeds, no interval)", "stats_v7",
                         ["seeds", "disattenuation", "dice|C16", "wound_pairs", pname, "observed_single_mean"],
                         ("scalar", "estimate"))
                ax.plot(v, yv, "o", ms=3.6, mfc="white", mec=MUT, mew=0.8, alpha=1.0)
        elif kind == "control":
            est, lo, hi = take("Figure 3", "a", f"control: {lab}", "stats_v7", pl, ("est_ci", "method_boot"))
            ax.plot([lo, hi], [yv, yv], color=SEC, lw=0.9, solid_capstyle="butt")
            ax.plot(est, yv, "D", ms=4.4, mfc="white", mec=INK, mew=0.9)
    ax.set_ylim(-0.7, n - 0.4)
    det = [l for l in labels if not l[2]]
    ax.set_yticks([l[0] for l in det])
    ax.set_yticklabels([l[1] for l in det], fontsize=8)
    ax.tick_params(axis="y", length=0)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for yv, lab, head in labels:
        if head:  # group headings span the label column and the plot, on their own empty row
            ax.text(0.012, yv, lab, transform=tr, ha="left", va="center", fontsize=8,
                    fontweight="bold", color=INK, clip_on=False,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_xlim(-0.3, 1.0)
    set_ticks(ax.xaxis, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Spearman agreement (Dice, 95% interval)", fontsize=8)
    style_ax(ax)
    for yv, lab, head in labels:
        if head and yv != ys[0]:
            ax.axhline(yv + 0.5, color=GRID, lw=0.8, zorder=0)
    fig.text(0.012, 0.975, "a  16 modern architectures, three training seeds each", ha="left", va="top",
             fontsize=8, color=INK)

    # ---- panel b: common threshold
    grid = sorted((k.split("|")[1] for k in get("robust_v7", ["threshold", "common"]) if k.startswith("M19|")),
                  key=float)
    check("Fig3b grid is 18 thresholds 0.05 to 0.90", len(grid) == 18 and grid[0] == "0.05" and grid[-1] == "0.90")
    xs, est, lo, hi, above = [], [], [], [], []
    for g in grid:
        e, l, h = take("Figure 3", "b", f"common threshold {g}", "robust_v7",
                       ["threshold", "common", f"M19|{g}", "wound pairs"], ("est_ci", "ci"))
        dl = take("Figure 3", "b", f"common threshold {g}: change from own threshold, 95% lower "
                  f"(filled marker if above 0)", "robust_v7",
                  ["threshold", "delta_vs_default", f"M19|{g}|wound pairs", "ci", 0], ("scalar", "value"))
        xs.append(float(g)); est.append(e); lo.append(l); hi.append(h); above.append(dl > 0)
    listed = get("robust_v7", ["common_threshold_wound_pairs_delta_ci_above0", "M19"])
    check("Fig3b filled markers equal robust_v7 common_threshold_wound_pairs_delta_ci_above0",
          [x for x, a in zip(xs, above) if a] == [float(v) for v in listed], f"{listed}")
    d_e, d_l, d_h = take("Figure 3", "b", "own threshold per model (default, line)", "robust_v7",
                         ["threshold", "default", "M19", "wound pairs"], ("est_ci", "ci"))
    o_e, o_l, o_h = take("Figure 3", "b", "per-dataset oracle threshold (line)", "robust_v7",
                         ["threshold", "oracle", "M19", "wound pairs"], ("est_ci", "ci"))
    check("Fig3b default equals stats_v7 M19 wound pairs",
          abs(d_e - get("stats_v7", ["bootstrap", "dice|M19", "wound pairs", "est"])) < TOL)
    bx.fill_between(xs, lo, hi, color=BAND, lw=0, zorder=1)
    bx.plot(xs, est, color=INK, lw=1.0, zorder=3)
    for x, e, a in zip(xs, est, above):
        bx.plot(x, e, "o", ms=4.0, mfc=INK if a else "white", mec=INK, mew=0.9, zorder=4)
    bx.axhline(d_e, color=INK, lw=0.9, ls=(0, (5, 2)), zorder=2)
    bx.axhline(o_e, color=SEC, lw=0.9, ls=(0, (1, 1.5)), zorder=2)
    bx.set_xlim(0, 0.95)
    set_ticks(bx.xaxis, [0.1, 0.3, 0.5, 0.7, 0.9], 1)
    bx.set_ylim(-0.3, 1.0)
    set_ticks(bx.yaxis, [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], 1)
    bx.tick_params(labelsize=8)
    bx.set_xlabel("one common threshold\nfor all 19 models", fontsize=8)
    bx.set_ylabel("mean Spearman agreement,\n6 wound pairs (Dice)", fontsize=8)
    style_ax(bx, zero="y")
    h = [(Patch(facecolor=BAND, edgecolor="none"), Line2D([], [], color=INK, lw=1.0)),
         Line2D([], [], color=INK, lw=0, marker="o", ms=4.0, mfc=INK, mec=INK),
         Line2D([], [], color=INK, lw=0.9, ls=(0, (5, 2))),
         Line2D([], [], color=SEC, lw=0.9, ls=(0, (1, 1.5)))]
    bx.legend(h, ["common threshold, 95% interval",
                  "rise over own threshold:\n95% interval above 0",
                  f"own threshold per model ({d_e:.2f})",
                  f"per-dataset oracle ({o_e:.2f})"],
              loc="upper left", bbox_to_anchor=(0.665, 0.205), bbox_transform=fig.transFigure,
              fontsize=8, frameon=False, handlelength=2.2, borderaxespad=0, labelspacing=0.4)
    fig.text(0.665, 0.975, "b  Common threshold, 19 models", ha="left", va="top",
             fontsize=8, color=INK)
    save(fig, "Fig3_modern_noise", 3)


# ------------------------------------------------------------------ value table
def write_values():
    p = os.path.join(RES, "FIGURE_VALUES_V6.md")
    w = []
    w.append("# Figure values, manuscript V6 (Figures 1 to 3)")
    w.append("")
    w.append("Written by `scripts/v7/fig_v6.py` while drawing; every value below is the exact number "
             "plotted, read from the JSON path given (paths relative to `rebuild/results/v7/`). "
             "Values are shown to 6 decimals; the figures themselves print no numbers except the two "
             "reference values in the Figure 3b legend (2 decimals).")
    w.append("")
    w.append("## Definitions")
    w.append("")
    w.append("- Agreement: Spearman correlation of per-method mean scores (Dice) between two image sets over a method set.")
    w.append("- Figure 1: `comp|<domain>|<category>` = mean agreement over the dataset pairs of that category "
             "(domain with domain; domain with other; other datasets, same modality; other datasets, across modality). "
             "Interval: method-only bootstrap, stats_v5 design, B 2000, seed 0 (`method_only_repro_B2000`), the design of V5 Figure 1.")
    w.append("- Figure 2: benchmark = `mean|bench` (generator B, texture amplitude 40, mean over synthetic seeds and over the 4 wound references); "
             "wound with each other = `pairs6` (mean of the 6 wound pairs; equals `mean|other`). Interval: method bootstrap B 2000, default_rng(0).")
    w.append("- Figure 3a (C16, 16 architectures with 3 seeds): per wound dataset r-bar = mean of the 3 seed-pair Spearman values on that dataset "
             "(interval `rbar_method_boot`); `rbar_wound` = mean over the 4 wound datasets; `single|wound pairs` = wound-pair mean at one seed, "
             "averaged over seeds 0, 1, 2; `crossseed|wound pairs` = seed s on one dataset vs seed t on the other (s not t), 36 values; "
             "`rbar_control` = r-bar on the AZH train slice images; `single|same dataset` = AZH vs AZH train slice at one seed, averaged over seeds. "
             "Per-pair values (`observed_single_mean`) have no stored interval. Intervals: architecture bootstrap B 2000, seed 7070 (`method_boot`).")
    w.append("- Figure 3b: M19 wound-pair agreement with one common grid threshold for all 19 models (P0 pipeline); interval "
             "`ci` = method bootstrap B 2000, seed 0. Filled marker where the paired change from the own-threshold value has a "
             "95% interval with lower bound above 0 (`delta_vs_default`).")
    w.append("")
    w.append("## Checks run before drawing")
    w.append("")
    for c in CHECKS:
        w.append(f"- {c}")
    w.append("")
    for figname in ("Figure 1", "Figure 2", "Figure 3"):
        w.append(f"## {figname}")
        w.append("")
        w.append("| panel | row / series | quantity | value | JSON path |")
        w.append("|---|---|---|---|---|")
        for f, pn, row, q, v, path in RECORDS:
            if f == figname:
                w.append(f"| {pn} | {row} | {q} | {v:.6f} | `{path}` |")
        w.append("")
    w.append(f"Total plotted values: {len(RECORDS)}.")
    w.append("")
    open(p, "w").write("\n".join(w))
    print("wrote", p, len(RECORDS), "values")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    write_values()

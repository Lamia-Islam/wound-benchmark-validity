#!/usr/bin/env python3
"""V7 training-noise control: how much of the modern-model ranking is training
seed noise? Writes results/v7/seeds_v7.json and results/v7/SEEDS_V7.md.

Data: scripts/perimage_v7.py (V6 exclusions and images; per-seed matrices from
load_seed). An architecture's score on a dataset = mean over images of its
per-image scores (Dice or NSD). Agreement = Spearman correlation (Pearson on
average ranks, ties averaged) between two vectors of architecture scores over
the architectures present in both.

Seeds available (three trainings diverged and were not saved):
  manet_r18 seeds {0, 1}; manet_r50 seed {1}; every other architecture {0, 1, 2}.
  nnU-Net has a single model and is never part of a cross-seed comparison.

Sections
  a  same dataset, different seed: Spearman between the seed-a and seed-b
     architecture scores of one dataset, seed pairs (0,1), (0,2), (1,2),
     architecture subsets old 10, new 8, all 18 (only architectures with both seeds)
  b  cross-dataset agreement among the 4 wound datasets (6 pairs):
     seed-averaged scores over M18p (18 architectures) and M19 (+ nnU-Net);
     per seed (seed-s models only, with and without nnU-Net);
     matched design: seed a on dataset 1 vs seed b on dataset 2 (a != b), so
     training noise enters exactly as in section a and only the dataset differs;
     control pair azh vs azh_train_slice for every variant
  c  variance split per dataset: between-architecture SD of mean score vs
     within-architecture SD across seeds; one-way ANOVA ICC on the 16
     architectures with all 3 seeds (balanced), plus a pooled within-SD that
     also uses manet_r18 (2 seeds); manet_r50 (1 seed) enters only the
     between-architecture SD of the 18 seed-averaged means
  d  the pre-specified question of BRIEF_V7.md, wound datasets reported apart
     from colonoscopy / dermoscopy / ultrasound; architecture bootstrap of the
     key difference (B = 2000, seed 7007)
  e  checks: seed-0 means of the old 10 equal the V6 M columns; seed-averaged
     columns of load7 equal the mean of the per-seed means; one Spearman value
     recomputed with scipy.stats.spearmanr
Run from the rebuild folder:  python scripts/v7/seeds_v7.py
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage_v6 as V6                                           # noqa: E402
import perimage_v7 as P                                            # noqa: E402

OUT = os.path.join(ROOT, "results", "v7")
METRICS = ("dice", "nsd")
REFS = list(P.REFS)
DS = REFS + [P.CONTROL]
WOUND = list(P.WOUND)
NONWOUND = [r for r in REFS if r not in WOUND]
COLON = list(P.COLON)
OTHER = [r for r in NONWOUND if r not in COLON]            # ph2, tn3k_test
SEED_PAIRS = [(0, 1), (0, 2), (1, 2)]
SUBSETS = {"old10": P.ARCH_OLD, "new8": P.ARCH_NEW, "all18": P.ARCH}
WPAIRS = list(itertools.combinations(WOUND, 2))            # 6 wound pairs
B_BOOT, SEED_BOOT = 2000, 7007
assert len(WPAIRS) == 6 and len(REFS) == 11


def sp(a, b):
    """Spearman = Pearson on average ranks; nan if either vector is constant."""
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ------------------------------------------------------------------ data
# MEAN[m][d][s] = {arch: dataset mean of the seed-s model}
MEAN = {m: {d: {} for d in DS} for m in METRICS}
NIMG = {}
for m in METRICS:
    for d in DS:
        for s in P.SEEDS:
            X, archs = P.load_seed(d, m, s)
            MEAN[m][d][s] = dict(zip(archs, X.mean(0).astype(float)))
            NIMG[d] = int(X.shape[0])
SEEDS_OF = {a: [s for s in P.SEEDS if a in MEAN["dice"]["medetec"][s]] for a in P.ARCH}
for m in METRICS:
    for d in DS:
        for a in P.ARCH:
            assert [s for s in P.SEEDS if a in MEAN[m][d][s]] == SEEDS_OF[a], (m, d, a)
assert SEEDS_OF["manet_r18"] == [0, 1] and SEEDS_OF["manet_r50"] == [1]
assert all(SEEDS_OF[a] == [0, 1, 2] for a in P.ARCH if a not in P.MANET)
AVG = {m: {d: {a: float(np.mean([MEAN[m][d][s][a] for s in SEEDS_OF[a]])) for a in P.ARCH}
           for d in DS} for m in METRICS}
NNU = {m: {d: float(V6.load(d, m)["X"][:, V6.NAMES_ALL.index("nnunet_2d")].mean()) for d in DS}
       for m in METRICS}
M19MEAN = {d: float(np.mean(list(AVG["dice"][d].values()) + [NNU["dice"][d]])) for d in DS}

S = {"definitions": {
    "score": "architecture score on a dataset = mean over images (V6 exclusions) of its per-image Dice or NSD",
    "agreement": "Spearman (Pearson on average ranks, ties averaged) over the architectures present in both vectors",
    "seeds_available": {a: SEEDS_OF[a] for a in P.ARCH},
    "seed_averaged": "mean over the architecture's available seeds (as perimage_v7 load7)",
    "wound_datasets": WOUND, "colonoscopy_datasets": COLON, "other_nonwound": OTHER,
    "wound_pairs": [f"{a}~{b}" for a, b in WPAIRS],
    "control_pair": "azh~azh_train_slice",
    "floor_rule": "M19 group mean Dice < 0.1 (V6 floor rule)"},
    "n_images": NIMG,
    "m19_mean_dice": M19MEAN,
    "floor_datasets": [d for d in DS if M19MEAN[d] < 0.1]}

# ============================================================ e  checks
chk_s0, chk_avg, n_vec_equal = [], [], 0
for m in METRICS:
    for d in DS:
        D6 = V6.load(d, m)
        X0, a0 = P.load_seed(d, m, 0)
        for a in P.ARCH_OLD:
            col = D6["X"][:, V6.NAMES_ALL.index(f"{a}_in")]
            n_vec_equal += int(np.array_equal(col, X0[:, a0.index(a)]))
            v6 = float(col.mean())
            chk_s0.append(abs(v6 - MEAN[m][d][0][a]))
        D7 = P.load7(d, m)
        for a in P.ARCH:
            chk_avg.append(abs(float(D7["X"][:, D7["names"].index(f"{a}@avg")].mean()) - AVG[m][d][a]))
x0 = [MEAN["dice"]["azh"][0][a] for a in P.ARCH if 0 in SEEDS_OF[a] and 1 in SEEDS_OF[a]]
x1 = [MEAN["dice"]["azh"][1][a] for a in P.ARCH if 0 in SEEDS_OF[a] and 1 in SEEDS_OF[a]]
S["checks"] = {
    "seed0_old10_vs_v6_M_columns": {"n_values": len(chk_s0), "max_abs_diff": float(max(chk_s0)),
                                    "n_per_image_vectors_identical": n_vec_equal,
                                    "what": "dataset means of the 10 old architectures, seed 0 (load_seed) vs V6 '<arch>_in' columns, 12 datasets x 2 metrics; differences are float summation order only"},
    "load7_avg_vs_mean_of_seed_means": {"n_values": len(chk_avg), "max_abs_diff": float(max(chk_avg))},
    "spearman_vs_scipy_azh_dice_seeds01": {"own": sp(x0, x1), "scipy": float(spearmanr(x0, x1)[0]), "n": len(x0)}}
assert S["checks"]["seed0_old10_vs_v6_M_columns"]["max_abs_diff"] < 1e-12 and n_vec_equal == len(chk_s0)
assert S["checks"]["load7_avg_vs_mean_of_seed_means"]["max_abs_diff"] < 1e-12
assert abs(S["checks"]["spearman_vs_scipy_azh_dice_seeds01"]["own"] -
           S["checks"]["spearman_vs_scipy_azh_dice_seeds01"]["scipy"]) < 1e-12
print("checks ok", S["checks"], flush=True)

# ============================================================ a  cross-seed
def cross_seed(m, d, archs, sa, sb):
    common = [a for a in archs if a in MEAN[m][d][sa] and a in MEAN[m][d][sb]]
    return {"rho": sp([MEAN[m][d][sa][a] for a in common], [MEAN[m][d][sb][a] for a in common]),
            "n": len(common)}


XS = {}
for m in METRICS:
    for d in DS:
        for sub, archs in SUBSETS.items():
            e = {f"{a},{b}": cross_seed(m, d, archs, a, b) for a, b in SEED_PAIRS}
            e["mean_of_3_pairs"] = nanmean([e[f"{a},{b}"]["rho"] for a, b in SEED_PAIRS])
            XS[f"{m}|{d}|{sub}"] = e


def xs_summary(m, sub, dsets):
    v = [XS[f"{m}|{d}|{sub}"][f"{a},{b}"]["rho"] for d in dsets for a, b in SEED_PAIRS]
    return {"mean": nanmean(v), "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
            "n_values": len(v), "n_datasets": len(dsets),
            "n_archs_per_pair": sorted({XS[f"{m}|{d}|{sub}"][f"{a},{b}"]["n"] for d in dsets for a, b in SEED_PAIRS})}


XSS = {}
for m in METRICS:
    for sub in SUBSETS:
        XSS[f"{m}|{sub}"] = {"wound (4)": xs_summary(m, sub, WOUND),
                             "colonoscopy (5)": xs_summary(m, sub, COLON),
                             "dermoscopy + ultrasound (2)": xs_summary(m, sub, OTHER),
                             "non-wound (7)": xs_summary(m, sub, NONWOUND),
                             "all 11 REFS": xs_summary(m, sub, REFS),
                             "control azh_train_slice": xs_summary(m, sub, [P.CONTROL])}
S["cross_seed"] = XS
S["cross_seed_summary"] = XSS

# ============================================================ b  cross-dataset
def vec(m, d, s, archs, with_nnu=False):
    src = AVG[m][d] if s == "avg" else MEAN[m][d][s]
    v = [src[a] for a in archs]
    return np.array(v + ([NNU[m][d]] if with_nnu else []))


def cd_block(m, s, archs, with_nnu):
    pairs = {f"{a}~{b}": sp(vec(m, a, s, archs, with_nnu), vec(m, b, s, archs, with_nnu)) for a, b in WPAIRS}
    return {"wound_pairs_mean": nanmean(list(pairs.values())), "pairs": pairs,
            "control": sp(vec(m, "azh", s, archs, with_nnu), vec(m, P.CONTROL, s, archs, with_nnu)),
            "n": len(archs) + int(with_nnu)}


CD = {}
for m in METRICS:
    CD[f"{m}|M18p seed-averaged"] = cd_block(m, "avg", P.ARCH, False)
    CD[f"{m}|M19 seed-averaged"] = cd_block(m, "avg", P.ARCH, True)
    for s in P.SEEDS:
        archs = [a for a in P.ARCH if s in SEEDS_OF[a]]
        CD[f"{m}|seed {s} architectures"] = cd_block(m, s, archs, False)
        CD[f"{m}|seed {s} architectures + nnU-Net"] = cd_block(m, s, archs, True)
    # matched: seed a on dataset 1 vs seed b on dataset 2, both orders, a != b
    vals, perpair = [], {}
    for sa, sb in SEED_PAIRS:
        archs = [a for a in P.ARCH if sa in SEEDS_OF[a] and sb in SEEDS_OF[a]]
        for d1, d2 in WPAIRS:
            r = [sp(vec(m, d1, sa, archs), vec(m, d2, sb, archs)),
                 sp(vec(m, d1, sb, archs), vec(m, d2, sa, archs))]
            perpair[f"{sa},{sb}|{d1}~{d2}"] = {"rho_both_orders": r, "n": len(archs)}
            vals += r
    ctrl = []
    for sa, sb in SEED_PAIRS:
        archs = [a for a in P.ARCH if sa in SEEDS_OF[a] and sb in SEEDS_OF[a]]
        ctrl += [sp(vec(m, "azh", sa, archs), vec(m, P.CONTROL, sb, archs)),
                 sp(vec(m, "azh", sb, archs), vec(m, P.CONTROL, sa, archs))]
    same = [XS[f"{m}|{d}|all18"][f"{a},{b}"]["rho"] for d in WOUND for a, b in SEED_PAIRS]
    CD[f"{m}|matched different seeds"] = {
        "cross_dataset_mean": nanmean(vals), "n_values": len(vals),
        "same_dataset_cross_seed_mean": nanmean(same), "n_values_same": len(same),
        "difference_same_minus_cross": nanmean(same) - nanmean(vals),
        "control_cross_seed_mean": nanmean(ctrl), "per_pair": perpair,
        "n_archs_per_seed_pair": {f"{a},{b}": len([x for x in P.ARCH if a in SEEDS_OF[x] and b in SEEDS_OF[x]])
                                  for a, b in SEED_PAIRS}}
S["cross_dataset_wound"] = CD

# ============================================================ c  variance split
BAL = [a for a in P.ARCH if SEEDS_OF[a] == [0, 1, 2]]                # 16 architectures
MULTI = [a for a in P.ARCH if len(SEEDS_OF[a]) >= 2]                 # + manet_r18
assert len(BAL) == 16 and len(MULTI) == 17


def var_split(m, d):
    Y = np.array([[MEAN[m][d][s][a] for s in P.SEEDS] for a in BAL])          # 16 x 3
    k, n = Y.shape[1], Y.shape[0]
    am = Y.mean(1)
    msb = k * ((am - am.mean()) ** 2).sum() / (n - 1)
    msw = ((Y - am[:, None]) ** 2).sum() / (n * (k - 1))
    icc1 = (msb - msw) / (msb + (k - 1) * msw)
    icck = (msb - msw) / msb
    # pooled within-SD with manet_r18 (2 seeds) added: df-weighted
    ss, df = 0.0, 0
    for a in MULTI:
        v = np.array([MEAN[m][d][s][a] for s in SEEDS_OF[a]])
        ss += ((v - v.mean()) ** 2).sum()
        df += len(v) - 1
    avg18 = np.array([AVG[m][d][a] for a in P.ARCH])
    return {"between_sd_avg18": float(avg18.std(ddof=1)), "n_between_avg18": 18,
            "between_sd_bal16": float(am.std(ddof=1)), "n_between_bal16": n,
            "within_sd_bal16": float(np.sqrt(msw)), "df_within_bal16": int(n * (k - 1)),
            "within_sd_pooled17": float(np.sqrt(ss / df)), "df_within_pooled17": int(df),
            "ratio_between_to_within_sd_bal16": float(am.std(ddof=1) / np.sqrt(msw)) if msw > 0 else float("nan"),
            "icc1_single_seed": float(icc1), "icc_k3_seed_mean": float(icck),
            "MSB": float(msb), "MSW": float(msw), "group_mean": float(Y.mean())}


VS = {f"{m}|{d}": var_split(m, d) for m in METRICS for d in DS}
S["variance_split"] = VS
S["variance_split_definitions"] = {
    "balanced_16": BAL, "pooled_17_adds": "manet_r18 (2 seeds, df 1)", "single_seed_left_out_of_within": ["manet_r50"],
    "between_sd_avg18": "SD (ddof 1) of the 18 seed-averaged architecture means",
    "between_sd_bal16": "SD (ddof 1) of the 16 balanced architectures' 3-seed means",
    "within_sd_bal16": "sqrt(MSW): pooled SD across seeds within architecture, 16 architectures x 3 seeds, df 32",
    "icc1_single_seed": "(MSB - MSW) / (MSB + 2 MSW): share of the variance of one-seed architecture means that is between architectures",
    "icc_k3_seed_mean": "(MSB - MSW) / MSB: same for the mean of 3 seeds (reliability of a seed-averaged architecture mean)",
    "note": "ICC can be negative when between-architecture differences are smaller than seed noise"}

# ============================================================ d  bootstrap of the key comparison
# Architectures resampled with replacement (18 draws from the 18), B = 2000.
# For each draw: same-dataset cross-seed mean over the 4 wound datasets x 3 seed
# pairs, matched cross-dataset mean (6 wound pairs x 3 seed pairs x 2 orders),
# and their difference. Only drawn architectures having both seeds of a pair enter it.
rng = np.random.default_rng(SEED_BOOT)
TAB = {m: {d: np.array([[MEAN[m][d][s].get(a, np.nan) for s in P.SEEDS] for a in P.ARCH]) for d in DS}
       for m in METRICS}


def zr(T):
    r = rankdata(T, axis=-1)
    r = r - r.mean(-1, keepdims=True)
    n = np.sqrt((r ** 2).sum(-1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return r / n


def key_stats(m, idx):
    """(same-dataset cross-seed mean, matched cross-dataset mean) over the
    architectures idx (indices into P.ARCH, repeats allowed)."""
    sv, cv = [], []
    for sa, sb in SEED_PAIRS:
        ok = idx[np.isfinite(TAB[m]["medetec"][idx, sa]) & np.isfinite(TAB[m]["medetec"][idx, sb])]
        Za = {d: zr(TAB[m][d][ok, sa]) for d in WOUND}
        Zb = {d: zr(TAB[m][d][ok, sb]) for d in WOUND}
        for d in WOUND:
            sv.append((Za[d] * Zb[d]).sum())
        for d1, d2 in WPAIRS:
            cv += [(Za[d1] * Zb[d2]).sum(), (Zb[d1] * Za[d2]).sum()]
    return np.nanmean(sv), np.nanmean(cv)


draws = rng.integers(0, len(P.ARCH), size=(B_BOOT, len(P.ARCH)))
BOOT = {}
for m in METRICS:
    pt = CD[f"{m}|matched different seeds"]
    s_full, c_full = key_stats(m, np.arange(len(P.ARCH)))
    assert abs(s_full - pt["same_dataset_cross_seed_mean"]) < 1e-12 and abs(c_full - pt["cross_dataset_mean"]) < 1e-12
    same_b = np.zeros(B_BOOT)
    cross_b = np.zeros(B_BOOT)
    for b in range(B_BOOT):
        same_b[b], cross_b[b] = key_stats(m, draws[b])
    diff = same_b - cross_b
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(same_b > 0, cross_b / same_b, np.nan)
    BOOT[m] = {"same_dataset_cross_seed": {"est": pt["same_dataset_cross_seed_mean"], "ci": [float(np.nanpercentile(same_b, 2.5)), float(np.nanpercentile(same_b, 97.5))]},
               "cross_dataset_matched": {"est": pt["cross_dataset_mean"], "ci": [float(np.nanpercentile(cross_b, 2.5)), float(np.nanpercentile(cross_b, 97.5))]},
               "difference": {"est": pt["difference_same_minus_cross"], "ci": [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
                              "p_le0": float(np.mean(diff <= 0))},
               "seed_noise_corrected_ratio": {"est": float(c_full / s_full), "ci": [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))],
                                              "n_undefined": int((~np.isfinite(ratio)).sum()),
                                              "what": "matched cross-dataset mean / same-dataset cross-seed mean (ratio of means; the V6 disattenuation formula with seed-to-seed agreement as the reliability)"},
               "check_full_sample_reproduces_point": True,
               "B": B_BOOT, "seed": SEED_BOOT, "n_archs_drawn": len(P.ARCH)}
S["bootstrap_key_comparison"] = BOOT

# ============================================================ d  pre-specified verdict
VERDICT = {}
for m in METRICS:
    xs_w = XSS[f"{m}|all18"]["wound (4)"]["mean"]
    xs_nw = XSS[f"{m}|all18"]["non-wound (7)"]["mean"]
    cd_m19 = CD[f"{m}|M19 seed-averaged"]["wound_pairs_mean"]
    cd_seed = nanmean([CD[f"{m}|seed {s} architectures"]["wound_pairs_mean"] for s in P.SEEDS])
    VERDICT[m] = {"cross_seed_wound_all18": xs_w, "cross_seed_nonwound_all18": xs_nw,
                  "cross_seed_all11_all18": XSS[f"{m}|all18"]["all 11 REFS"]["mean"],
                  "cross_dataset_wound_M19": cd_m19,
                  "cross_dataset_wound_M18p": CD[f"{m}|M18p seed-averaged"]["wound_pairs_mean"],
                  "cross_dataset_wound_per_seed_mean": cd_seed,
                  "cross_dataset_wound_matched": CD[f"{m}|matched different seeds"]["cross_dataset_mean"],
                  "seed_noise_corrected_ratio": BOOT[m]["seed_noise_corrected_ratio"]["est"],
                  "spearman_brown_3seed_wound": 3 * xs_w / (1 + 2 * xs_w),
                  "criterion_cross_seed_gt_0.7_wound": bool(xs_w > 0.7),
                  "criterion_cross_dataset_in_0.1_to_0.3": bool(0.1 <= cd_m19 <= 0.3)}
S["prespecified"] = VERDICT
json.dump(S, open(os.path.join(OUT, "seeds_v7.json"), "w"), indent=1)
print("saved results/v7/seeds_v7.json", flush=True)

# ================================================================ markdown
def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


L = []
w = L.append
w("# SEEDS V7: training-noise control for the modern models\n")
w("Generated by `scripts/v7/seeds_v7.py` from `scripts/perimage_v7.py`. All numbers in `results/v7/seeds_v7.json`.\n")
w("## Definitions\n")
w("- Architecture score on a dataset: mean over images (V6 exclusions applied) of the per-image Dice or NSD of one trained model.")
w("- Agreement: Spearman correlation (Pearson on average ranks, ties averaged) over the architectures present in both vectors. n = number of architectures.")
w("- Seeds available: every architecture has seeds 0, 1, 2 except manet_r18 (0, 1) and manet_r50 (1 only). The three missing trainings diverged and were not saved. "
  "So a seed pair has 17 architectures for (0,1) and 16 for (0,2) and (1,2) (old 10: 10, 9, 9; new 8: 7, 7, 7). nnU-Net has one model and never enters a cross-seed comparison.")
w("- Wound datasets: " + ", ".join(WOUND) + " (6 pairs). Colonoscopy: " + ", ".join(COLON) + ". Other: " + ", ".join(OTHER) + ". Control: azh vs azh_train_slice.")
w("- Floor datasets (M19 group mean Dice below 0.1, V6 rule): " + ", ".join(f"{d} {M19MEAN[d]:.3f}" for d in S["floor_datasets"]) +
  ". On these, many architectures score near or exactly 0 (tn3k_test has several exact zeros, which tie), so their ranks are close to noise.\n")

w("## Checks\n")
c = S["checks"]
w(f"- Seed-0 dataset means of the old 10 architectures vs the V6 M columns ({c['seed0_old10_vs_v6_M_columns']['n_values']} values, 12 datasets x 2 metrics x 10): max abs diff {c['seed0_old10_vs_v6_M_columns']['max_abs_diff']:.1e} (float summation order); the per-image vectors are identical in {c['seed0_old10_vs_v6_M_columns']['n_per_image_vectors_identical']} of {c['seed0_old10_vs_v6_M_columns']['n_values']}.")
w(f"- Seed-averaged columns of load7 vs mean of the per-seed dataset means ({c['load7_avg_vs_mean_of_seed_means']['n_values']} values): max abs diff {c['load7_avg_vs_mean_of_seed_means']['max_abs_diff']:.1e}.")
c3 = c["spearman_vs_scipy_azh_dice_seeds01"]
w(f"- Spearman, azh Dice seed 0 vs seed 1 (n {c3['n']}): own {c3['own']:.6f}, scipy {c3['scipy']:.6f}.\n")

w("## a. Same dataset, different seed\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| dataset | M19 mean Dice | old10 (0,1) | (0,2) | (1,2) | new8 (0,1) | (0,2) | (1,2) | all18 (0,1) | (0,2) | (1,2) | all18 mean |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        cells = []
        for sub in SUBSETS:
            for a, b in SEED_PAIRS:
                e = XS[f"{m}|{d}|{sub}"][f"{a},{b}"]
                cells.append(f"{f3(e['rho'])} (n {e['n']})")
        w(f"| {d} | {M19MEAN[d]:.3f} | " + " | ".join(cells) + f" | {f3(XS[f'{m}|{d}|all18']['mean_of_3_pairs'])} |")
    w(f"\n{m}: mean cross-seed Spearman by dataset category (mean over datasets x 3 seed pairs; min to max):\n")
    w("| category | old10 | new8 | all18 |")
    w("|---|---|---|---|")
    for cat in XSS[f"{m}|all18"]:
        cells = []
        for sub in SUBSETS:
            e = XSS[f"{m}|{sub}"][cat]
            cells.append(f"{f3(e['mean'])} ({f3(e['min'])} to {f3(e['max'])}; {e['n_values']} values, n archs {'/'.join(map(str, e['n_archs_per_pair']))})")
        w(f"| {cat} | " + " | ".join(cells) + " |")

w("\n## b. Cross-dataset agreement among the wound datasets\n")
w("Mean Spearman over the 6 wound pairs. 'Matched' = seed a on one dataset vs seed b on the other (a != b, both orders, 3 seed pairs, 36 values), "
  "so training noise is present exactly as in the same-dataset cross-seed values and only the dataset differs.\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| method set | n | wound pairs mean | " + " | ".join(f"{a}~{b}" for a, b in WPAIRS) + " | control azh~azh_train_slice |")
    w("|---|---|---|" + "---|" * 6 + "---|")
    for k, v in CD.items():
        if not k.startswith(m + "|") or "matched" in k:
            continue
        w(f"| {k.split('|')[1]} | {v['n']} | {f3(v['wound_pairs_mean'])} | " + " | ".join(f3(v["pairs"][f"{a}~{b}"]) for a, b in WPAIRS) + f" | {f3(v['control'])} |")
    mt = CD[f"{m}|matched different seeds"]
    w(f"\nMatched design ({m}, n archs per seed pair " + ", ".join(f"({k}) {v}" for k, v in mt["n_archs_per_seed_pair"].items()) + "):")
    w(f"- same dataset, different seed (4 wound datasets x 3 seed pairs, {mt['n_values_same']} values): {f3(mt['same_dataset_cross_seed_mean'])}")
    w(f"- different dataset, different seed (6 wound pairs x 3 seed pairs x 2 orders, {mt['n_values']} values): {f3(mt['cross_dataset_mean'])}")
    w(f"- control azh vs azh_train_slice, different seed (6 values): {f3(mt['control_cross_seed_mean'])}")
    bb = BOOT[m]
    w(f"- architecture bootstrap (B {B_BOOT}, 18 architectures resampled): same-dataset {f3(bb['same_dataset_cross_seed']['est'])} "
      f"[{f3(bb['same_dataset_cross_seed']['ci'][0])}, {f3(bb['same_dataset_cross_seed']['ci'][1])}], cross-dataset "
      f"{f3(bb['cross_dataset_matched']['est'])} [{f3(bb['cross_dataset_matched']['ci'][0])}, {f3(bb['cross_dataset_matched']['ci'][1])}], "
      f"difference {f3(bb['difference']['est'])} [{f3(bb['difference']['ci'][0])}, {f3(bb['difference']['ci'][1])}], share of draws with difference <= 0: {bb['difference']['p_le0']:.3f}.")
    rr = bb["seed_noise_corrected_ratio"]
    w(f"- seed-noise-corrected cross-dataset agreement (cross-dataset mean / same-dataset cross-seed mean, the V6 disattenuation ratio with seed-to-seed agreement as reliability): "
      f"{f3(rr['est'])} [{f3(rr['ci'][0])}, {f3(rr['ci'][1])}] ({rr['n_undefined']} undefined draws). 1 would mean the datasets rank the architectures the same apart from seed noise.")
w("\n## c. Variance split: between architectures vs between seeds\n")
w("Balanced design: the 16 architectures with all 3 seeds. Between SD = SD of their 3-seed means (n 16); also SD of all 18 seed-averaged means (n 18; manet_r50 is a single seed, manet_r18 a 2-seed mean). "
  "Within SD = sqrt(MSW), pooled SD across seeds (16 x 3, df 32); pooled-17 adds manet_r18 (df 33). manet_r50 has one seed and is left out of every within-SD. "
  "ICC(1) = (MSB - MSW)/(MSB + 2 MSW): share of the variance of single-seed architecture means that is between architectures. "
  "ICC(3) = (MSB - MSW)/MSB: reliability of a 3-seed mean. Negative ICC means seed noise exceeds the architecture differences.\n")
for m in METRICS:
    w(f"\n### {m}\n")
    w("| dataset | mean of the 16 | between SD (18 avg) | between SD (16) | within SD (16, df 32) | within SD (17, df 33) | between/within (16) | ICC(1) single seed | ICC(3) 3-seed mean |")
    w("|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        v = VS[f"{m}|{d}"]
        w(f"| {d} | {v['group_mean']:.3f} | {v['between_sd_avg18']:.4f} | {v['between_sd_bal16']:.4f} | {v['within_sd_bal16']:.4f} | "
          f"{v['within_sd_pooled17']:.4f} | {f3(v['ratio_between_to_within_sd_bal16'])} | {f3(v['icc1_single_seed'])} | {f3(v['icc_k3_seed_mean'])} |")

w("\n## d. Pre-specified question\n")
w("Brief: the claim 'modern models disagree across wound datasets because the datasets differ, not because of training noise' is supported if "
  "same-dataset cross-seed agreement is high (> 0.7) while cross-dataset wound agreement stays low (about 0.1 to 0.3).\n")
w("| metric | cross-seed, wound (all18, 12 values) | cross-seed, non-wound (all18, 21 values) | cross-dataset wound M19 (n 19, 6 pairs) | M18p (n 18) | per-seed mean (3 seeds) | matched (36 values) | cross-seed > 0.7 on wound? | M19 cross-dataset in 0.1 to 0.3? |")
w("|---|---|---|---|---|---|---|---|---|")
for m in METRICS:
    v = VERDICT[m]
    w(f"| {m} | {f3(v['cross_seed_wound_all18'])} | {f3(v['cross_seed_nonwound_all18'])} | {f3(v['cross_dataset_wound_M19'])} | {f3(v['cross_dataset_wound_M18p'])} | "
      f"{f3(v['cross_dataset_wound_per_seed_mean'])} | {f3(v['cross_dataset_wound_matched'])} | {'yes' if v['criterion_cross_seed_gt_0.7_wound'] else 'no'} | "
      f"{'yes' if v['criterion_cross_dataset_in_0.1_to_0.3'] else 'no'} |")


def iv(e):
    return f"{f3(e['est'])} [{f3(e['ci'][0])}, {f3(e['ci'][1])}]"


w("\n### Plain reading\n")
for m in METRICS:
    v, bb = VERDICT[m], BOOT[m]
    xw = XSS[f"{m}|all18"]["wound (4)"]
    ok1, ok2 = v["criterion_cross_seed_gt_0.7_wound"], v["criterion_cross_dataset_in_0.1_to_0.3"]
    dlo = bb["difference"]["ci"][0]
    w(f"**{m}.** On the 4 wound datasets, two training seeds of the same architectures agree at Spearman {f3(xw['mean'])} on average "
      f"(12 values, 16 or 17 architectures, range {f3(xw['min'])} to {f3(xw['max'])}). "
      + ("This is above the 0.7 the brief requires. " if ok1 else "This is below the 0.7 the brief requires, so the first condition fails. ")
      + f"Seed-averaged cross-dataset wound agreement is {f3(v['cross_dataset_wound_M19'])} for M19 (n 19, 6 pairs) and {f3(v['cross_dataset_wound_M18p'])} for M18p (n 18)"
      + (", inside 0.1 to 0.3. " if ok2 else ", outside 0.1 to 0.3. ")
      + ("The pre-specified claim is supported. " if (ok1 and ok2) else "The pre-specified claim is not supported as stated. ")
      + f"In the matched design, where both comparisons use different seeds, same-dataset agreement is {f3(bb['same_dataset_cross_seed']['est'])} and cross-dataset agreement is "
      f"{f3(bb['cross_dataset_matched']['est'])}; difference {iv(bb['difference'])}. "
      + ("The interval excludes 0: the datasets add disagreement beyond seed noise, but seed noise alone already removes a large part of the ranking. "
         if dlo > 0 else
         "The interval includes 0: cross-seed agreement is statistically as low as cross-dataset agreement, so for this metric the modern models' wound rankings cannot be told apart from training noise. ")
      + f"Corrected for seed noise, cross-dataset agreement is {iv(bb['seed_noise_corrected_ratio'])} of what seed noise allows. "
      f"Cross-seed agreement on the non-wound datasets is {f3(v['cross_seed_nonwound_all18'])} (21 values), on the wound datasets {f3(xw['mean'])}: "
      + ("the low seed stability is not a floor-dataset artefact; it is present on the wound datasets where the models work. " if xw["mean"] <= 0.7 else "")
      + f"By subset (wound, mean of 12): old 10 {f3(XSS[f'{m}|old10']['wound (4)']['mean'])} (n 9 or 10), new 8 {f3(XSS[f'{m}|new8']['wound (4)']['mean'])} (n 7). "
      f"Per-seed cross-dataset wound agreement (architectures only): seed 0 {f3(CD[f'{m}|seed 0 architectures']['wound_pairs_mean'])} (n 17), "
      f"seed 1 {f3(CD[f'{m}|seed 1 architectures']['wound_pairs_mean'])} (n 18), seed 2 {f3(CD[f'{m}|seed 2 architectures']['wound_pairs_mean'])} (n 16). "
      f"A rough Spearman-Brown projection of the wound cross-seed value to a 3-seed mean gives {f3(v['spearman_brown_3seed_wound'])} "
      "(M19 averages 3 seeds for 16 of its 18 architectures).\n")
open(os.path.join(OUT, "SEEDS_V7.md"), "w").write("\n".join(L))
print("saved results/v7/SEEDS_V7.md")

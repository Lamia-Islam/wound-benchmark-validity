#!/usr/bin/env python3
"""V7 practical analysis: what the phase-2 modern group means for someone
choosing a wound-segmentation model.

Sections 1 to 4 of scripts/v6/practical_v6.py rerun on M19. practical_v6 runs
its whole analysis at import time, so its logic is COPIED here (functions
boot_weights, the per-set block, selection, miss_stats, mean_pair_rho), and
generalised in one way: a candidate "model" may have several members (training
seeds). Rerunning the copy on M11 must reproduce practical_v6.json exactly;
that is checked below.

Definitions (as V6; per-image clinical Dice from perimage_v6 / perimage_v7)
- Per-image Dice on an empty reference mask is 1 when the prediction is also
  empty and 0 otherwise.
- Candidate sets. M19: 18 ImageNet-pretrained architectures + nnU-Net. An
  architecture's per-image Dice is the mean over its available training seeds
  (perimage_v7.load7), so its mean Dice is the mean of its per-seed mean Dice.
  M11: V6 (10 run-1 seed-0 architectures + nnU-Net). M17: M19 without the two
  MAnet architectures. C52: every trained architecture-seed model as its own
  candidate (51 = 10 run-1 architectures x available seeds (29) + 8 phase-2
  architectures x available seeds (22)) + nnU-Net.
- Mean Dice: mean over ALL kept images, empty references included.
- Bootstrap: resample resampling units (photograph for Medetec and AZH, image
  otherwise), B = 2000, each image of a drawn unit weighted by the unit's draw
  count; the same draws for every candidate and every analysis of one dataset
  (paired). Generator default_rng([20261010, dataset index]) exactly as V6, so
  the M11 and M19 analyses use identical draws. 95% percentile intervals.
- Top: highest point-estimate mean Dice. Indistinguishable from the top: the
  95% paired interval of (top minus candidate) has lower bound <= 0.
- Selection loss A -> B: best mean Dice on B minus mean Dice on B of the
  candidate ranked first on A (A != B, 12 ordered wound pairs). Random choice:
  mean over the candidates of (best on B minus candidate on B).
- Synthetic chooser: generator B at amplitude 40 (and 11); per synthetic seed
  the mean Dice over all 45 images, then the mean over seeds 2024, 7, 101. For
  an architecture the per-image synthetic Dice is first averaged over its
  training seeds (scripts/v7/bench_v7.synth_avg).
- Miss: reference non-empty and Dice < t (t = 0.1 main; 0.05, 0.2
  sensitivity). Miss rate = misses / non-empty images. Dice when detected =
  mean Dice over non-empty images with Dice >= t. For a seed-averaged
  architecture both are computed per training seed and then averaged over
  seeds (the expected value for one trained model of that architecture).
  Sensitivity: miss rate of the seed-averaged per-image Dice.
  Group summary: "pretrained mean" = mean over the pretrained architectures of
  their miss rates. Intervals use the paired unit bootstrap above (weighted
  misses / weighted non-empty images per draw).
- Spearman: scipy.stats.spearmanr (average ranks) across the candidates
  between two wound datasets; mean over the 6 pairs.

Seed-aware analysis (new in V7)
- C52 rerun of top / indistinguishable / selection / synthetic chooser.
- Pairwise distinguishability: for every pair of the 51 trained
  architecture-seed models, whether the paired 95% interval of the difference
  in mean Dice excludes 0, split into same-architecture pairs (49) and
  different-architecture pairs (1226).
- One-training-per-architecture world: pool P_s = architectures that have
  training seed s (as their seed-s model) + nnU-Net (its single model in every
  pool). A chooser looks at the seed-s models on dataset A and picks the
  architecture with the highest mean Dice. The deployed model is that
  architecture trained with seed s' (nnU-Net: the same model), evaluated on B.
  Loss = best mean Dice on B within P_s' minus the deployed model's mean Dice
  on B. Cases: cross-dataset same seed (A != B, s' = s), seed only (A = B,
  s' != s), both (A != B, s' != s). Cases where the chosen architecture has no
  seed s' model are dropped and counted. Random: mean over P_s' of
  (best on B minus candidate on B). Reported over all seed pairs and for the
  requested chooser on seed 0 (s = 0, s' in {1, 2}); also without nnU-Net.

Output: results/v7/practical_v7.json and results/v7/PRACTICAL_V7.md
"""
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
import perimage_v6 as P6                                          # noqa: E402
import perimage_v7 as P7                                          # noqa: E402
import bench_v7 as BV                                             # noqa: E402

SEED = 20261010
B = 2000
THRESH = (0.05, 0.1, 0.2)
WOUND = list(P6.WOUND)
CTRL = P6.CONTROL
SETS = WOUND + [CTRL]
NN = "nnunet_2d"
ARCH = P7.ARCH
TS = P7.SEEDS
OUT = os.path.join(ROOT, "results", "v7")
V6JSON = os.path.join(ROOT, "results", "v6", "practical_v6.json")
M11 = list(P6.GROUPS["M"])
M19 = list(P7.GROUPS7["M19"])
M17 = list(P7.GROUPS7["M17"])
SEEDMODELS = [f"{a}@s{k}" for a in ARCH for k in P7.seeds_of(a)]
C52 = SEEDMODELS + [NN]
assert len(M11) == 11 and len(M19) == 19 and len(M17) == 17 and len(SEEDMODELS) == 51


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def boot_weights(units, rng):
    """Copied from practical_v6."""
    u, inv = np.unique(units, return_inverse=True)
    draws = rng.integers(0, len(u), size=(B, len(u)))
    counts = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    return counts[:, inv]                                           # B x images


DATA = {s: P6.load(s) for s in SETS}
WS = {s: boot_weights(DATA[s]["units"], np.random.default_rng([SEED, k])) for k, s in enumerate(SETS)}
_LS = {}


def seedcol(s, a, k):
    if (s, k) not in _LS:
        _LS[(s, k)] = P7.load_seed(s, "dice", k)
    X, archs = _LS[(s, k)]
    return X[:, archs.index(a)]


def members(s, n):
    """Per-image Dice vectors of the trained models behind candidate n."""
    if n.endswith("@avg"):
        a = n[:-4]
        return [seedcol(s, a, k) for k in P7.seeds_of(a)]
    m = re.match(r"^(.+)@s(\d)$", n)
    if m:
        return [seedcol(s, m.group(1), int(m.group(2)))]
    return [DATA[s]["X"][:, P6.NAMES_ALL.index(n)]]


def score_matrix(s, names):
    return np.stack([np.mean(members(s, n), axis=0) for n in names], 1)


def synth_members(sset, n):
    if n.endswith("@avg"):
        a = n[:-4]
        return [BV.synth_seed(sset, a, k) for k in P7.seeds_of(a)]
    return [BV.synth_X7(sset, [n])[:, 0]]


def is_pretrained(n):
    return n != NN


# ------------------------------------------------------------ core analysis

def miss_stats(s, names, t):
    """Copy of practical_v6.miss_stats, member-averaged."""
    ne = ~DATA[s]["empty"]
    mr, dwd = [], []
    for n in names:
        r_, d_ = [], []
        for x in members(s, n):
            r_.append((ne & (x < t)).sum() / ne.sum())
            det = ne & (x >= t)
            d_.append(x[det].mean() if det.any() else np.nan)
        mr.append(np.mean(r_))
        dd = [v for v in d_ if not np.isnan(v)]
        dwd.append(np.mean(dd) if dd else np.nan)
    return np.array(mr), np.array(dwd)


def mean_pair_rho(vals):
    """Copy of practical_v6.mean_pair_rho."""
    rhos = {}
    for a, b in combinations(WOUND, 2):
        x, y = vals[a], vals[b]
        rho = np.nan if (np.ptp(x) == 0 or np.ptp(y) == 0) else spearmanr(x, y)[0]
        rhos[f"{a}|{b}"] = r4(rho)
    v = [x for x in rhos.values() if x is not None]
    return {"pairs": rhos, "mean": r4(np.mean(v)) if v else None, "n_defined": len(v)}


def analyse(M, with_miss=True, group_keys=("U", "F", "K")):
    """Sections 1 to 4 of practical_v6 on candidate list M. Returns (res, internals)."""
    res = {"models": list(M), "n_models": len(M)}
    per_set, means, BMs = {}, {}, {}
    for s in SETS:
        D = DATA[s]
        X = score_matrix(s, M)
        W = WS[s]
        BM = (W @ X) / W.sum(1, keepdims=True)
        BMs[s] = BM
        mu = X.mean(0)
        means[s] = dict(zip(M, mu))
        top = int(np.argmax(mu))
        rows, n_ind = {}, 0
        for j, n in enumerate(M):
            r = {"mean": r4(mu[j]), "ci": [r4(np.percentile(BM[:, j], 2.5)),
                                           r4(np.percentile(BM[:, j], 97.5))]}
            if j != top:
                d = BM[:, top] - BM[:, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                r["diff_from_top"] = r4(mu[top] - mu[j])
                r["diff_ci"] = [r4(lo), r4(hi)]
                r["indistinguishable"] = bool(lo <= 0)
                n_ind += bool(lo <= 0)
            rows[n] = r
        am = np.argmax(BM, 1)
        per_set[s] = {
            "n_images": int(len(X)), "n_units": int(len(set(D["units"]))),
            "n_empty": int(D["empty"].sum()), "models": rows, "top": M[top],
            "n_indistinguishable_from_top": int(n_ind),
            "share_draws_top_is_top": r4(np.mean(am == top)),
            "share_draws_top_by_model": {M[j]: r4(np.mean(am == j)) for j in range(len(M)) if np.any(am == j)},
            "best_minus_median": r4(mu[top] - np.median(mu)),
            "best_minus_worst": r4(mu[top] - mu.min()), "worst": M[int(np.argmin(mu))]}
    res["per_set"] = per_set

    def loss_of(choice, b):
        v = means[b]
        return max(v.values()) - v[choice]

    chosen = {a: max(means[a], key=means[a].get) for a in SETS}
    mat = {a: {b: r4(loss_of(chosen[a], b)) for b in WOUND} for a in WOUND}
    off = [loss_of(chosen[a], b) for a in WOUND for b in WOUND if a != b]
    rand = {b: r4(np.mean([max(means[b].values()) - means[b][n] for n in M])) for b in WOUND}
    synth, synth_raw = {}, {}
    for amp in (11, 40):
        sm = {}
        for n in M:
            vals = [float(np.mean(np.mean(synth_members(f"genB_amp{amp}_s{seed}", n), axis=0)))
                    for seed in (2024, 7, 101)]
            sm[n] = float(np.mean(vals))
        synth_raw[amp] = dict(sm)
        ch = max(sm, key=sm.get)
        rk = [sm[n] for n in M]
        srt = sorted(sm.values(), reverse=True)
        synth[f"amp{amp}"] = {
            "mean_dice": {n: r4(v) for n, v in sm.items()}, "chosen": ch,
            "chosen_synthetic_dice": r4(srt[0]), "runner_up_synthetic_dice": r4(srt[1]),
            "loss": {b: r4(loss_of(ch, b)) for b in WOUND},
            "mean_loss": r4(np.mean([loss_of(ch, b) for b in WOUND])),
            "spearman_with_wound": {b: r4(spearmanr(rk, [means[b][n] for n in M])[0]) for b in WOUND}}
    gap = {n: max(max(means[b].values()) - means[b][n] for b in WOUND) for n in M}
    gap_where = {n: max(WOUND, key=lambda b: max(means[b].values()) - means[b][n]) for n in M}
    res["selection"] = {
        "chosen_on": chosen, "matrix_rows_choose_on_cols_deploy_on": mat,
        "offdiag_mean": r4(np.mean(off)), "offdiag_min": r4(min(off)), "offdiag_max": r4(max(off)),
        "n_offdiag_pairs": len(off),
        "random_choice_expected_loss": rand, "random_mean_over_B": r4(np.mean(list(rand.values()))),
        "control_chooser_loss": {b: r4(loss_of(chosen[CTRL], b)) for b in WOUND},
        "synthetic": synth,
        "max_gap_from_best": {n: r4(v) for n, v in gap.items()},
        "max_gap_where": gap_where,
        "minimax_model": min(gap, key=gap.get), "minimax_gap": r4(min(gap.values()))}
    internals = {"means": means, "BM": BMs, "synth_raw": synth_raw}
    if not with_miss:
        return res, internals

    pt = [n for n in M if is_pretrained(n)]
    ipt = [M.index(n) for n in pt]
    miss = {}
    for t in THRESH:
        tab, mr_all, dwd_all = {}, {}, {}
        for s in SETS:
            mr, dwd = miss_stats(s, M, t)
            mr_all[s], dwd_all[s] = mr, dwd
            tab[s] = {n: {"miss_rate": r4(mr[j]), "dice_when_detected": r4(dwd[j])} for j, n in enumerate(M)}
            tab[s][f"_pretrained{len(pt)}_mean_miss"] = r4(mr[ipt].mean())
            tab[s]["_nnunet_miss"] = r4(mr[M.index(NN)])
        groups = {}
        for gk in group_keys:
            groups[gk] = {s: r4(miss_stats(s, P7.GROUPS7[gk], t)[0].mean()) for s in SETS}
        miss[str(t)] = {
            "table": tab, "group_mean_miss_rate": groups,
            "spearman_miss_rate": mean_pair_rho(mr_all),
            "spearman_dice_when_detected": mean_pair_rho(dwd_all)}
    res["miss"] = miss
    res["spearman_mean_dice"] = mean_pair_rho({s: np.array([means[s][n] for n in M]) for s in WOUND})
    res["spearman_mean_dice_nonempty"] = mean_pair_rho(
        {s: score_matrix(s, M)[~DATA[s]["empty"]].mean(0) for s in WOUND})
    return res, internals


def miss_intervals(M, t):
    """Paired unit-bootstrap intervals (same draws as above) for the pretrained
    mean miss rate, the nnU-Net miss rate and their difference."""
    out = {}
    pt = [n for n in M if is_pretrained(n)]
    for s in SETS:
        ne = (~DATA[s]["empty"]).astype(float)
        W = WS[s]
        den = W @ ne
        arch_rates, arch_pts = [], []
        for n in pt:
            mem = members(s, n)
            arch_rates.append(np.mean([(W @ (ne * (x < t))) / den for x in mem], axis=0))
            arch_pts.append(np.mean([(ne * (x < t)).sum() / ne.sum() for x in mem]))
        ptb = np.mean(arch_rates, axis=0)
        x = members(s, NN)[0]
        nnb = (W @ (ne * (x < t))) / den
        nnp = (ne * (x < t)).sum() / ne.sum()
        ptp = float(np.mean(arch_pts))
        q = lambda v: [r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))]   # noqa: E731
        out[s] = {"n_nonempty": int(ne.sum()), "n_units": int(len(set(DATA[s]["units"]))),
                  "n_pretrained": len(pt),
                  "pretrained_mean_miss": r4(ptp), "pretrained_ci": q(ptb),
                  "nnunet_miss": r4(nnp), "nnunet_ci": q(nnb),
                  "diff_pretrained_minus_nnunet": r4(ptp - nnp), "diff_ci": q(ptb - nnb)}
    return out


def avg_dice_miss(M, t):
    """Sensitivity: miss rate of the seed-averaged per-image Dice."""
    out = {}
    pt = [n for n in M if is_pretrained(n)]
    for s in SETS:
        ne = ~DATA[s]["empty"]
        X = score_matrix(s, pt)
        out[s] = r4(((X < t) & ne[:, None]).sum(0).mean() / ne.sum())
    return out


# --------------------------------------------------------- seed-aware parts

def arch_of(n):
    return n.split("@s")[0] if "@s" in n else n


def pairwise_distinguishable(BMs, means):
    out = {}
    idx = {n: j for j, n in enumerate(C52)}
    for s in SETS:
        BM = BMs[s]
        same, diff = [0, 0], [0, 0]
        same_list = []
        for a, b in combinations(SEEDMODELS, 2):
            d = BM[:, idx[a]] - BM[:, idx[b]]
            lo, hi = np.percentile(d, [2.5, 97.5])
            dist = bool(lo > 0 or hi < 0)
            if arch_of(a) == arch_of(b):
                same[0] += dist; same[1] += 1
                same_list.append({"pair": f"{a}|{b}", "diff": r4(means[s][a] - means[s][b]),
                                  "ci": [r4(lo), r4(hi)], "distinguishable": dist})
            else:
                diff[0] += dist; diff[1] += 1
        out[s] = {"same_arch_distinguishable": same[0], "same_arch_pairs": same[1],
                  "same_arch_share": r4(same[0] / same[1]),
                  "diff_arch_distinguishable": diff[0], "diff_arch_pairs": diff[1],
                  "diff_arch_share": r4(diff[0] / diff[1]),
                  "same_arch_detail": same_list}
        # spread of one architecture across seeds vs spread across architectures
        rng_ = [max(means[s][f"{a}@s{k}"] for k in P7.seeds_of(a)) - min(means[s][f"{a}@s{k}"] for k in P7.seeds_of(a))
                for a in ARCH if len(P7.seeds_of(a)) >= 2]
        sdb = [float(np.std(BMs[s][:, idx[n]], ddof=1)) for n in SEEDMODELS]
        out[s]["mean_seed_range_of_arch"] = r4(np.mean(rng_))
        out[s]["max_seed_range_of_arch"] = r4(np.max(rng_))
        out[s]["n_arch_with_2plus_seeds"] = len(rng_)
        out[s]["median_bootstrap_sd_of_model_mean"] = r4(np.median(sdb))
    return out


def one_training_world(means, include_nn=True, synth=None):
    """Selection losses when each architecture is trained once with seed s."""
    def pool(b, s):
        p = {a: means[b][f"{a}@s{s}"] for a in ARCH if s in P7.seeds_of(a)}
        if include_nn:
            p[NN] = means[b][NN]
        return p

    def deployed(b, s2, a):
        p = pool(b, s2)
        return None if a not in p else max(p.values()) - p[a]

    rows = []
    for s in TS:
        for s2 in TS:
            for A in WOUND:
                pa = pool(A, s)
                ch = max(pa, key=pa.get)
                for Bq in WOUND:
                    if A == Bq and s == s2:
                        continue
                    rows.append({"A": A, "B": Bq, "s": s, "s2": s2, "chosen": ch,
                                 "loss": deployed(Bq, s2, ch)})

    def agg(fl):
        sel = [r for r in rows if fl(r)]
        ok = [r["loss"] for r in sel if r["loss"] is not None]
        return {"mean": r4(np.mean(ok)) if ok else None, "n_cases": len(ok),
                "n_dropped_no_such_seed": len(sel) - len(ok),
                "min": r4(min(ok)) if ok else None, "max": r4(max(ok)) if ok else None}

    out = {
        "cross_dataset_same_seed": agg(lambda r: r["A"] != r["B"] and r["s"] == r["s2"]),
        "seed_only_same_dataset": agg(lambda r: r["A"] == r["B"] and r["s"] != r["s2"]),
        "cross_dataset_other_seed": agg(lambda r: r["A"] != r["B"] and r["s"] != r["s2"]),
        "chooser_seed0": {
            "seed_only_same_dataset": agg(lambda r: r["A"] == r["B"] and r["s"] == 0 and r["s2"] != 0),
            "cross_dataset_other_seed": agg(lambda r: r["A"] != r["B"] and r["s"] == 0 and r["s2"] != 0),
            "cross_dataset_same_seed": agg(lambda r: r["A"] != r["B"] and r["s"] == 0 and r["s2"] == 0)},
        "seed_only_by_dataset": {b: agg(lambda r, b=b: r["A"] == b and r["B"] == b and r["s"] != r["s2"])
                                 for b in WOUND},
    }
    rnd = [np.mean([max(pool(b, s2).values()) - v for v in pool(b, s2).values()]) for b in WOUND for s2 in TS]
    out["random_mean"] = r4(np.mean(rnd))
    out["random_n_cases"] = len(rnd)
    tops = {b: {s: max(pool(b, s), key=pool(b, s).get) for s in TS} for b in WOUND}
    out["top_architecture_by_seed"] = tops
    same_top = [(b, s, s2) for b in WOUND for s, s2 in combinations(TS, 2) if tops[b][s] == tops[b][s2]]
    out["same_top_architecture_across_seed_pairs"] = {"n": len(same_top), "of": len(WOUND) * 3,
                                                       "cases": [f"{b} s{s}-s{s2}" for b, s, s2 in same_top]}
    out["pool_sizes"] = {f"s{s}": len(pool(WOUND[0], s)) for s in TS}
    out["rows"] = rows
    if synth is not None:
        sy = {}
        for amp, sc in synth.items():
            cases = []
            for s in TS:
                ps = {a: sc[f"{a}@s{s}"] for a in ARCH if s in P7.seeds_of(a)}
                if include_nn:
                    ps[NN] = sc[NN]
                ch = max(ps, key=ps.get)
                for s2 in TS:
                    for b in WOUND:
                        cases.append({"s": s, "s2": s2, "B": b, "chosen": ch, "loss": deployed(b, s2, ch)})
            def ag(fl, cases=cases):
                ok = [c["loss"] for c in cases if fl(c) and c["loss"] is not None]
                return {"mean": r4(np.mean(ok)) if ok else None, "n_cases": len(ok),
                        "n_dropped_no_such_seed": sum(1 for c in cases if fl(c) and c["loss"] is None)}
            sy[f"amp{amp}"] = {"chosen_by_seed": {f"s{s}": next(c["chosen"] for c in cases if c["s"] == s) for s in TS},
                               "same_seed": ag(lambda c: c["s"] == c["s2"]),
                               "other_seed": ag(lambda c: c["s"] != c["s2"])}
        out["synthetic_chooser"] = sy
    return out


def diverged():
    log = open(os.path.join(P6.RES, "kaggle2", "log.txt")).read().splitlines()
    rep2 = json.load(open(os.path.join(P6.RES, "kaggle2", "train_report_k2.json")))
    rep1 = json.load(open(os.path.join(P6.RES, "kaggle", "train_report_pretrained.json")))
    failed = []
    for i, line in enumerate(log):
        m = re.search(r"\]\s+(\S+): FAILED$", line)
        if m:
            name = m.group(1)
            nf = [re.search(r"epoch (\d+)/40 .*?(\d+) non-finite", x) for x in log[max(0, i - 8):i] if name in x]
            nf = [(int(x.group(1)), int(x.group(2))) for x in nf if x]
            err = next((x.strip() for x in log[i:i + 12] if "non-finite weights" in x), None)
            failed.append({"model": name, "nonfinite_losses_by_epoch": nf, "error": err})
    planned = [(a, k) for a in ARCH for k in TS]
    trained = [(a, k) for a in ARCH for k in P7.seeds_of(a)]
    missing = sorted(set(planned) - set(trained), key=lambda x: (ARCH.index(x[0]), x[1]))
    names = {f"{a}_in_s{k}" if a in P7.ARCH_OLD else f"{a}_s{k}" for a, k in missing}
    assert names == {f["model"] for f in failed}, (names, failed)
    return {"planned": len(planned), "trained_and_scored": len(trained),
            "diverged": [f"{a} seed {k}" for a, k in missing], "log": failed,
            "train_report_phase2_entries": len(rep2),
            "train_report_phase2_nonfinite_steps_nonzero": [k for k, v in rep2.items() if v.get("nonfinite_loss_steps")],
            "run1_entries": len(rep1),
            "run1_nonfinite_recorded": False,
            "surviving_manet": {a: P7.seeds_of(a) for a in P7.MANET}}


# ------------------------------------------------------------------ main

def cmp(a, b, path=""):
    """Max abs difference of two nested structures (exact for strings/bools)."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        return max([cmp(a[k], b[k], path + "/" + k) for k in a] + [0.0])
    if isinstance(a, list):
        assert len(a) == len(b), path
        return max([cmp(x, y, path) for x, y in zip(a, b)] + [0.0])
    if isinstance(a, bool) or isinstance(a, str) or a is None:
        assert a == b, (path, a, b)
        return 0.0
    return abs(float(a) - float(b))


def main():
    V6 = json.load(open(V6JSON))
    res = {"definitions": __doc__, "seed": SEED, "B": B}

    # M11 rerun: must reproduce practical_v6.json (sections 1 to 4)
    r11, i11 = analyse(M11, True, ("U", "F", "K", "M", "S"))
    chk = {}
    chk["per_set"] = cmp(r11["per_set"], V6["per_set"])
    sel = {k: v for k, v in r11["selection"].items() if k != "n_offdiag_pairs"}
    sel["synthetic"] = {a: {k: v for k, v in sel["synthetic"][a].items()
                            if k not in ("chosen_synthetic_dice", "runner_up_synthetic_dice")}
                        for a in sel["synthetic"]}
    chk["selection"] = cmp(sel, V6["selection"])
    for t in THRESH:
        m = r11["miss"][str(t)]
        chk[f"miss_{t}"] = max(cmp(m["table"], V6["miss"][str(t)]["table"]),
                               cmp(m["group_mean_miss_rate"], V6["miss"][str(t)]["group_mean_miss_rate"]),
                               cmp(m["spearman_miss_rate"], V6["miss"][str(t)]["spearman_miss_rate"]),
                               cmp(m["spearman_dice_when_detected"], V6["miss"][str(t)]["spearman_dice_when_detected"]))
    chk["spearman_mean_dice"] = cmp(r11["spearman_mean_dice"], V6["spearman_mean_dice"])
    chk["spearman_mean_dice_nonempty"] = cmp(r11["spearman_mean_dice_nonempty"], V6["spearman_mean_dice_nonempty"])
    assert all(v == 0.0 for v in chk.values()), chk
    print("M11 reproduces practical_v6 exactly", flush=True)

    # independent route for the seed-averaged scores and the synthetic chooser
    route = 0.0
    for s in SETS:
        D7 = P7.load7(s, "dice")
        route = max(route, float(np.abs(score_matrix(s, M19) - D7["X"][:, P7.cols(M19)]).max()))
    bj = json.load(open(os.path.join(OUT, "bench_v7.json")))
    # synthetic chooser score, second route: per training seed the mean over the
    # 3 synthetic sets, then the mean over training seeds (order swapped)
    syn2 = {}
    for amp in (11, 40):
        for a in ARCH:
            per_ts = [np.mean([BV.synth_seed(f"genB_amp{amp}_s{ss}", a, k).mean() for ss in (2024, 7, 101)])
                      for k in P7.seeds_of(a)]
            syn2[(amp, f"{a}@avg")] = float(np.mean(per_ts))
    res["checks"] = {"m11_reproduces_v6_max_abs": chk, "m19_scores_vs_load7_max_abs": route}

    r19, i19 = analyse(M19, True, ("U", "F", "K", "M19"))
    d = max(abs(syn2[(amp, n)] - i19["synth_raw"][amp][n]) for (amp, n) in syn2)
    res["checks"]["synthetic_chooser_second_route_max_abs"] = d
    assert d < 1e-12, d
    r17, _ = analyse(M17, False)
    r52, i52 = analyse(C52, False)
    res["M19"] = r19
    res["M17_summary"] = {
        "n_indistinguishable_from_top": {s: r17["per_set"][s]["n_indistinguishable_from_top"] for s in SETS},
        "top": {s: r17["per_set"][s]["top"] for s in SETS},
        "offdiag_mean": r17["selection"]["offdiag_mean"], "random_mean_over_B": r17["selection"]["random_mean_over_B"],
        "synthetic": {a: {k: r17["selection"]["synthetic"][a][k] for k in ("chosen", "loss", "mean_loss")}
                      for a in ("amp40", "amp11")}}

    res["miss_intervals"] = {"M11": {str(t): miss_intervals(M11, t) for t in THRESH},
                             "M19": {str(t): miss_intervals(M19, t) for t in THRESH}}
    res["miss_seed_averaged_dice_sensitivity"] = {str(t): avg_dice_miss(M19, t) for t in THRESH}
    for s in SETS:   # point estimates of the interval routine = the table route
        for t in THRESH:
            assert res["miss_intervals"]["M19"][str(t)][s]["pretrained_mean_miss"] == r19["miss"][str(t)]["table"][s]["_pretrained18_mean_miss"]
            assert res["miss_intervals"]["M11"][str(t)][s]["pretrained_mean_miss"] == V6["miss"][str(t)]["table"][s]["_pretrained10_mean_miss"]
            assert res["miss_intervals"]["M11"][str(t)][s]["nnunet_miss"] == V6["miss"][str(t)]["table"][s]["_nnunet_miss"]

    # seed-aware
    sa = {"C52": {k: r52[k] for k in ("per_set", "selection")}}
    for s in SETS:
        ps = r52["per_set"][s]
        top = ps["top"]
        ind = [n for n, v in ps["models"].items() if v.get("indistinguishable")]
        ps["indistinguishable_models"] = ind
        ps["indistinguishable_same_arch_as_top"] = [n for n in ind if arch_of(n) == arch_of(top)]
        ps["n_distinct_arch_among_top_and_indistinguishable"] = len({arch_of(n) for n in ind + [top]})
        sib = [n for n in SEEDMODELS if arch_of(n) == arch_of(top) and n != top]
        ps["siblings_of_top"] = {n: {"mean": ps["models"][n]["mean"], "diff_ci": ps["models"][n]["diff_ci"],
                                     "indistinguishable": ps["models"][n]["indistinguishable"]} for n in sib}
    sa["siblings_summary"] = {
        "n_sibling_models": sum(len(c52_ps["siblings_of_top"]) for c52_ps in r52["per_set"].values()),
        "n_indistinguishable": sum(v["indistinguishable"] for c52_ps in r52["per_set"].values()
                                   for v in c52_ps["siblings_of_top"].values()),
        "top_training_seed": {s: r52["per_set"][s]["top"].split("@s")[-1] if "@s" in r52["per_set"][s]["top"] else "nnU-Net"
                              for s in SETS}}
    full3 = [a for a in ARCH if P7.seeds_of(a) == list(TS)]
    sa["mean_over_full_seed_archs_by_seed"] = {
        "n_arch": len(full3),
        "by_set": {s: {f"s{k}": r4(np.mean([i52["means"][s][f"{a}@s{k}"] for a in full3])) for k in TS} for s in SETS}}
    sa["pairwise"] = pairwise_distinguishable(i52["BM"], i52["means"])
    sa["one_training_world"] = one_training_world(i52["means"], True, i52["synth_raw"])
    sa["one_training_world_no_nnunet"] = one_training_world(i52["means"], False, i52["synth_raw"])
    # spread of one architecture's synthetic score across training seeds
    ssr = {}
    for amp in (40, 11):
        rr = {a: max(i52["synth_raw"][amp][f"{a}@s{k}"] for k in P7.seeds_of(a))
              - min(i52["synth_raw"][amp][f"{a}@s{k}"] for k in P7.seeds_of(a))
              for a in ARCH if len(P7.seeds_of(a)) >= 2}
        ssr[f"amp{amp}"] = {"by_arch": {a: r4(v) for a, v in rr.items()}, "mean": r4(np.mean(list(rr.values()))),
                            "max": r4(max(rr.values())), "max_arch": max(rr, key=rr.get), "n_arch": len(rr),
                            "per_seed_scores_of_max_arch": {f"s{k}": r4(i52["synth_raw"][amp][f"{max(rr, key=rr.get)}@s{k}"])
                                                            for k in P7.seeds_of(max(rr, key=rr.get))},
                            "nnunet": r4(i52["synth_raw"][amp][NN])}
    sa["synthetic_seed_range"] = ssr
    # single-training candidate sets (one model per architecture, seed s) + nnU-Net
    per_seed = {}
    for k in TS:
        cand = [f"{a}@s{k}" for a in ARCH if k in P7.seeds_of(a)] + [NN]
        rk, _ = analyse(cand, False)
        per_seed[f"s{k}"] = {
            "n_candidates": len(cand),
            "top": {s: rk["per_set"][s]["top"] for s in SETS},
            "n_indistinguishable_from_top": {s: rk["per_set"][s]["n_indistinguishable_from_top"] for s in SETS},
            "offdiag_mean": rk["selection"]["offdiag_mean"],
            "random_mean_over_B": rk["selection"]["random_mean_over_B"],
            "synthetic": {a: {kk: rk["selection"]["synthetic"][a][kk] for kk in ("chosen", "mean_loss", "loss")}
                          for a in ("amp40", "amp11")}}
    sa["per_seed_candidate_sets"] = per_seed
    # second route: synthetic chooser, same seed, = mean of the per-seed sets' synthetic losses
    for amp in ("amp40", "amp11"):
        v1 = sa["one_training_world"]["synthetic_chooser"][amp]["same_seed"]["mean"]
        v2 = r4(np.mean([per_seed[f"s{k}"]["synthetic"][amp]["mean_loss"] for k in TS]))
        assert abs(v1 - v2) <= 1e-4, (amp, v1, v2)
    # second route: cross-dataset same-seed loss = mean of the per-seed sets' offdiag means
    v1 = sa["one_training_world"]["cross_dataset_same_seed"]["mean"]
    v2 = r4(np.mean([per_seed[f"s{k}"]["offdiag_mean"] for k in TS]))
    assert abs(v1 - v2) <= 1e-4, (v1, v2)
    res["checks"]["one_training_world_vs_per_seed_sets"] = "agree within 1e-4 (rounding)"
    for k in ("one_training_world", "one_training_world_no_nnunet"):
        rows = sa[k].pop("rows")
        sa[k]["n_rows"] = len(rows)
        so = [r for r in rows if r["A"] == r["B"] and r["s"] != r["s2"]]
        sa[k]["seed_only_nnunet_chosen"] = {"n": sum(r["chosen"] == NN for r in so), "of": len(so),
                                            "cases": sorted({f"{r['A']} s{r['s']}" for r in so if r["chosen"] == NN})}
        # independent recomputation of one aggregate from the rows
        cs = [r["loss"] for r in rows if r["A"] != r["B"] and r["s"] == r["s2"] and r["loss"] is not None]
        assert r4(np.mean(cs)) == sa[k]["cross_dataset_same_seed"]["mean"]
    # one_training_world, seed s = s': cross-dataset loss within seed-0 pool must
    # equal the analyse() offdiag loss on the seed-0 candidate set (second route)
    r_s0, _ = analyse([f"{a}@s0" for a in ARCH if 0 in P7.seeds_of(a)] + [NN], False)
    w = one_training_world(i52["means"], True)["rows"]
    l0 = [r["loss"] for r in w if r["A"] != r["B"] and r["s"] == 0 and r["s2"] == 0]
    assert r4(np.mean(l0)) == r_s0["selection"]["offdiag_mean"], (np.mean(l0), r_s0["selection"]["offdiag_mean"])
    sa["seed0_candidates_offdiag_mean"] = r_s0["selection"]["offdiag_mean"]
    sa["seed0_candidates_random_mean"] = r_s0["selection"]["random_mean_over_B"]
    res["seed_aware"] = sa
    res["diverged_trainings"] = diverged()

    # headline V6 vs V7
    hl = {"selection_offdiag_mean": {"v6": V6["selection"]["offdiag_mean"], "v7": r19["selection"]["offdiag_mean"]},
          "random_mean": {"v6": V6["selection"]["random_mean_over_B"], "v7": r19["selection"]["random_mean_over_B"]}}
    for a in ("amp40", "amp11"):
        hl[f"synthetic_{a}"] = {"v6": {k: V6["selection"]["synthetic"][a][k] for k in ("chosen", "mean_loss", "loss")},
                                "v7": {k: r19["selection"]["synthetic"][a][k] for k in ("chosen", "mean_loss", "loss")}}
    hl["n_indistinguishable_from_top"] = {
        s: {"v6": V6["per_set"][s]["n_indistinguishable_from_top"], "v6_of": 10,
            "v7": r19["per_set"][s]["n_indistinguishable_from_top"], "v7_of": 18,
            "top_v6": V6["per_set"][s]["top"], "top_v7": r19["per_set"][s]["top"]} for s in SETS}
    hl["azh_miss_t0.1"] = {"v6": res["miss_intervals"]["M11"]["0.1"]["azh"],
                           "v7": res["miss_intervals"]["M19"]["0.1"]["azh"]}
    hl["spearman_mean_dice_6pairs"] = {"v6": V6["spearman_mean_dice"]["mean"], "v7": r19["spearman_mean_dice"]["mean"]}
    BJ = bj["headline_amp40_mean"]
    hl["bench_rel_amp40_mean"] = BJ
    res["headline"] = hl

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "practical_v7.json"), "w"), indent=1)
    write_md(res)
    print("written", os.path.join(OUT, "practical_v7.json"))


NICE = {"medetec": "Medetec", "co2wounds": "CO2Wounds", "fuseg": "FUSeg", "azh": "AZH",
        "azh_train_slice": "control azh_train_slice"}


def f2(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci2(c):
    return f"[{f2(c[0])}, {f2(c[1])}]"


def write_md(res):
    r = res["M19"]
    hl = res["headline"]
    sa = res["seed_aware"]
    L = ["# Practical consequences for choosing a model (V7, modern group M19)", "",
         "Script: `scripts/v7/practical_v7.py`. Full numbers: `results/v7/practical_v7.json`. "
         "Benchmark numbers quoted in the headline come from `results/v7/bench_v7.json` (`scripts/v7/bench_v7.py`).", "",
         "## Definitions", "",
         "- Candidates. M19: 18 ImageNet-pretrained architectures trained on FUSeg with one shared recipe, plus nnU-Net (one training). "
         "An architecture's per-image Dice is the mean over its available training seeds (3 for 16 architectures, 2 for manet_r18, 1 for manet_r50). "
         "M11: the V6 set (10 run-1 seed-0 architectures + nnU-Net). C52: every trained architecture-seed model as its own candidate (51) plus nnU-Net.",
         "- Mean Dice over all kept images, empty references included (Dice 1 if the prediction is also empty, else 0).",
         "- Bootstrap: 2,000 draws over resampling units (photograph for Medetec and AZH, image otherwise), paired across candidates, `default_rng([20261010, dataset index])` as in V6. M11, M19 and C52 use the same draws. 95% percentile intervals.",
         "- Indistinguishable from the top: the 95% paired interval of (top minus candidate) has lower bound 0 or less.",
         "- Selection loss A to B: best mean Dice on B minus the mean Dice on B of the candidate ranked first on A (12 ordered wound pairs). Random choice: mean over candidates of (best on B minus candidate on B).",
         "- Synthetic chooser: generator B, amplitude 40 (and 11), mean Dice over 45 images per synthetic seed, averaged over seeds 2024, 7, 101; architectures first averaged over training seeds per image.",
         "- Miss: non-empty reference and Dice < t (t = 0.1; 0.05 and 0.2 as sensitivity). For an architecture, the miss rate and Dice when detected are computed per training seed and averaged. "
         "'Pretrained mean' averages the pretrained architectures. Intervals: same paired unit bootstrap.",
         "- Spearman: scipy, average ranks, across the candidates, between two wound datasets; mean over the 6 pairs.", ""]

    L += ["## Headline: V6 next to V7", "",
          "| quantity | V6 (M11) | V7 (M19) |", "|---|---|---|",
          f"| Selection loss, choose on one wound set and deploy on another (mean of 12 pairs) | {f2(hl['selection_offdiag_mean']['v6'])} | {f2(hl['selection_offdiag_mean']['v7'])} |",
          f"| Random-choice loss (mean of 4 wound sets) | {f2(hl['random_mean']['v6'])} (11 candidates) | {f2(hl['random_mean']['v7'])} (19 candidates) |"]
    for a in ("amp40", "amp11"):
        h = hl[f"synthetic_{a}"]
        L.append(f"| Synthetic chooser {a}: pick, mean loss over 4 wound sets | {h['v6']['chosen']}, {f2(h['v6']['mean_loss'])} | {h['v7']['chosen']}, {f2(h['v7']['mean_loss'])} |")
    for s in SETS:
        h = hl["n_indistinguishable_from_top"][s]
        L.append(f"| {NICE[s]}: top; others indistinguishable from it | {h['top_v6']}; {h['v6']} of {h['v6_of']} | {h['top_v7']}; {h['v7']} of {h['v7_of']} |")
    a6, a7 = hl["azh_miss_t0.1"]["v6"], hl["azh_miss_t0.1"]["v7"]
    L += [f"| AZH miss rate t = 0.1, pretrained mean (n non-empty images {a7['n_nonempty']}, {a7['n_units']} units) | {f2(a6['pretrained_mean_miss'])} {ci2(a6['pretrained_ci'])} ({a6['n_pretrained']} archs) | {f2(a7['pretrained_mean_miss'])} {ci2(a7['pretrained_ci'])} ({a7['n_pretrained']} archs) |",
          f"| AZH miss rate t = 0.1, nnU-Net | {f2(a6['nnunet_miss'])} {ci2(a6['nnunet_ci'])} | {f2(a7['nnunet_miss'])} {ci2(a7['nnunet_ci'])} |",
          f"| AZH miss rate difference, pretrained mean minus nnU-Net | {f2(a6['diff_pretrained_minus_nnunet'])} {ci2(a6['diff_ci'])} | {f2(a7['diff_pretrained_minus_nnunet'])} {ci2(a7['diff_ci'])} |",
          f"| Spearman of mean Dice across modern models, mean of 6 wound pairs | {f2(hl['spearman_mean_dice_6pairs']['v6'])} (11 models) | {f2(hl['spearman_mean_dice_6pairs']['v7'])} (19 models) |"]
    bj = hl["bench_rel_amp40_mean"]
    for k, lab in (("U -> U", "U (18 methods, unchanged)"), ("M -> M19", "M (11) -> M19 (19)"), ("U+M -> U+M19", "U+M (29) -> U+M19 (37)")):
        v = bj[k]["diff"]
        L.append(f"| Benchmark relative agreement, amp 40, diff = bench - other, {lab}, method bootstrap | "
                 f"{v['v6']['est']:+.3f} [{v['v6']['lo']:+.3f}, {v['v6']['hi']:+.3f}] | {v['v7']['est']:+.3f} [{v['v7']['lo']:+.3f}, {v['v7']['hi']:+.3f}] |")
    L += ["", "V6 had no intervals for miss rates; the V6 intervals above were computed here with the same draws and reproduce the V6 point estimates exactly.", ""]

    L += ["## 1 and 2. Spread within M19", "",
          "| set | images (units) | top | top mean [95% CI] | indistinguishable (of 18) | top wins draws | best minus median | best minus worst (worst) |",
          "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        p = r["per_set"][s]
        t = p["models"][p["top"]]
        L.append(f"| {NICE[s]} | {p['n_images']} ({p['n_units']}) | {p['top']} | {f2(t['mean'])} {ci2(t['ci'])} | {p['n_indistinguishable_from_top']} | "
                 f"{f2(p['share_draws_top_is_top'])} | {f2(p['best_minus_median'])} | {f2(p['best_minus_worst'])} ({p['worst']}) |")
    L += ["", "Candidates indistinguishable from the top:", ""]
    for s in SETS:
        p = r["per_set"][s]
        ind = [n for n, v in p["models"].items() if v.get("indistinguishable")]
        L.append(f"- {NICE[s]}: {', '.join(ind) or 'none'}.")
    az = r["per_set"]["azh"]["models"]
    L += ["", f"On AZH, segformer_mitb2 ({f2(az['segformer_mitb2@avg']['mean'])}) is ahead of nnU-Net ({f2(az['nnunet_2d']['mean'])}); "
          f"the paired interval of the gap is {ci2(az['nnunet_2d']['diff_ci'])}. In V6 nnU-Net was top on AZH with 8 of 10 models indistinguishable from it.", ""]
    m17 = res["M17_summary"]
    L += ["Sensitivity, M17 (no MAnet, 17 candidates): indistinguishable from the top "
          + ", ".join(f"{NICE[s]} {m17['n_indistinguishable_from_top'][s]} of 16 ({m17['top'][s]})" for s in SETS) + ".", ""]

    sel = r["selection"]
    L += ["## 3. Selection loss (rows: chosen on; columns: deployed on)", "",
          "| chosen on (pick) | " + " | ".join(NICE[b] for b in WOUND) + " |", "|---" * 5 + "|"]
    for a in WOUND:
        L.append(f"| {NICE[a]} ({sel['chosen_on'][a]}) | " + " | ".join(f2(sel['matrix_rows_choose_on_cols_deploy_on'][a][b]) for b in WOUND) + " |")
    L += ["", f"- Off-diagonal ({sel['n_offdiag_pairs']} pairs): mean {f2(sel['offdiag_mean'])}, range {f2(sel['offdiag_min'])} to {f2(sel['offdiag_max'])}. V6: {f2(hl['selection_offdiag_mean']['v6'])}.",
          "- Random choice, expected loss: " + ", ".join(f"{NICE[b]} {f2(v)}" for b, v in sel["random_choice_expected_loss"].items())
          + f" (mean {f2(sel['random_mean_over_B'])}).",
          "- Control chooser (azh_train_slice, pick " + sel["chosen_on"][CTRL] + "): " + ", ".join(f"{NICE[b]} {f2(v)}" for b, v in sel["control_chooser_loss"].items()) + "."]
    for a in ("amp40", "amp11"):
        sy = sel["synthetic"][a]
        L.append(f"- Synthetic chooser {a}: picks {sy['chosen']} (synthetic Dice {f2(sy['chosen_synthetic_dice'])}, runner-up {f2(sy['runner_up_synthetic_dice'])}); losses "
                 + ", ".join(f"{NICE[b]} {f2(v)}" for b, v in sy["loss"].items()) + f" (mean {f2(sy['mean_loss'])}). Spearman of synthetic with wound means: "
                 + ", ".join(f"{NICE[b]} {v:+.2f}" for b, v in sy["spearman_with_wound"].items()) + ".")
    L.append(f"- Minimax: {sel['minimax_model']} has the smallest maximum gap from the best ({f2(sel['minimax_gap'])}).")
    L.append(f"- M17 sensitivity: off-diagonal mean {f2(m17['offdiag_mean'])}, random {f2(m17['random_mean_over_B'])}, synthetic pick (amp 40) {m17['synthetic']['amp40']['chosen']} with mean loss {f2(m17['synthetic']['amp40']['mean_loss'])}.")
    L += [f"- The FUSeg chooser picks manet_r50, which rests on a single training (its seeds 0 and 2 diverged).", ""]

    L += ["## 4. Misses", "", "t = 0.1. Pretrained mean with 95% interval and the range over the 18 architectures; nnU-Net with interval; group means for U (18), F (14), K (14).", "",
          "| set | n non-empty | pretrained-18 mean miss [95%] (range) | nnU-Net miss [95%] | difference [95%] | U | F | K |", "|---|---|---|---|---|---|---|---|"]
    mi = res["miss_intervals"]["M19"]["0.1"]
    tab = r["miss"]["0.1"]["table"]
    gm = r["miss"]["0.1"]["group_mean_miss_rate"]
    for s in SETS:
        x = mi[s]
        rates = [tab[s][n]["miss_rate"] for n in M19 if n != NN]
        L.append(f"| {NICE[s]} | {x['n_nonempty']} | {f2(x['pretrained_mean_miss'])} {ci2(x['pretrained_ci'])} ({f2(min(rates))} to {f2(max(rates))}) | "
                 f"{f2(x['nnunet_miss'])} {ci2(x['nnunet_ci'])} | {f2(x['diff_pretrained_minus_nnunet'])} {ci2(x['diff_ci'])} | {f2(gm['U'][s])} | {f2(gm['F'][s])} | {f2(gm['K'][s])} |")
    L += ["", "AZH by threshold (pretrained mean vs nnU-Net, M19):", "", "| t | pretrained mean [95%] | nnU-Net [95%] | difference [95%] | seed-averaged-Dice miss (sensitivity) |", "|---|---|---|---|---|"]
    for t in THRESH:
        x = res["miss_intervals"]["M19"][str(t)]["azh"]
        L.append(f"| {t} | {f2(x['pretrained_mean_miss'])} {ci2(x['pretrained_ci'])} | {f2(x['nnunet_miss'])} {ci2(x['nnunet_ci'])} | {f2(x['diff_pretrained_minus_nnunet'])} {ci2(x['diff_ci'])} | {f2(res['miss_seed_averaged_dice_sensitivity'][str(t)]['azh'])} |")
    L += ["", f"Best pretrained architecture on AZH by miss rate: segformer_mitb2, {f2(tab['azh']['segformer_mitb2@avg']['miss_rate'])} at t = 0.1 (nnU-Net {f2(tab['azh']['nnunet_2d']['miss_rate'])}).", "",
          "Mean Spearman over the 6 wound pairs, 19 models:", "", "| quantity | t = 0.05 | t = 0.1 | t = 0.2 |", "|---|---|---|---|"]
    for q, lab in (("spearman_miss_rate", "miss rate"), ("spearman_dice_when_detected", "Dice when detected")):
        L.append(f"| {lab} | " + " | ".join(f"{r['miss'][str(t)][q]['mean']:+.2f}" for t in THRESH) + " |")
    L += ["", f"Mean Dice: {r['spearman_mean_dice']['mean']:+.2f} (pairs: " + ", ".join(f"{k} {v:+.2f}" for k, v in r["spearman_mean_dice"]["pairs"].items())
          + f"); non-empty images only: {r['spearman_mean_dice_nonempty']['mean']:+.2f}.", ""]

    c52 = sa["C52"]
    L += ["## 5. Seed-aware analysis", "",
          "### 5a. Every trained model as its own candidate (C52: 51 architecture-seed models + nnU-Net)", "",
          "| set | top | indistinguishable (of 51) | of these, same architecture as top | distinct architectures among top + indistinguishable | top wins draws |", "|---|---|---|---|---|---|"]
    for s in SETS:
        p = c52["per_set"][s]
        L.append(f"| {NICE[s]} | {p['top']} | {p['n_indistinguishable_from_top']} | {len(p['indistinguishable_same_arch_as_top'])} | {p['n_distinct_arch_among_top_and_indistinguishable']} | {f2(p['share_draws_top_is_top'])} |")
    ss = sa["siblings_summary"]
    L += ["", f"Every C52 top is a single trained model. Its other training seeds (same architecture) are never indistinguishable from it: "
          f"{ss['n_indistinguishable']} of {ss['n_sibling_models']} sibling models across the 5 sets. Training seed of the top model per set: "
          + ", ".join(f"{NICE[s]} {v}" for s, v in ss["top_training_seed"].items())
          + ". Seed 1 is not better on average (mean Dice of the " + str(sa["mean_over_full_seed_archs_by_seed"]["n_arch"])
          + " architectures with all 3 seeds, seeds 0/1/2: "
          + "; ".join(f"{NICE[s]} " + "/".join(f2(x) for x in v.values()) for s, v in sa["mean_over_full_seed_archs_by_seed"]["by_set"].items()) + ")."]
    cs = c52["selection"]
    L += ["", f"Selection loss over C52: off-diagonal mean {f2(cs['offdiag_mean'])} (12 pairs), random {f2(cs['random_mean_over_B'])}. "
          f"Synthetic chooser (amp 40) picks {cs['synthetic']['amp40']['chosen']}, mean loss {f2(cs['synthetic']['amp40']['mean_loss'])}; amp 11 picks {cs['synthetic']['amp11']['chosen']}, {f2(cs['synthetic']['amp11']['mean_loss'])}.", "",
          "### 5b. One training per architecture: the result depends on which seed was trained", "",
          "Each candidate set holds one model per architecture (all trained with seed s) plus nnU-Net.", "",
          "| seed set (n) | indistinguishable from top: " + ", ".join(NICE[s] for s in SETS) + " | selection loss | random | synthetic pick (amp 40), loss |", "|---|---|---|---|---|"]
    for k, v in sa["per_seed_candidate_sets"].items():
        L.append(f"| {k} ({v['n_candidates']}) | " + ", ".join(str(v["n_indistinguishable_from_top"][s]) for s in SETS)
                 + f" | {f2(v['offdiag_mean'])} | {f2(v['random_mean_over_B'])} | {v['synthetic']['amp40']['chosen']}, {f2(v['synthetic']['amp40']['mean_loss'])} |")
    L += ["", "Top per seed set: " + "; ".join(f"{k}: " + ", ".join(f"{NICE[s]} {v['top'][s]}" for s in WOUND) for k, v in sa["per_seed_candidate_sets"].items()) + ".", "",
          "### 5c. Are two seeds of one architecture more alike than two architectures?", "",
          "Share of model pairs whose paired 95% interval of the mean-Dice difference excludes 0 (51 trained models; nnU-Net excluded).", "",
          "| set | same architecture, distinguishable | different architectures, distinguishable | mean (max) seed range of an architecture | median bootstrap SD of one model's mean |", "|---|---|---|---|---|"]
    for s in SETS:
        x = sa["pairwise"][s]
        L.append(f"| {NICE[s]} | {x['same_arch_distinguishable']} of {x['same_arch_pairs']} ({f2(x['same_arch_share'])}) | {x['diff_arch_distinguishable']} of {x['diff_arch_pairs']} ({f2(x['diff_arch_share'])}) | "
                 f"{f2(x['mean_seed_range_of_arch'])} ({f2(x['max_seed_range_of_arch'])}), {x['n_arch_with_2plus_seeds']} archs | {f2(x['median_bootstrap_sd_of_model_mean'])} |")
    L += ["", "### 5d. How much of the selection loss is seed-to-seed variation?", "",
          "Chooser sees the seed-s models on A and picks an architecture; the deployed model is that architecture trained with seed s' and evaluated on B. "
          "Loss is measured against the best model of the seed-s' set on B. nnU-Net has one model, so it is the same model in every seed set.", "",
          "| scenario | with nnU-Net: mean loss (cases, dropped) | without nnU-Net |", "|---|---|---|"]
    W1, W0 = sa["one_training_world"], sa["one_training_world_no_nnunet"]

    def cell(x):
        return f"{f2(x['mean'])} ({x['n_cases']}, {x['n_dropped_no_such_seed']})"
    for key, lab in (("seed_only_same_dataset", "Seed only: A = B, s' differs from s"),
                     ("cross_dataset_same_seed", "Dataset only: A differs from B, s' = s"),
                     ("cross_dataset_other_seed", "Both: A differs from B, s' differs from s")):
        L.append(f"| {lab} | {cell(W1[key])} | {cell(W0[key])} |")
    for key, lab in (("seed_only_same_dataset", "Chooser on seed 0, seed only (s' = 1, 2)"),
                     ("cross_dataset_other_seed", "Chooser on seed 0, both"),
                     ("cross_dataset_same_seed", "Chooser on seed 0, dataset only")):
        L.append(f"| {lab} | {cell(W1['chooser_seed0'][key])} | {cell(W0['chooser_seed0'][key])} |")
    L.append(f"| Random choice (4 sets x 3 seeds) | {f2(W1['random_mean'])} | {f2(W0['random_mean'])} |")
    for amp in ("amp40", "amp11"):
        sy1, sy0 = W1["synthetic_chooser"][amp], W0["synthetic_chooser"][amp]
        L.append(f"| Synthetic chooser {amp}, same seed (picks by seed: {', '.join(sy1['chosen_by_seed'].values())}) | {cell(sy1['same_seed'])} | {cell(sy0['same_seed'])} |")
        L.append(f"| Synthetic chooser {amp}, other seed (without nnU-Net picks: {', '.join(sy0['chosen_by_seed'].values())}) | {cell(sy1['other_seed'])} | {cell(sy0['other_seed'])} |")
    so = W1["seed_only_by_dataset"]
    L += ["", "Dropped cases: the chosen architecture has no model for seed s' (MAnet). Seed-only loss by dataset: "
          + ", ".join(f"{NICE[b]} {f2(v['mean'])} ({v['n_cases']} cases)" for b, v in so.items()) + ".",
          "Top architecture per seed (with nnU-Net): " + "; ".join(f"{NICE[b]} " + "/".join(v[str(k)] if str(k) in v else v[k] for k in TS) for b, v in W1["top_architecture_by_seed"].items())
          + f". The same architecture tops both seeds in {W1['same_top_architecture_across_seed_pairs']['n']} of {W1['same_top_architecture_across_seed_pairs']['of']} dataset x seed-pair cases.", ""]
    ssr = sa["synthetic_seed_range"]
    L += ["### 5e. Synthetic scores move a lot between training seeds", "",
          f"- Generator B amplitude 40: range of an architecture's synthetic mean Dice across its training seeds, mean {f2(ssr['amp40']['mean'])}, max {f2(ssr['amp40']['max'])} "
          f"({ssr['amp40']['max_arch']}: " + ", ".join(f"{k} {f2(v)}" for k, v in ssr['amp40']['per_seed_scores_of_max_arch'].items()) + f"), {ssr['amp40']['n_arch']} architectures with 2 or more seeds. nnU-Net scores {f2(ssr['amp40']['nnunet'])}.",
          f"- Amplitude 11: mean {f2(ssr['amp11']['mean'])}, max {f2(ssr['amp11']['max'])} ({ssr['amp11']['max_arch']}).",
          "- For comparison, the mean seed range on the clinical wound sets is " + ", ".join(f"{NICE[s]} {f2(sa['pairwise'][s]['mean_seed_range_of_arch'])}" for s in WOUND) + ".", ""]

    dv = res["diverged_trainings"]
    L += ["## 6. Diverged trainings (practical finding)", "",
          f"Under the shared training recipe, {len(dv['diverged'])} of {dv['planned']} planned trainings (18 architectures x 3 seeds) diverged: "
          + ", ".join(dv["diverged"]) + f". {dv['trained_and_scored']} trainings were saved and scored. All three are MAnet; the log shows non-finite losses building up during training and non-finite weights at the end, so the checkpoints were not saved:", ""]
    for f in dv["log"]:
        L.append(f"- {f['model']}: cumulative non-finite loss steps " + ", ".join(f"by epoch {e}: {n}" for e, n in f["nonfinite_losses_by_epoch"]) + ".")
    L += [f"- The {dv['train_report_phase2_entries']} saved phase-2 trainings recorded no non-finite loss steps. The 10 run-1 trainings did not record this count.",
          f"- Surviving MAnet seeds: manet_r18 {dv['surviving_manet']['manet_r18']}, manet_r50 {dv['surviving_manet']['manet_r50']}. "
          "MAnet is the only architecture family that failed, and it failed at both encoder sizes; a practitioner reusing one recipe across architectures should expect such failures and check for them.", ""]

    ck = res["checks"]
    L += ["## Checks", "",
          "- The copied V6 logic run on M11 reproduces `results/v6/practical_v6.json` exactly (per-set tables, selection, synthetic chooser, misses at all 3 thresholds, Spearman): max abs difference "
          + ", ".join(f"{k} {v:.0e}" for k, v in ck["m11_reproduces_v6_max_abs"].items()) + ".",
          f"- M19 per-image scores built from per-seed members equal `perimage_v7.load7` columns: max abs difference {ck['m19_scores_vs_load7_max_abs']:.0e}.",
          f"- Synthetic chooser scores recomputed with the averaging order swapped (training seeds outside, synthetic seeds inside): max abs difference {ck['synthetic_chooser_second_route_max_abs']:.0e}.",
          "- Miss-rate interval routine point estimates equal the table route (M19) and the V6 values (M11) at all thresholds and sets (asserted).",
          "- One-training-world losses (dataset only, seed s' = s) equal the mean of the per-seed candidate sets' selection losses, and the synthetic same-seed loss equals the mean of their synthetic losses (asserted, within rounding).",
          "- The three diverged trainings found in the log are exactly the architecture-seed models missing from the score files (asserted).", "",
          "## Limitations", "",
          "- nnU-Net has a single training, so it carries no seed averaging and contributes no seed-to-seed variation. In the seed-only scenario it is the chosen model in "
          + f"{sa['one_training_world']['seed_only_nnunet_chosen']['n']} of {sa['one_training_world']['seed_only_nnunet_chosen']['of']} cases ("
          + (", ".join(sa['one_training_world']['seed_only_nnunet_chosen']['cases']) or "none") + "); the 'without nnU-Net' column removes this.",
          "- manet_r50 rests on one training and manet_r18 on two; the FUSeg chooser picks manet_r50, so the largest FUSeg-to-other losses depend on a single training. M17 removes both.",
          "- Seed averaging (M19) makes each architecture's score less noisy than any single trained model. A practitioner deploys one trained model, so section 5 is the closer description of practice.",
          "- Three seeds per architecture give a rough view of training noise; seed ranges and seed-only losses are point estimates without intervals.",
          "- Each wound set has 112 to 152 images, Medetec only 38 units, so intervals are wide. Pairwise distinguishability counts 1,275 overlapping comparisons and is descriptive, not a multiplicity-corrected test.",
          "- The AZH explanation of V6 (padding and scale) was not rerun; it rested on the 10 run-1 models.", ""]
    open(os.path.join(OUT, "PRACTICAL_V7.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()

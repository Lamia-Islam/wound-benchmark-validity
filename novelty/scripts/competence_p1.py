#!/usr/bin/env python3
"""Phase 1 of the competence-score study (novelty/PREREG.md).

Writes novelty/results/features.npz (cached features) and novelty/results/p1.json.
Run from the repository root:  nice -n 10 python3 novelty/scripts/competence_p1.py
"""
import json
import os
import sys
import time
from dataclasses import replace

import cv2
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RB = os.path.join(ROOT, "rebuild")
OUT = os.path.join(ROOT, "novelty", "results")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, RB)
sys.path.insert(0, os.path.join(RB, "scripts"))
os.environ.setdefault("WB_EXCL", "exclusions_v6.json")
import perimage32 as P32                                          # noqa: E402
import perimage_v7 as V                                           # noqa: E402
import pools_v5 as PV                                             # noqa: E402
from woundbench.imagestats import texture_of_image                # noqa: E402
from woundbench.generator_a import build as build_a               # noqa: E402
from woundbench.generator_b import GenBConfig, GeneratorB         # noqa: E402
from woundbench.generator_c import build as build_c               # noqa: E402

T0 = time.time()
RES = os.path.join(RB, "results")
EXCL = json.load(open(os.path.join(RES, "exclusions_v6.json")))
SETS = ["medetec", "fuseg", "azh", "azh_train_slice"]
REFS3 = ["medetec", "fuseg", "azh"]                   # CO2Wounds-V2 images not available
B = 2000
SHRINK, VFLOOR, BLACK = 0.1, 1e-3, 8


def J(*p):
    return json.load(open(os.path.join(*p)))


# ------------------------------------------------------------------ features
def features(bgr):
    rgb = bgr[..., ::-1].reshape(-1, 3).astype(np.float64)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
    f = []
    for ch in np.concatenate([rgb, hsv], 1).T:
        f += [ch.mean(), ch.std(), *np.percentile(ch, [5, 50, 95])]
    f.append(float((bgr.max(2) <= BLACK).mean()))
    f.append(texture_of_image(bgr))
    h, w = bgr.shape[:2]
    f += [h, w, w / h]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f.append(float((cv2.Canny(g, 100, 200) > 0).mean()))
    return np.array(f, float)


FNAMES = [f"{c}_{s}" for c in ("R", "G", "B", "H", "S", "V") for s in ("mean", "sd", "p5", "p50", "p95")] + \
         ["black_share", "texture", "height", "width", "aspect", "edge_density"]
I_BLACK = FNAMES.index("black_share")


def load_set(s):
    d = J(RES, "perimage32", s + ".json")
    load = dict(P32.LOADERS)[d["loader"]]
    drop = {e[0] for e in EXCL.get(s, [])}
    F, frac = [], []
    for k, (ip, mp) in enumerate(zip(d["image_paths"], d["mask_paths"])):
        if k in drop:
            continue
        im, m = load(os.path.join(RB, ip), os.path.join(RB, mp))
        F.append(features(im))
        frac.append(float(m.mean()))
    return np.array(F), np.array(frac)


cache = os.path.join(OUT, "features.npz")
if os.path.exists(cache):
    C = dict(np.load(cache))
else:
    C = {}
    idx = J(RES, "v7", "fuseg_train_index.json")
    trd = os.path.join(RB, "data", "wound-segmentation", "data", "Foot Ulcer Segmentation Challenge", "train")
    imgs, masks = P32.wound_pairs(os.path.join(trd, "images"), os.path.join(trd, "labels"), 0, 550)
    assert [os.path.basename(p) for p in imgs] == idx["files"]
    Ft, ft = [], []
    for ip, mp in zip(imgs, masks):
        im, m = P32.load_wound(ip, mp)
        Ft.append(features(im))
        ft.append(float(m.mean()))
    C["train_F"], C["train_frac"] = np.array(Ft), np.array(ft)
    for s in SETS:
        C[f"{s}_F"], C[f"{s}_frac"] = load_set(s)
    np.savez(cache, **C)
    print(f"features {time.time() - T0:.0f}s", flush=True)

idx = J(RES, "v7", "fuseg_train_index.json")
dup = set(idx["eval_duplicates"])
FIT = [i for i in range(0, 393) if i not in dup]
HOLD = [i for i in range(400, 550) if i not in dup]

# ------------------------------------------------------------------ the score
Xf = C["train_F"][FIT]
mu, sd = Xf.mean(0), Xf.std(0)
sd[sd == 0] = 1.0


def z(F):
    return (F - mu) / sd


Zf = z(Xf)
q = np.quantile(C["train_frac"][FIT], [1 / 3, 2 / 3])
cls = np.digitize(C["train_frac"][FIT], q)
vpool = Zf.var(0)
GAUSS = []
for c in range(3):
    Zc = Zf[cls == c]
    v = np.maximum((1 - SHRINK) * Zc.var(0) + SHRINK * vpool, VFLOOR)
    GAUSS.append((Zc.mean(0), v))


def score(F):
    Z = z(F)
    ll = [np.log(1 / 3) - 0.5 * np.sum(np.log(2 * np.pi * v) + (Z - m) ** 2 / v, 1) for m, v in GAUSS]
    return logsumexp(np.stack(ll, 1), 1)


THR = float(np.percentile(score(C["train_F"][HOLD]), 5))
SC = {s: score(C[f"{s}_F"]) for s in SETS}
R = {"prereg": "novelty/PREREG.md", "n_fit": len(FIT), "n_threshold_images": len(HOLD),
     "class_counts": [int((cls == c).sum()) for c in range(3)], "threshold": THR, "features": FNAMES,
     "resnet50_embedding": "skipped: torchvision not installed and the weight download was refused by the network proxy (HTTP 403)",
     "co2wounds": "dropped: images not available", "exclusions": "exclusions_v6.json"}
R["score_summary"] = {s: {"n": int(len(v)), "median": float(np.median(v)),
                          "share_below_threshold": float((v < THR).mean())} for s, v in SC.items()}

# ------------------------------------------------------------------ method scores
DATA = {s: V.load(s) for s in SETS}
for s in SETS:
    assert len(DATA[s]["ids"]) == len(SC[s]), s
    assert np.allclose(DATA[s]["frac"], C[f"{s}_frac"], atol=1e-9), s
INDIV = [m for c in V.GROUPS["P10s"] + V.GROUPS["N8s"] for m in V.SEEDCOLS[c]]
R["n_individual_pretrained_models"] = len(INDIV)


def auroc(s, y):
    """P(s of a positive > s of a negative), ties count half."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spear(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, y, c):
    ra, ry, rc = rankdata(a), rankdata(y), rankdata(c)
    X = np.stack([np.ones_like(rc), rc], 1)
    ea = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    if ea.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ea, ey)[0, 1])


# ------------------------------------------------------------------ H1
H1 = {}
per = {}
for s in SETS:
    D = DATA[s]
    ne = ~D["empty"]
    y = (D["X"][:, V.cols(INDIV)] < 0.1).mean(1)[ne]
    per[s] = {"neg_score": -SC[s][ne], "black": C[f"{s}_F"][ne, I_BLACK], "neg_frac": -D["frac"][ne],
              "y": y, "units": D["units"][ne]}
PRED = {"score": "neg_score", "black_share": "black", "lesion_fraction": "neg_frac"}


def h1_stats(d):
    maj = (d["y"] >= 0.5).astype(int)
    o = {k: {"spearman": spear(d[v], d["y"]), "auroc": auroc(d[v], maj)} for k, v in PRED.items()}
    o["partial_spearman_score_given_black"] = partial_spear(d["neg_score"], d["y"], d["black"])
    return o


def flat(o):
    return {f"{k}|{q}": v for k, vv in o.items() for q, v in (vv.items() if isinstance(vv, dict) else [("est", vv)])}


rng = np.random.default_rng(2026)


def boot_idx(units):
    u = np.unique(units)
    pick = rng.choice(u, len(u), replace=True)
    pos = {k: np.where(units == k)[0] for k in u}
    return np.concatenate([pos[k] for k in pick])


def combine(sets, idxs=None):
    keys = ["neg_score", "black", "neg_frac", "y"]
    return {k: np.concatenate([per[s][k] if idxs is None else per[s][k][idxs[s]] for s in sets]) for k in keys}


groups = {s: [s] for s in SETS}
groups["pooled"] = SETS
for g, sets in groups.items():
    d = combine(sets)
    est = flat(h1_stats(d))
    n_maj = int((d["y"] >= 0.5).sum())
    draws = {k: [] for k in est}
    for _ in range(B):
        ii = {s: boot_idx(per[s]["units"]) for s in sets}
        fb = flat(h1_stats(combine(sets, ii)))
        for k in est:
            draws[k].append(fb[k])
    H1[g] = {"n_nonempty": int(len(d["y"])), "n_majority_miss": n_maj,
             "auroc_reported": n_maj >= (10 if g == "pooled" else 5),
             "stats": {k: {"est": est[k], "ci": [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]}
                       for k in est}}
pst = H1["pooled"]["stats"]
H1["verdict"] = {"testable": H1["pooled"]["n_majority_miss"] >= 10,
                 "auroc_lower_ci": pst["score|auroc"]["ci"][0],
                 "partial_ci": pst["partial_spearman_score_given_black|est"]["ci"]}
H1["verdict"]["supported"] = bool(H1["verdict"]["testable"] and H1["verdict"]["auroc_lower_ci"] > 0.6 and
                                  (pst["partial_spearman_score_given_black|est"]["ci"][0] > 0 or
                                   pst["partial_spearman_score_given_black|est"]["ci"][1] < 0))
R["H1"] = H1
print(f"H1 done {time.time() - T0:.0f}s", flush=True)

# ------------------------------------------------------------------ H2
M19 = V.cols(V.GROUPS["M19"])
PAIRS = [(a, b) for i, a in enumerate(REFS3) for b in REFS3[i + 1:]]


def wound_agree(means, takes=None):
    """means: dict set -> vector over the 19 methods; takes: (B, 19) indices or None."""
    if takes is None:
        return float(np.mean([spear(means[a], means[b]) for a, b in PAIRS]))
    out = np.zeros(len(takes))
    for a, b in PAIRS:
        ra = np.apply_along_axis(rankdata, 1, means[a][takes])
        rb = np.apply_along_axis(rankdata, 1, means[b][takes])
        ra -= ra.mean(1, keepdims=True)
        rb -= rb.mean(1, keepdims=True)
        den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            out += np.where(den > 0, (ra * rb).sum(1) / den, np.nan)
    return out / len(PAIRS)


keep = {s: SC[s] >= THR for s in REFS3}
m_all = {s: DATA[s]["X"][:, M19].mean(0) for s in REFS3}
n_removed = {s: int((~keep[s]).sum()) for s in REFS3}
H2 = {"pairs": [f"{a}~{b}" for a, b in PAIRS], "n_images": {s: int(len(keep[s])) for s in REFS3},
      "n_removed": n_removed}
if sum(n_removed.values()) == 0:
    H2["verdict"] = {"testable": False, "supported": False}
else:
    m_f = {s: DATA[s]["X"][keep[s]][:, M19].mean(0) for s in REFS3}
    a_all, a_f = wound_agree(m_all), wound_agree(m_f)
    takes = np.random.default_rng(0).integers(0, len(M19), size=(B, len(M19)))
    d_b = wound_agree(m_f, takes) - wound_agree(m_all, takes)
    rr = np.random.default_rng(2027)
    ctrl = []
    for _ in range(1000):
        m_r = {}
        for s in REFS3:
            n = len(keep[s])
            k = np.ones(n, bool)
            k[rr.choice(n, n_removed[s], replace=False)] = False
            m_r[s] = DATA[s]["X"][k][:, M19].mean(0)
        ctrl.append(wound_agree(m_r) - a_all)
    ci = [float(np.nanpercentile(d_b, 2.5)), float(np.nanpercentile(d_b, 97.5))]
    H2.update({"agree_all": a_all, "agree_filtered": a_f, "change": a_f - a_all, "change_ci": ci,
               "control_p95": float(np.percentile(ctrl, 95)), "control_mean": float(np.mean(ctrl)),
               "share_control_ge_observed": float(np.mean(np.array(ctrl) >= a_f - a_all))})
    H2["verdict"] = {"testable": True, "supported": bool(ci[0] > 0 and a_f - a_all > H2["control_p95"])}
R["H2"] = H2
print(f"H2 done {time.time() - T0:.0f}s", flush=True)

# ------------------------------------------------------------------ H3 (exploratory)
SEEDS, AMPS = [2024, 7, 101], [11.0, 28.0, 40.0]


def synth_sets():
    for amp in AMPS:
        for s in SEEDS:
            def f(amp=amp, s=s):
                r = GeneratorB(seed=s, config=replace(GenBConfig(), mottle_amplitude=amp)).generate(15)
                return [x[2] for x in r], [x[3] for x in r]
            yield f"genB_amp{int(amp)}_s{s}", f
    for s in SEEDS:
        yield f"nrA_s{s}", (lambda s=s: build_a(seed=s, n_per_class=15)[:2])
        yield f"nrB0_s{s}", (lambda s=s: tuple(map(list, zip(*[(x[2], x[3]) for x in GeneratorB(seed=s, config=replace(GenBConfig(), mottle_amplitude=0.0)).generate(15)]))))
        yield f"nrC0_s{s}", (lambda s=s: build_c(seed=s, n_per_class=15, noise_amplitude=0.0)[:2])
        yield f"nrC34_s{s}", (lambda s=s: build_c(seed=s, n_per_class=15, noise_amplitude=3.4)[:2])


syn_cache = os.path.join(OUT, "features_synth.npz")
if os.path.exists(syn_cache):
    SF = dict(np.load(syn_cache))
else:
    SF, TEX = {}, {}
    from woundbench.imagestats import texture_of_set
    for name, f in synth_sets():
        ims, mks = f()
        SF[name] = np.array([features(im) for im in ims])
        TEX[name] = float(texture_of_set(ims, mks)[0])
    np.savez(syn_cache, **SF)
    json.dump(TEX, open(os.path.join(OUT, "synth_texture_regenerated.json"), "w"), indent=1)
print(f"synthetic features {time.time() - T0:.0f}s", flush=True)
n_syn = {k: int(len(v)) for k, v in SF.items()}


def frechet_diag(A, Bm):
    za, zb = z(A), z(Bm)
    m1, m2, v1, v2 = za.mean(0), zb.mean(0), za.var(0), zb.var(0)
    return float(np.sum((m1 - m2) ** 2) + np.sum(v1 + v2 - 2 * np.sqrt(v1 * v2)))


TT = PV.load()["dice"]
U = V.GROUPS["U"]
H3 = {"sets": {}, "n_images": n_syn}
for name, Fs in SF.items():
    syn = V.load_synth(name)
    assert syn["X"].shape[0] == len(Fs), (name, syn["X"].shape, len(Fs))
    ms = syn["X"][:, V.cols(V.GROUPS["M19"], syn["names"])].mean(0)
    e = {"mean_score": float(score(Fs).mean()),
         "frechet_mean_3refs": float(np.mean([frechet_diag(Fs, C[f"{s}_F"]) for s in REFS3])),
         "bench_M19_3refs": float(np.mean([spear(ms, m_all[s]) for s in REFS3]))}
    if name in TT and all(n in TT[name] for n in U):
        mu_u = np.array([TT[name][n] for n in U])
        e["bench_UM19_3refs"] = float(np.mean([spear(np.concatenate([mu_u, ms]),
                                                     np.concatenate([DATA[s]["X"][:, V.cols(U)].mean(0), m_all[s]]))
                                               for s in REFS3]))
    H3["sets"][name] = e
for pool in ("M19", "UM19"):
    ks = [k for k, v in H3["sets"].items() if f"bench_{pool}_3refs" in v]
    bv = [H3["sets"][k][f"bench_{pool}_3refs"] for k in ks]
    H3[pool] = {"n_sets": len(ks),
                "spearman_mean_score_vs_bench": spear([H3["sets"][k]["mean_score"] for k in ks], bv) if len(ks) > 2 else None,
                "spearman_frechet_vs_bench": spear([H3["sets"][k]["frechet_mean_3refs"] for k in ks], bv) if len(ks) > 2 else None}
# check that the regenerated sets match the scored ones: per-method synthetic agreement uses the
# stored scores; the regenerated images only give features. Texture of the regenerated generator-B
# sets is compared with the stored kit values.
b6 = J(RES, "v6", "bench_v6.json")
TEX = J(OUT, "synth_texture_regenerated.json")
stored = {**b6["genB_texture"], **b6["nr_texture"]}
H3["regeneration_check_texture_max_abs_diff"] = float(max(abs(TEX[k] - stored[k]) for k in stored))
H3["regeneration_check_n_sets"] = len(stored)
R["H3"] = H3
R["runtime_s"] = round(time.time() - T0)
json.dump(R, open(os.path.join(OUT, "p1.json"), "w"), indent=1)
print(json.dumps({"H1": H1["verdict"], "H2": H2.get("verdict"), "thr": THR}, indent=1))

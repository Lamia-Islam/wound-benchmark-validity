#!/usr/bin/env python3
"""Independent audit of the run-1 and run-2 GPU results before V7 analysis.

Reads the raw JSON files directly (no perimage_v6 / perimage_v7 import except
in check 8, where perimage_v7.load is imported ONLY to compare its output with
an independent recomputation). Writes results/v7/audit/audit_v7.json.

Checks
  1 inventory of every score and rescore file
  2 rescore P0 (Dice, NSD 2 px) versus the plain score files; nnU-Net detail
  3 grid Dice at each model's own threshold versus P0 Dice; grid sanity
  4 thresholds, held-out Dice, training losses, degenerate-model screen, MAnet
  5 nnunet_heldout.json (n = 148)
  6 sam_v2 groups versus run-1 SAM/MedSAM and versus nnU-Net; empty nnU-Net boxes
  7 synth_check.json
  8 independent recomputation of the 19 seed-averaged modern means (medetec, azh)
  9 log.txt scan
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
K1 = os.path.join(RES, "kaggle")
K2 = os.path.join(RES, "kaggle2")
OUTD = os.path.join(RES, "v7", "audit")
KIT = "/home/claude/kaggle_kit"          # run-1 kit (refs, train, manifest), read only

CLIN = ["medetec", "co2wounds", "fuseg", "azh", "azh_train_slice", "kvasir_disjoint",
        "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib", "ph2", "tn3k_test"]
WOUND5 = ["medetec", "co2wounds", "fuseg", "azh", "azh_train_slice"]
SYN = [f"genB_amp{a}_s{s}" for a in (11, 28, 40) for s in (2024, 7, 101)] + \
      [f"{n}_s{s}" for n in ("nrA", "nrB0", "nrC0", "nrC34") for s in (2024, 7, 101)]
R256 = "fuseg_r256"

ORIG10 = ["unet_r18", "unetpp_r18", "deeplabv3p_r18", "fpn_r18", "pspnet_r18",
          "linknet_r18", "manet_r18", "segformer_mitb0", "unet_mbv2", "unet_effb0"]
NEW8 = ["unet_r50", "deeplabv3p_r50", "upernet_convnext_t", "unet_swin_t",
        "segformer_mitb2", "unet_effb3", "fpn_regnety016", "manet_r50"]


def pre_name(b, s):
    return f"{b}_in" if s == 0 else f"{b}_in_s{s}"


def new_name(b, s):
    return f"{b}_s{s}"


def J(p):
    with open(p) as f:
        return json.load(f)


def fl(x):
    return None if x is None else float(x)


EXCL = J(os.path.join(RES, "exclusions_v5.json"))
PI32 = {s: J(os.path.join(RES, "perimage32", s + ".json")) for s in CLIN}


def keep_mask(s, n):
    k = np.ones(n, bool)
    for e in EXCL.get(s, []):
        k[int(e[0])] = False
    return k


# folder -> (root, expected set list, expected names)
SCORE_FOLDERS = {
    "k1/pretrained": (os.path.join(K1, "scores", "pretrained"), CLIN + SYN,
                      [pre_name(b, 0) for b in ORIG10]),
    "k1/nnunet": (os.path.join(K1, "scores", "nnunet"), CLIN + SYN, ["nnunet_2d"]),
    "k1/kvasir": (os.path.join(K1, "scores", "kvasir"), CLIN + SYN,
                  ["logreg_all_kv", "gbm_all_kv", "logreg_colour_kv", "unet_kv"]
                  + [b + "_kv" for b in ORIG10]),
    "k1/sam": (os.path.join(K1, "scores", "sam"), CLIN + SYN, ["sam_vitb_box", "medsam_vitb_box"]),
}
for s in (1, 2):
    SCORE_FOLDERS[f"k2/pretrained_s{s}"] = (os.path.join(K2, "scores", f"pretrained_s{s}"),
                                            CLIN + [R256] + SYN,
                                            [pre_name(b, s) for b in ORIG10])
for s in (0, 1, 2):
    SCORE_FOLDERS[f"k2/modern2_s{s}"] = (os.path.join(K2, "scores", f"modern2_s{s}"),
                                         CLIN + [R256] + SYN, [new_name(b, s) for b in NEW8])
SCORE_FOLDERS["k2/sam_v2_oracle"] = (os.path.join(K2, "scores", "sam_v2_oracle"), CLIN + [R256],
                                     ["medsam_fixed_gtbox"])
SCORE_FOLDERS["k2/sam_v2_nnbox"] = (os.path.join(K2, "scores", "sam_v2_nnbox"), CLIN + [R256],
                                    ["sam_nnbox", "medsam_fixed_nnbox"])
DIVERGED = {"manet_r18_in_s2", "manet_r50_s0", "manet_r50_s2"}

RS_FAMILIES = ["pretrained_s0", "pretrained_s1", "pretrained_s2", "modern2_s0", "modern2_s1",
               "modern2_s2", "kvasir_nets", "kvasir_unet", "nnunet"]
# plain score source for the P0 comparison
RS_SRC = {"pretrained_s0": "k1/pretrained", "pretrained_s1": "k2/pretrained_s1",
          "pretrained_s2": "k2/pretrained_s2", "modern2_s0": "k2/modern2_s0",
          "modern2_s1": "k2/modern2_s1", "modern2_s2": "k2/modern2_s2",
          "kvasir_nets": "k1/kvasir", "kvasir_unet": "k1/kvasir", "nnunet": "k1/nnunet"}
GRID = [round(0.05 * i, 2) for i in range(1, 19)]


def load_scores(folder, s):
    return J(os.path.join(SCORE_FOLDERS[folder][0], s + ".json"))


def load_rs(fam, s):
    return J(os.path.join(K2, "rescore", fam, s + ".json"))


AUD = {}


# ---------------------------------------------------------------- check 1
def check1():
    out = {"folders": {}, "problems": []}
    gt_ref_syn, gt_ref_r256 = {}, None
    for folder, (d, sets, exp_names) in SCORE_FOLDERS.items():
        files = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(d, "*.json")))
        exp_names_present = [n for n in exp_names if n not in DIVERGED]
        fo = {"n_files": len(files), "missing_sets": sorted(set(sets) - set(files)),
              "extra_sets": sorted(set(files) - set(sets)), "names": None, "sets": {}}
        names0 = None
        for s in sets:
            if s not in files:
                continue
            j = load_scores(folder, s)
            n = j["n"]
            D = np.asarray(j["dice"], float)
            N = np.asarray(j["nsd"], float)
            names = j["names"]
            names0 = names if names0 is None else names0
            rec = {"n": n, "n_names": len(names)}
            probs = []
            if j.get("set") != s:
                probs.append(f"set field {j.get('set')}")
            if names != names0:
                probs.append("names differ from first set")
            if len(set(names)) != len(names):
                probs.append("duplicate names")
            if D.shape != (n, len(names)) or N.shape != (n, len(names)):
                probs.append(f"shape dice {D.shape} nsd {N.shape}")
            if len(j["gt_empty"]) != n or len(j["gt_fraction"]) != n:
                probs.append("gt lists length")
            for nm, A in (("dice", D), ("nsd", N)):
                if not np.isfinite(A).all():
                    probs.append(f"{nm} non-finite {int((~np.isfinite(A)).sum())}")
                elif A.min() < 0 or A.max() > 1:
                    probs.append(f"{nm} outside [0,1] {A.min()} {A.max()}")
            if s in CLIN:
                P = PI32[s]
                rec["n_perimage32"] = len(P["ids"])
                if n != len(P["ids"]):
                    probs.append(f"n {n} vs perimage32 {len(P['ids'])}")
                else:
                    ge = j["gt_empty"] == P["gt_empty"]
                    gf = float(np.max(np.abs(np.asarray(j["gt_fraction"]) - np.asarray(P["gt_fraction"]))))
                    rec["gt_empty_equal"] = ge
                    rec["gt_fraction_max_abs_diff"] = gf
                    rec["gt_fraction_bitwise_equal"] = j["gt_fraction"] == P["gt_fraction"]
                    if not ge:
                        probs.append("gt_empty differs from perimage32")
                    if gf > 1e-12:
                        probs.append(f"gt_fraction differs from perimage32 by {gf}")
            elif s in SYN:
                key = (tuple(j["gt_empty"]), tuple(j["gt_fraction"]))
                if s not in gt_ref_syn:
                    gt_ref_syn[s] = (folder, key)
                elif key != gt_ref_syn[s][1]:
                    probs.append(f"synthetic gt differs from {gt_ref_syn[s][0]}")
                rec["n_empty"] = int(sum(j["gt_empty"]))
            elif s == R256:
                key = (tuple(j["gt_empty"]), tuple(j["gt_fraction"]))
                if gt_ref_r256 is None:
                    gt_ref_r256 = (folder, key)
                elif key != gt_ref_r256[1]:
                    probs.append(f"fuseg_r256 gt differs from {gt_ref_r256[0]}")
            rec["problems"] = probs
            if probs:
                out["problems"].append(f"{folder}/{s}: {probs}")
            fo["sets"][s] = rec
        fo["names"] = names0
        fo["expected_names_present"] = exp_names_present
        fo["names_match_expected"] = names0 == exp_names_present
        if names0 != exp_names_present:
            out["problems"].append(f"{folder}: names {names0} vs expected {exp_names_present}")
        if fo["missing_sets"] or fo["extra_sets"]:
            out["problems"].append(f"{folder}: missing {fo['missing_sets']} extra {fo['extra_sets']}")
        out["folders"][folder] = fo
    # synthetic n per set must be identical in every folder that has it
    out["synthetic_sets_checked"] = len(gt_ref_syn)
    out["synthetic_reference_folder"] = sorted({v[0] for v in gt_ref_syn.values()})
    # fuseg_r256: independent recomputation of gt_fraction from the kit references
    try:
        import cv2
        src = sorted(glob.glob(os.path.join(KIT, "refs", "fuseg", "*_img.png")))
        fr, em = [], []
        for f in src:
            im = cv2.imread(f)
            m = cv2.imread(f.replace("_img.png", "_mask.png"), cv2.IMREAD_GRAYSCALE)
            s_ = 256 / max(im.shape[:2])
            if s_ < 1:
                wh = (round(im.shape[1] * s_), round(im.shape[0] * s_))
                m = cv2.resize(m, wh, interpolation=cv2.INTER_NEAREST)
            m = ((m > 127) * 255).astype(np.uint8)       # as written to disk
            y = (m > 0).astype(np.uint8)                  # as read by kit_run.read_dir
            fr.append(float(y.mean()))
            em.append(bool(y.sum() == 0))
        ref = gt_ref_r256[1]
        out["fuseg_r256_recomputed"] = {
            "n_images": len(src), "gt_empty_equal": tuple(em) == ref[0],
            "gt_fraction_max_abs_diff": float(np.max(np.abs(np.asarray(fr) - np.asarray(ref[1]))))}
    except Exception as e:                       # noqa: BLE001
        out["fuseg_r256_recomputed"] = {"error": repr(e)}
    # rescore files
    rs = {"problems": [], "families": {}}
    thr_files = {}
    for f in ("thresholds_pretrained.json", "thresholds_seeds.json", "thresholds_modern2.json",
              "thresholds_kvasir.json"):
        thr_files.update(J(os.path.join(K2, f)))
    for fam in RS_FAMILIES:
        files = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(K2, "rescore", fam, "*.json")))
        fr_ = {"sets": files, "names": None, "thresholds": None}
        for s in CLIN + [R256]:
            if s not in files:
                rs["problems"].append(f"{fam}/{s} missing")
                continue
            r = load_rs(fam, s)
            n, m = r["n"], len(r["names"])
            G = np.asarray(r["grid_dice"], float)
            V = np.asarray(r["var"], float)
            p = []
            if r["set"] != s or r["family"] != fam:
                p.append("set/family field")
            if G.shape != (n, m, 18):
                p.append(f"grid shape {G.shape}")
            if V.shape != (n, m, 3, 6):
                p.append(f"var shape {V.shape}")
            if r["grid_thresholds"] != GRID:
                p.append("grid thresholds")
            if r["variants"] != ["P0", "P1", "P2"] or r["metrics"] != ["dice", "nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1"]:
                p.append("variant/metric labels")
            for nm, A in (("grid", G), ("var", V)):
                if not np.isfinite(A).all():
                    p.append(f"{nm} non-finite")
                elif A.min() < 0 or A.max() > 1:
                    p.append(f"{nm} outside [0,1]")
            if s in CLIN:
                P = PI32[s]
                if n != len(P["ids"]) or r["gt_empty"] != P["gt_empty"] or \
                        float(np.max(np.abs(np.asarray(r["gt_fraction"]) - np.asarray(P["gt_fraction"])))) > 1e-12:
                    p.append("gt differs from perimage32")
            else:
                if (tuple(r["gt_empty"]), tuple(r["gt_fraction"])) != gt_ref_r256[1]:
                    p.append("fuseg_r256 gt differs from score files")
            for nm_, t in zip(r["names"], r["thresholds"]):
                if fam == "nnunet":
                    if t != 0.5:
                        p.append(f"nnunet threshold {t}")
                elif fam == "kvasir_unet":
                    pass
                elif nm_ not in thr_files or abs(thr_files[nm_] - t) > 1e-12:
                    p.append(f"threshold {nm_} {t} vs file {thr_files.get(nm_)}")
            if fr_["names"] is None:
                fr_["names"], fr_["thresholds"] = r["names"], r["thresholds"]
            elif r["names"] != fr_["names"] or r["thresholds"] != fr_["thresholds"]:
                p.append("names/thresholds differ between sets")
            if p:
                rs["problems"].append(f"{fam}/{s}: {p}")
        # names must equal the score-file names of the same models
        src = RS_SRC[fam]
        sn = SCORE_FOLDERS[src][2]
        sn = [x for x in sn if x not in DIVERGED]
        if fam == "kvasir_unet":
            sn = ["unet_kv"]
        elif fam == "kvasir_nets":
            sn = [b + "_kv" for b in ORIG10]
        fr_["names_equal_score_files"] = fr_["names"] == sn
        if fr_["names"] != sn:
            rs["problems"].append(f"{fam}: names {fr_['names']} vs {sn}")
        rs["families"][fam] = fr_
    out["rescore"] = rs
    out["woundseg_files"] = sorted(glob.glob(os.path.join(K2, "**", "woundseg*.json"), recursive=True))
    out["status"] = "PASS" if not out["problems"] and not rs["problems"] else "FAIL"
    AUD["check1_inventory"] = out


# ---------------------------------------------------------------- check 2
def v6_modern_names():
    return [pre_name(b, 0) for b in ORIG10] + ["nnunet_2d"]


def check2():
    out = {"families": {}}
    for fam in RS_FAMILIES:
        src = RS_SRC[fam]
        dd, dn, pairs, n_round_mismatch, sets = 0.0, 0.0, 0, 0, []
        for s in CLIN + [R256]:
            if s not in SCORE_FOLDERS[src][1]:
                continue
            r = load_rs(fam, s)
            j = load_scores(src, s)
            V = np.asarray(r["var"], float)
            sets.append(s)
            for k_, nm in enumerate(r["names"]):
                i1 = j["names"].index(nm)
                a = np.asarray(j["dice"], float)[:, i1]
                b = V[:, k_, 0, 0]
                a2 = np.asarray(j["nsd"], float)[:, i1]
                b2 = V[:, k_, 0, 2]
                dd = max(dd, float(np.max(np.abs(a - b))))
                dn = max(dn, float(np.max(np.abs(a2 - b2))))
                n_round_mismatch += int(np.sum(np.round(a, 5) != b) + np.sum(np.round(a2, 5) != b2))
                pairs += len(a)
        out["families"][fam] = {"source": src, "sets": sets, "image_model_pairs": pairs,
                                "max_abs_diff_dice": dd, "max_abs_diff_nsd2": dn,
                                "values_not_equal_to_round5_of_score": n_round_mismatch,
                                "pass_le_5e-6": max(dd, dn) <= 5e-6 + 1e-12}
    # nnU-Net detail
    nn = {"per_set": {}, "all": {}}
    allD, allN = [], []
    for s in CLIN:
        r = load_rs("nnunet", s)
        j = load_scores("k1/nnunet", s)
        a = np.asarray(j["dice"], float)[:, 0]
        b = np.asarray(r["var"], float)[:, 0, 0, 0]
        a2 = np.asarray(j["nsd"], float)[:, 0]
        b2 = np.asarray(r["var"], float)[:, 0, 0, 2]
        d, d2 = b - a, b2 - a2
        allD.append(d)
        allN.append(d2)
        k = keep_mask(s, len(a))
        empty = np.asarray(j["gt_empty"], bool)
        nn["per_set"][s] = {
            "n": len(a), "n_dice_gt_1e-4": int((np.abs(d) > 1e-4).sum()),
            "n_dice_gt_1e-3": int((np.abs(d) > 1e-3).sum()),
            "n_nsd2_gt_1e-4": int((np.abs(d2) > 1e-4).sum()),
            "max_abs_dice": float(np.abs(d).max()), "max_abs_nsd2": float(np.abs(d2).max()),
            "n_on_empty_gt_changed": int((np.abs(d[empty]) > 1e-4).sum()),
            "mean_dice_run1_all": float(a.mean()), "mean_dice_run2_all": float(b.mean()),
            "mean_dice_run1_kept": float(a[k].mean()), "mean_dice_run2_kept": float(b[k].mean()),
            "delta_mean_dice_kept": float(b[k].mean() - a[k].mean()),
            "mean_nsd2_run1_kept": float(a2[k].mean()), "mean_nsd2_run2_kept": float(b2[k].mean())}
    D = np.concatenate(allD)
    N = np.concatenate(allN)
    ad = np.abs(D)
    nz = ad[ad > 1e-4]
    nn["all"] = {"pairs": int(len(D)), "n_dice_gt_1e-4": int((ad > 1e-4).sum()),
                 "n_dice_gt_1e-3": int((ad > 1e-3).sum()),
                 "n_dice_le_5e-6": int((ad <= 5e-6 + 1e-12).sum()),
                 "n_nsd2_gt_1e-4": int((np.abs(N) > 1e-4).sum()),
                 "n_nsd2_gt_1e-3": int((np.abs(N) > 1e-3).sum()),
                 "abs_dice_diff_quantiles_among_gt_1e-4": {q: float(np.quantile(nz, q)) for q in (0.25, 0.5, 0.75, 0.9, 1.0)} if len(nz) else {},
                 "signed_dice_mean_all": float(D.mean()),
                 "n_run2_higher": int((D > 1e-4).sum()), "n_run2_lower": int((D < -1e-4).sum()),
                 "max_abs_dice": float(ad.max()), "max_abs_nsd2": float(np.abs(N).max())}
    # rank of nnU-Net among the 11 V6 modern methods, run-1 vs run-2 nnU-Net
    ranks = {}
    mod = v6_modern_names()
    for s in CLIN:
        j = load_scores("k1/pretrained", s)
        k = keep_mask(s, j["n"])
        means = {nm: float(np.asarray(j["dice"], float)[k, i].mean()) for i, nm in enumerate(j["names"])}
        nsdm = {nm: float(np.asarray(j["nsd"], float)[k, i].mean()) for i, nm in enumerate(j["names"])}
        r1 = nn["per_set"][s]["mean_dice_run1_kept"]
        r2 = nn["per_set"][s]["mean_dice_run2_kept"]
        n1 = nn["per_set"][s]["mean_nsd2_run1_kept"]
        n2 = nn["per_set"][s]["mean_nsd2_run2_kept"]

        def rank(v, others):
            return 1 + sum(o > v for o in others)
        oth = [means[m] for m in mod[:-1]]
        othn = [nsdm[m] for m in mod[:-1]]
        gap = min(abs(r1 - o) for o in oth)
        ranks[s] = {"rank_dice_run1": rank(r1, oth), "rank_dice_run2": rank(r2, oth),
                    "rank_nsd2_run1": rank(n1, othn), "rank_nsd2_run2": rank(n2, othn),
                    "closest_other_gap_dice": gap,
                    "abs_delta_dice": abs(r2 - r1)}
    nn["rank_among_11_v6_modern"] = ranks
    nn["rank_changes"] = [s for s, v in ranks.items()
                          if v["rank_dice_run1"] != v["rank_dice_run2"]
                          or v["rank_nsd2_run1"] != v["rank_nsd2_run2"]]
    # do run-2 P1/P2 nnU-Net numbers exist only in run 2? (yes; P0 of run 2 is the base)
    out["nnunet_detail"] = nn
    fail = [f for f, v in out["families"].items() if not v["pass_le_5e-6"]]
    out["families_failing_5e-6"] = fail
    out["status"] = "PASS except nnunet (NOTE)" if fail == ["nnunet"] else ("PASS" if not fail else "FAIL")
    AUD["check2_rescore_vs_scores"] = out


# ---------------------------------------------------------------- check 3
def check3():
    out = {"families": {}, "problems": []}
    for fam in RS_FAMILIES:
        mx, n_off, n_cmp, edge = 0.0, 0, 0, []
        for s in CLIN + [R256]:
            r = load_rs(fam, s)
            G = np.asarray(r["grid_dice"], float)
            V = np.asarray(r["var"], float)
            for k, (nm, t) in enumerate(zip(r["names"], r["thresholds"])):
                idx = [i for i, g in enumerate(r["grid_thresholds"]) if abs(g - t) < 1e-9]
                if not idx:
                    n_off += 1
                    continue
                d = np.abs(G[:, k, idx[0]] - V[:, k, 0, 0])
                mx = max(mx, float(d.max()))
                n_cmp += len(d)
                if t in (r["grid_thresholds"][0], r["grid_thresholds"][-1]):
                    edge.append(nm)
        out["families"][fam] = {"max_abs_grid_minus_P0": mx, "comparisons": n_cmp,
                                "models_threshold_off_grid": n_off,
                                "models_at_grid_edge": sorted(set(edge))}
        if mx > 0 or n_off:
            out["problems"].append(fam)
    # sanity: argmax of mean grid Dice on FUSeg test vs own threshold (information only)
    info = {}
    for fam in RS_FAMILIES:
        r = load_rs(fam, "fuseg")
        G = np.asarray(r["grid_dice"], float)
        k = keep_mask("fuseg", r["n"])
        best = [GRID[int(np.argmax(G[k, i, :].mean(0)))] for i in range(len(r["names"]))]
        info[fam] = {nm: {"own": t, "fuseg_test_best": b,
                          "dice_own": float(np.asarray(r["var"], float)[k, i, 0, 0].mean()),
                          "dice_best": float(G[k, i, :].mean(0).max())}
                     for i, (nm, t, b) in enumerate(zip(r["names"], r["thresholds"], best))}
    out["fuseg_test_grid_optimum_info"] = info
    out["status"] = "PASS" if not out["problems"] else "FAIL"
    AUD["check3_grid"] = out


# ---------------------------------------------------------------- check 4
def model_means(folder, nm, sets):
    o = {}
    for s in sets:
        j = load_scores(folder, s)
        if nm not in j["names"]:
            continue
        i = j["names"].index(nm)
        k = keep_mask(s, j["n"])
        D = np.asarray(j["dice"], float)[k, i]
        e = np.asarray(j["gt_empty"], bool)[k]
        o[s] = {"mean": float(D.mean()), "sd": float(D.std()),
                "frac_zero_on_nonempty": float((D[~e] == 0).mean()) if (~e).any() else None}
    return o


def check4():
    rep = J(os.path.join(K2, "train_report_k2.json"))
    rep1 = J(os.path.join(K1, "train_report_pretrained.json"))
    th1 = J(os.path.join(K1, "thresholds_pretrained.json"))
    ths = J(os.path.join(K2, "thresholds_seeds.json"))
    thm = J(os.path.join(K2, "thresholds_modern2.json"))
    rows, flags, problems = [], [], []
    archs = [(b, False) for b in ORIG10] + [(b, True) for b in NEW8]
    for b, new in archs:
        for s in (0, 1, 2):
            nm = new_name(b, s) if new else pre_name(b, s)
            if new:
                folder = f"k2/modern2_s{s}"
            else:
                folder = "k1/pretrained" if s == 0 else f"k2/pretrained_s{s}"
            row = {"arch": b, "new": new, "seed": s, "name": nm}
            if nm in DIVERGED:
                row.update({"status": "diverged, not saved", "in_report": nm in rep,
                            "in_thresholds": nm in (thm if new else ths)})
                if nm in rep or nm in (thm if new else ths):
                    problems.append(f"{nm}: diverged but present in report/thresholds")
                rows.append(row)
                continue
            if not new and s == 0:
                row.update({"threshold": th1[nm], "heldout_dice": rep1[nm]["sweep_dice"],
                            "final_loss": None, "nonfinite_steps": None, "source": "run 1"})
            else:
                r = rep[nm]
                tf = (thm if new else ths)[nm]
                if abs(tf - r["threshold"]) > 1e-12:
                    problems.append(f"{nm}: threshold file {tf} vs report {r['threshold']}")
                el = r.get("epoch_loss") or []
                row.update({"threshold": tf, "heldout_dice": r["sweep_dice"],
                            "final_loss": el[-1] if el else None,
                            "nonfinite_steps": r.get("nonfinite_loss_steps"),
                            "n_train": r.get("n_train"), "n_sweep": r.get("n_sweep"),
                            "epochs": r.get("epochs"), "seed_in_report": r.get("seed"),
                            "weights_source": r.get("weights_source"), "source": "run 2"})
                if r.get("seed") != s or r.get("epochs") != 40 or r.get("n_train") != 392 \
                        or r.get("n_sweep") != 148 or r.get("dry") or r.get("encoder_weights") != "imagenet":
                    problems.append(f"{nm}: report protocol fields {r.get('seed')}, {r.get('epochs')}, "
                                    f"{r.get('n_train')}, {r.get('n_sweep')}, {r.get('dry')}")
            row["clinical"] = model_means(folder, nm, CLIN)
            row["status"] = "ok"
            rows.append(row)
    # flags
    by_arch = {}
    for r in rows:
        if r["status"] == "ok":
            by_arch.setdefault(r["arch"], []).append(r)
    for b, rr in by_arch.items():
        for r in rr:
            sib = [x for x in rr if x is not r]
            if r["heldout_dice"] < 0.7:
                flags.append(f"{r['name']}: held-out Dice {r['heldout_dice']:.3f} < 0.7")
            if sib:
                m = np.mean([x["heldout_dice"] for x in sib])
                if r["heldout_dice"] < m - 0.03:
                    flags.append(f"{r['name']}: held-out {r['heldout_dice']:.3f} vs sibling mean {m:.3f}")
            for s, v in r["clinical"].items():
                if v["sd"] < 0.01:
                    tag = "WOUND SET" if s in WOUND5 else f"off-domain, mean {v['mean']:.4f}"
                    flags.append(f"{r['name']}/{s}: per-image Dice SD {v['sd']:.4f} (near constant; {tag})")
                if s in WOUND5 and sib:
                    ms = np.mean([x["clinical"][s]["mean"] for x in sib])
                    if abs(v["mean"] - ms) > 0.10:
                        flags.append(f"{r['name']}/{s}: mean Dice {v['mean']:.3f} vs sibling mean {ms:.3f}")
    # high final training loss (information)
    loss_note = [f"{r['name']}: final epoch loss {r['final_loss']:.3f}" for r in rows
                 if r["status"] == "ok" and r.get("final_loss") is not None and r["final_loss"] > 0.15]
    # wound-set spread of each architecture across seeds
    spread = {}
    for b, rr in by_arch.items():
        spread[b] = {s: {"seeds": [x["seed"] for x in rr],
                         "means": [round(x["clinical"][s]["mean"], 4) for x in rr],
                         "range": float(max(x["clinical"][s]["mean"] for x in rr) - min(x["clinical"][s]["mean"] for x in rr))}
                     for s in WOUND5}
    # MAnet from the log
    log = open(os.path.join(K2, "log.txt")).read().splitlines()
    manet = {}
    for nm in ("manet_r18_in_s1", "manet_r18_in_s2", "manet_r50_s0", "manet_r50_s1", "manet_r50_s2"):
        ep = []
        for ln in log:
            m = re.search(rf"\]\s+{nm}: epoch (\d+)/40 mean loss (\S+?)(?:,|$)(?: (\d+) non-finite)?", ln)
            if m:
                ep.append({"epoch": int(m.group(1)), "mean_loss": None if m.group(2) == "None" else float(m.group(2)),
                           "nonfinite_so_far": int(m.group(3)) if m.group(3) else 0})
        err = [ln for ln in log if ln.startswith(f"RuntimeError: {nm}:")]
        manet[nm] = {"epochs_logged": ep, "error": err[0] if err else None}
    # all models: any non-finite steps among saved models?
    nf_saved = {r["name"]: r["nonfinite_steps"] for r in rows if r["status"] == "ok" and r.get("nonfinite_steps")}
    # MAnet sensitivity: seed-averaged per-dataset means over the 19 modern methods
    # (run-1 nnU-Net) with and without the two MAnet columns, MAnet ranks among the
    # 18 architectures, and the pooled within-architecture seed SD of dataset means.
    sens = {}
    full3 = [b for b in ORIG10 + NEW8 if b not in ("manet_r18", "manet_r50")]
    for s in CLIN:
        cols = {}
        for b, rr in by_arch.items():
            cols[b] = float(np.mean([x["clinical"][s]["mean"] for x in rr]))
        j = load_scores("k1/nnunet", s)
        k = keep_mask(s, j["n"])
        nnm = float(np.asarray(j["dice"], float)[k, 0].mean())
        allv = list(cols.values()) + [nnm]
        noman = [v for b, v in cols.items() if not b.startswith("manet")] + [nnm]
        order = sorted(cols, key=lambda b: -cols[b])
        wsd = np.sqrt(np.mean([np.var([x["clinical"][s]["mean"] for x in by_arch[b]], ddof=1) for b in full3]))
        sens[s] = {"M19_mean": float(np.mean(allv)), "M17_no_manet_mean": float(np.mean(noman)),
                   "delta": float(np.mean(noman) - np.mean(allv)),
                   "rank_manet_r18_of_18": order.index("manet_r18") + 1,
                   "rank_manet_r50_of_18": order.index("manet_r50") + 1,
                   "pooled_within_arch_seed_sd": float(wsd),
                   "manet_r18_seed_mean": cols["manet_r18"], "manet_r50_seed_mean": cols["manet_r50"],
                   "nnunet_rank_of_19": 1 + sum(v > nnm for v in cols.values()),
                   "nnunet_run2_rank_of_19": 1 + sum(v > float(np.asarray(load_rs("nnunet", s)["var"], float)[k, 0, 0, 0].mean())
                                                     for v in cols.values()),
                   "nnunet_rank_of_17_no_manet": 1 + sum(v > nnm for b, v in cols.items() if not b.startswith("manet"))}
    out = {"rows": rows, "flags": flags, "problems": problems, "loss_note": loss_note,
           "manet_sensitivity": sens,
           "wound_seed_spread": spread, "manet_log": manet, "saved_models_with_nonfinite_steps": nf_saved,
           "status": "FAIL" if problems else ("NOTE" if flags else "PASS")}
    AUD["check4_models"] = out


# ---------------------------------------------------------------- check 5
def check5():
    h = J(os.path.join(K2, "nnunet_heldout.json"))
    man = J(os.path.join(KIT, "manifest.json"))
    dup = man["fuseg_train"]["eval_duplicates"]
    sw = [i for i in range(400, 550) if i not in dup]
    tr = [i for i in range(0, 393) if i not in dup]
    d = np.asarray(h["dice"], float)
    # identity of the removed training images with evaluation images
    ident = {}
    try:
        import cv2
        refs = sorted(glob.glob(os.path.join(KIT, "refs", "fuseg", "*_img.png")))
        names = man["fuseg_train"]["files"]
        for i in dup:
            a = cv2.imread(os.path.join(KIT, "train", "fuseg", f"{i:04d}_img.png"))
            hit = [os.path.basename(p) for p in refs if (b := cv2.imread(p)) is not None
                   and b.shape == a.shape and np.array_equal(a, b)]
            ident[str(i)] = {"fuseg_train_file": names[i], "identical_eval_images": hit}
    except Exception as e:                        # noqa: BLE001
        ident = {"error": repr(e)}
    j = load_scores("k1/nnunet", "fuseg")
    a = np.asarray(j["dice"], float)[:, 0]
    k = keep_mask("fuseg", len(a))
    r = load_rs("nnunet", "fuseg")
    b = np.asarray(r["var"], float)[:, 0, 0, 0]
    rep1 = J(os.path.join(K1, "train_report_pretrained.json"))
    rep2 = J(os.path.join(K2, "train_report_k2.json"))
    sweeps = [v["sweep_dice"] for v in rep1.values()] + [v["sweep_dice"] for v in rep2.values()]
    out = {"n": h["n"], "len_dice": len(d), "mean_dice_file": h["mean_dice"],
           "mean_dice_recomputed": float(d.mean()), "finite": bool(np.isfinite(d).all()),
           "threshold": h["threshold"], "variant": h["variant"],
           "eval_duplicates": dup, "dup_in_sweep_range": [i for i in dup if 400 <= i < 550],
           "dup_in_train_range": [i for i in dup if i < 393],
           "expected_n_sweep": len(sw), "expected_n_train": len(tr), "duplicate_identity": ident,
           "run1_nnunet_fuseg_test_mean_all120": float(a.mean()),
           "run1_nnunet_fuseg_test_mean_kept": float(a[k].mean()), "n_kept": int(k.sum()),
           "run2_nnunet_fuseg_test_mean_kept": float(b[k].mean()),
           "smp_models_heldout_sweep_dice_median": float(np.median(sweeps)),
           "smp_models_heldout_sweep_dice_range": [float(min(sweeps)), float(max(sweeps))],
           "n_heldout_images_below_0.5": int((d < 0.5).sum()), "min_heldout": float(d.min())}
    out["status"] = "PASS" if (h["n"] == len(d) == len(sw) == 148) else "FAIL"
    AUD["check5_nnunet_heldout"] = out


# ---------------------------------------------------------------- check 6
def check6():
    per = {}
    counts = {}
    for s in CLIN + [R256]:
        o = load_scores("k2/sam_v2_oracle", s)
        nb = load_scores("k2/sam_v2_nnbox", s)
        r = load_rs("nnunet", s)
        n = o["n"]
        k = keep_mask(s, n) if s in CLIN else np.ones(n, bool)
        e = np.asarray(o["gt_empty"], bool)
        O = np.asarray(o["dice"], float)[:, 0]
        SN = np.asarray(nb["dice"], float)[:, nb["names"].index("sam_nnbox")]
        MN = np.asarray(nb["dice"], float)[:, nb["names"].index("medsam_fixed_nnbox")]
        V = np.asarray(r["var"], float)[:, 0, 0, :]          # nnU-Net run-2 P0 metrics
        NN2 = V[:, 0]
        row = {"n": n, "n_kept": int(k.sum()), "n_gt_empty": int(e.sum()),
               "medsam_fixed_gtbox": float(O[k].mean()),
               "sam_nnbox": float(SN[k].mean()), "medsam_fixed_nnbox": float(MN[k].mean()),
               "nnunet_run2_P0": float(NN2[k].mean())}
        if s in CLIN:
            s1 = load_scores("k1/sam", s)
            row["sam_vitb_box_run1"] = float(np.asarray(s1["dice"], float)[k, s1["names"].index("sam_vitb_box")].mean())
            row["medsam_vitb_box_run1"] = float(np.asarray(s1["dice"], float)[k, s1["names"].index("medsam_vitb_box")].mean())
            j = load_scores("k1/nnunet", s)
            row["nnunet_run1"] = float(np.asarray(j["dice"], float)[k, 0].mean())
        per[s] = row
        # empty nnU-Net masks (no box). With non-empty GT, an empty P0 mask gives
        # Dice 0 and every NSD 0; then both nnbox methods get no prompt and Dice 0.
        cand = (~e) & (V[:, 0] == 0) & np.all(V[:, 1:] == 0, axis=1) & (SN == 0) & (MN == 0)
        # with empty GT: nnU-Net Dice 1 means an empty mask (both empty)
        empty_on_empty = e & (V[:, 0] == 1)
        # consistency: wherever nnU-Net is empty on empty GT, both nnbox methods must score 1
        incons = int((empty_on_empty & ((SN != 1) | (MN != 1))).sum())
        # nnbox > 0 on a nonempty GT proves a box existed
        counts[s] = {"gt_nonempty_nnunet_empty_upper_bound": int(cand.sum()),
                     "gt_nonempty_nnunet_dice0": int(((~e) & (V[:, 0] == 0)).sum()),
                     "gt_empty_nnunet_empty": int(empty_on_empty.sum()),
                     "gt_empty_nnunet_fp": int((e & (V[:, 0] == 0)).sum()),
                     "inconsistent_empty_handling": incons,
                     "oracle_gt_empty_scored_1": int((e & (O == 1)).sum())}
    tot = {k: int(sum(v[k] for v in counts.values())) for k in next(iter(counts.values()))}
    out = {"per_set_mean_dice_after_exclusions": per, "empty_box_counts": counts, "totals": tot,
           "status": "PASS" if tot["inconsistent_empty_handling"] == 0 else "FAIL"}
    AUD["check6_sam_v2"] = out


# ---------------------------------------------------------------- check 7
def check7():
    out = {}
    for tag, p in (("run2", os.path.join(K2, "synth_check.json")), ("run1", os.path.join(K1, "synth_check.json"))):
        if os.path.exists(p):
            c = J(p)
            out[tag] = {"sets": len(c), "n": sum(v["n"] for v in c.values()),
                        "identical": sum(v["identical"] for v in c.values()),
                        "set_names_match": sorted(c) == sorted(SYN),
                        "all_45": all(v["n"] == 45 for v in c.values())}
    ok = out.get("run2", {}).get("identical") == out.get("run2", {}).get("n") == 945
    out["status"] = "PASS" if ok and out["run2"]["set_names_match"] else "FAIL"
    AUD["check7_synth"] = out


# ---------------------------------------------------------------- check 8
def check8():
    out = {}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import perimage_v7 as V7                      # only to compare
    for s in ("medetec", "azh"):
        n = len(PI32[s]["ids"])
        k = keep_mask(s, n)
        mine = {}
        for b in ORIG10:
            cols = []
            for sd in (0, 1, 2):
                nm = pre_name(b, sd)
                f = "k1/pretrained" if sd == 0 else f"k2/pretrained_s{sd}"
                j = load_scores(f, s)
                if nm in j["names"]:
                    cols.append(np.asarray(j["dice"], float)[:, j["names"].index(nm)])
            mine[f"{b}_in@seeds"] = (float(np.mean(np.mean(np.stack(cols, 1), 1)[k])), len(cols))
        for b in NEW8:
            cols = []
            for sd in (0, 1, 2):
                nm = new_name(b, sd)
                j = load_scores(f"k2/modern2_s{sd}", s)
                if nm in j["names"]:
                    cols.append(np.asarray(j["dice"], float)[:, j["names"].index(nm)])
            mine[f"{b}@seeds"] = (float(np.mean(np.mean(np.stack(cols, 1), 1)[k])), len(cols))
        j = load_scores("k1/nnunet", s)
        mine["nnunet_2d"] = (float(np.asarray(j["dice"], float)[k, 0].mean()), 1)
        D = V7.load(s)
        theirs = {nm: float(D["X"][:, D["names"].index(nm)].mean()) for nm in mine}
        diff = {nm: abs(mine[nm][0] - theirs[nm]) for nm in mine}
        out[s] = {"n_kept": int(k.sum()), "n_v7": int(D["X"].shape[0]),
                  "means": {nm: {"audit": mine[nm][0], "perimage_v7": theirs[nm], "n_seeds": mine[nm][1]}
                            for nm in mine},
                  "max_abs_diff": max(diff.values()), "M19_group_matches": V7.GROUPS["M19"] == list(mine)}
    out["status"] = "PASS" if all(out[s]["max_abs_diff"] <= 1e-12 and out[s]["M19_group_matches"]
                                  for s in ("medetec", "azh")) else "FAIL"
    # exclusion ids match the stored file names
    exc = {}
    for s, lst in EXCL.items():
        ids = PI32[s]["ids"]
        exc[s] = sum(1 for e in lst if not str(e[1]).startswith(ids[int(e[0])]))
    out["exclusion_index_name_mismatches"] = exc
    AUD["check8_recompute"] = out


# ---------------------------------------------------------------- check 9
def check9():
    log = open(os.path.join(K2, "log.txt")).read().splitlines()
    pats = {"FAILED": r"FAILED", "WARNING": r"WARNING", "skipped": r"skipped",
            "missing": r"missing", "LARGE": r"LARGE", "non-finite": r"non-finite",
            "Traceback": r"Traceback", "moved": r"\bmoved\b", "recomputing": r"recomputing",
            "differ": r"differ"}
    hits = {k: [f"{i + 1}: {ln.strip()[:220]}" for i, ln in enumerate(log) if re.search(p, ln)]
            for k, p in pats.items()}
    stages = [f"{i + 1}: {ln.strip()}" for i, ln in enumerate(log)
              if re.search(r"\] stage \w+ (on|done|FAILED)", ln)]
    out = {"n_lines": len(log), "stage_lines": stages,
           "counts": {k: len(v) for k, v in hits.items()}, "hits": hits,
           "last_line": log[-1] if log else None,
           "done_markers": sorted(os.path.basename(p) for p in glob.glob(os.path.join(K2, "DONE_*"))),
           "run1_done_markers": sorted(os.path.basename(p) for p in glob.glob(os.path.join(K1, "DONE_*")))}
    # scored set counts per group from the log
    cnt = {}
    for ln in log:
        m = re.search(r"\]\s+(pretrained_s\d|modern2_s\d|sam_v2_\w+?)/(\w+): (\d+) images", ln)
        if m:
            cnt.setdefault(m.group(1), set()).add(m.group(2))
    out["sets_scored_in_log"] = {k: len(v) for k, v in cnt.items()}
    rlog = open(os.path.join(K1, "log.txt")).read().splitlines()
    out["run1_log_flag_lines"] = [f"{i + 1}: {ln.strip()[:220]}" for i, ln in enumerate(rlog)
                                  if re.search(r"FAIL|WARN|could not|error|nan|non-finite", ln, re.I)]
    AUD["check9_log"] = out


def main():
    os.makedirs(OUTD, exist_ok=True)
    for f in (check1, check2, check3, check4, check5, check6, check7, check8, check9):
        f()
        print(f.__name__, AUD[list(AUD)[-1]].get("status", "info"))
    with open(os.path.join(OUTD, "audit_v7.json"), "w") as fh:
        json.dump(AUD, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", os.path.join(OUTD, "audit_v7.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Facts for manuscript V6: every number printed in manuscript_v6 comes from here.

Reads the results JSON files (read-only) and writes manuscript_v6/facts_v6.json,
a flat map  key -> {"v": formatted string, "raw": number or list, "src": "file:path"}.
build_v6.js substitutes {{key}} in all text, table cells and captions.

Formatting rules (BRIEF_MS_V6.md): correlations and Dice to 2 decimals, p values
to 2 significant digits, percentages to 1 decimal, negative numbers with the
Unicode minus sign U+2212. Selection losses and gaps between neighbouring
methods are Dice differences below 0.1 and are given to 3 decimals.
Intervals are written "a to b".
"""
import json
import math
import os
import re
import statistics

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                 # .../rebuild
RES = os.path.join(ROOT, "results")
V7 = os.path.join(RES, "v7")
OUT = os.path.join(os.path.dirname(ROOT), "manuscript_v6", "facts_v6.json")
MINUS = "−"

_cache = {}


def load(rel):
    if rel not in _cache:
        with open(os.path.join(RES, rel)) as f:
            _cache[rel] = json.load(f)
    return _cache[rel]


def J(rel, path):
    """Value at a '/'-separated path inside results/<rel>."""
    x = load(rel)
    for k in path.split("/"):
        if isinstance(x, list):
            x = x[int(k)]
        else:
            if k not in x:
                raise KeyError(f"{rel}:{path} (missing {k!r})")
            x = x[k]
    return x


F = {}


def put(key, v, raw, src):
    if key in F:
        raise KeyError("duplicate fact " + key)
    if not isinstance(v, str):
        raise TypeError(key)
    if "-" in re.sub(r"[A-Za-z]-[A-Za-z0-9]", "", v) and re.search(r"(^|\s)-\d", v):
        raise ValueError(f"{key}: ASCII minus in {v!r}")
    F[key] = {"v": v, "raw": raw, "src": src}


def neg(s):
    return s.replace("-", MINUS) if s.startswith("-") else s


def f2(x):
    s = f"{x:.2f}"
    if s in ("-0.00",):
        s = "0.00"
    return neg(s)


def f3(x):
    s = f"{x:.3f}"
    if s == "-0.000":
        s = "0.000"
    return neg(s)


def f1(x):
    s = f"{x:.1f}"
    if s == "-0.0":
        s = "0.0"
    return neg(s)


def pct(x):
    return f1(100 * x) + "%"


PMIN = 2 / 13860


def fp(p):
    if p < 1e-4:
        raise ValueError(p)
    s = f"{p:#.2g}"
    if "e" in s:
        raise ValueError(s)
    return s


def ival(lo, hi, fmt=f2):
    return f"{fmt(lo)} to {fmt(hi)}"


def num(key, rel, path, fmt=f2):
    x = J(rel, path)
    put(key, fmt(x), x, f"{rel}:{path}")
    return x


def ci(key, rel, path, fmt=f2):
    lo, hi = J(rel, path)
    put(key, ival(lo, hi, fmt), [lo, hi], f"{rel}:{path}")
    return lo, hi


def pval(key, rel, path):
    p = J(rel, path)
    put(key, fp(p), p, f"{rel}:{path}")
    tail = ", the smallest attainable" if abs(p - PMIN) < 1e-12 else ""
    put(key + ".t", fp(p) + tail, p, f"{rel}:{path}")
    return p


def intval(key, x, src):
    put(key, f"{x:,}", x, src)


ST = "v7/stats_v7.json"
BE = "v7/bench_v7.json"
PR = "v7/practical_v7.json"
RB = "v7/robust_v7.json"
RP = "v7/robust_pooled_v7.json"
AU = "v7/audit/audit_v7.json"

SETK = {"U": "U", "F": "F", "K": "K", "M": "M", "M11s": "M11s", "M19": "M19",
        "N8s": "N8s", "U+F": "UF", "U+K": "UK", "U+M": "UM", "U+M19": "UM19",
        "ALL57": "ALL57", "ALL": "ALL"}
WOUND = ["medetec", "co2wounds", "fuseg", "azh"]
OTHER = ["kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb",
         "etis_larib", "ph2", "tn3k_test"]
SETS12 = WOUND + ["azh_train_slice"] + OTHER
SK = {"medetec": "med", "co2wounds": "co2", "fuseg": "fus", "azh": "azh",
      "azh_train_slice": "ctrl", "kvasir_disjoint": "kva", "kvasir_sessile": "kvs",
      "cvc_clinicdb": "cvc1", "cvc_colondb": "cvc2", "etis_larib": "etis",
      "ph2": "ph2", "tn3k_test": "tn3k"}

NAME = {  # display names of architectures (from the method identifiers in the JSON)
    "unet_r18": "U-Net (ResNet-18)", "unetpp_r18": "U-Net++ (ResNet-18)",
    "deeplabv3p_r18": "DeepLabV3+ (ResNet-18)", "fpn_r18": "FPN (ResNet-18)",
    "pspnet_r18": "PSPNet (ResNet-18)", "linknet_r18": "LinkNet (ResNet-18)",
    "manet_r18": "MA-Net (ResNet-18)", "segformer_mitb0": "SegFormer (MiT-B0)",
    "unet_mbv2": "U-Net (MobileNetV2)", "unet_effb0": "U-Net (EfficientNet-B0)",
    "unet_r50": "U-Net (ResNet-50)", "deeplabv3p_r50": "DeepLabV3+ (ResNet-50)",
    "upernet_convnext_t": "UPerNet (ConvNeXt-T)", "unet_swin_t": "U-Net (Swin-T)",
    "segformer_mitb2": "SegFormer (MiT-B2)", "unet_effb3": "U-Net (EfficientNet-B3)",
    "fpn_regnety016": "FPN (RegNetY-1.6GF)", "manet_r50": "MA-Net (ResNet-50)",
    "nnunet_2d": "nnU-Net",
}


def disp(ident):
    b = ident.replace("@seeds", "")
    b = re.sub(r"_s[012]$", "", b)
    b = re.sub(r"_in$", "", b)
    return NAME[b]


# ------------------------------------------------------------------ datasets
EX = load("exclusions_v6.json")
for s in SETS12:
    k = SK[s]
    num(f"n.img.{k}", ST, f"n_images/{s}", lambda x: f"{x:,}")
    num(f"n.unit.{k}", ST, f"n_units/{s}", lambda x: f"{x:,}")
    num(f"n.empty.{k}", ST, f"n_empty/{s}", lambda x: f"{x:,}")
    d = load(f"perimage32/{s}.json")
    keep = np.ones(len(d["ids"]), bool)
    for r in EX.get(s, []):
        keep[r[0]] = False
    fr = float(np.median(np.asarray(d["gt_fraction"])[keep]))
    put(f"frac.{k}", f"{fr:.3f}", fr, f"perimage32/{s}.json:gt_fraction (median of kept images)")
    assert int(keep.sum()) == J(ST, f"n_images/{s}")
    assert int(np.asarray(d["gt_empty"])[keep].sum()) == J(ST, f"n_empty/{s}")
fr_w = [F[f"frac.{SK[s]}"]["raw"] for s in WOUND]
ratio = max(fr_w) / min(fr_w)
put("frac.ratio", str(round(ratio)), ratio, "perimage32: max/min median lesion fraction of the 4 wound datasets")
put("frac.wound.range", f"{min(fr_w):.3f} to {max(fr_w):.3f}", [min(fr_w), max(fr_w)], "perimage32 medians")

fx = EX["fuseg"]
EXN = "exclusions_v6.json"
intval("n.excl.fus", len(fx), EXN + ":fuseg")
intval("n.excl.fus.train", sum("FUSeg train" in r[2] for r in fx), EXN + ":fuseg reasons")
intval("n.excl.fus.eval", sum("evaluation image" in r[2] for r in fx), EXN + ":fuseg reasons")
cx = EX["azh_train_slice"]
intval("n.excl.ctrl", len(cx), EXN + ":azh_train_slice")
intval("n.excl.ctrl.crop", sum(r[2].startswith("crop of FUSeg") for r in cx), EXN + " reasons")
intval("n.excl.ctrl.train", sum("crop of FUSeg train" in r[2] for r in cx), EXN + " reasons")
intval("n.excl.ctrl.val", sum("crop of FUSeg validation" in r[2] for r in cx), EXN + " reasons")
intval("n.excl.ctrl.azhdup", sum(r[2].startswith("pixel-identical to AZH test") for r in cx), EXN + " reasons")
assert F["n.excl.ctrl.crop"]["raw"] + F["n.excl.ctrl.azhdup"]["raw"] == len(cx)
minin = min(int(re.search(r"\((\d+) SIFT inliers", r[2]).group(1)) for r in cx if r[2].startswith("crop of FUSeg"))
intval("n.excl.ctrl.mininl", minin, EXN + " reasons (smallest SIFT inlier count)")
intval("n.ctrl.before", J(ST, "n_images/azh_train_slice") + len(cx), "stats_v7 n_images + exclusions")
ax = EX["azh"]
intval("n.excl.azh", len(ax), EXN + ":azh")
intval("n.azh.before", J(ST, "n_images/azh") + len(ax), "stats_v7 n_images + exclusions")
# overlap check of AZH test and control against all FUSeg photographs (results/v7/azh_overlap_v7.json)
OV, XS, CO = "v7/azh_overlap_v7.json", "v7/exclusions_v6_summary.json", "v7/azh_control_overlap_v7.json"
put("sift.ratio", re.search(r"ratio ([0-9.]+)", J(OV, "rule")).group(1), J(OV, "rule"), OV + ":rule")
put("sift.min", re.search(r">= (\d+) inliers", J(OV, "rule")).group(1), J(OV, "rule"), OV + ":rule")
_fn = re.search(r"train \((\d+)\) \+ validation \((\d+)\)", J(OV, "rule"))
intval("sift.fus.n", int(_fn.group(1)) + int(_fn.group(2)), OV + ":rule (FUSeg train + validation photographs)")
intval("sift.fus.train", int(_fn.group(1)), OV + ":rule")
intval("sift.fus.val", int(_fn.group(2)), OV + ":rule")
intval("ov.azh.n", J(OV, "azh_test/n"), OV + ":azh_test/n")
intval("ov.azh.match", J(OV, "azh_test/n_match"), OV + ":azh_test/n_match")
intval("ov.azh.train", J(OV, "azh_test/n_match_train"), OV + ":azh_test/n_match_train")
intval("ov.azh.val", J(OV, "azh_test/n_match_validation"), OV + ":azh_test/n_match_validation")
intval("ov.azh.valref", J(XS, "azh/n_match_validation_in_reference"), XS + ":azh/n_match_validation_in_reference")
intval("ov.azh.used", J(XS, "azh/n_excluded"), XS + ":azh/n_excluded")
intval("ov.azh.used.train", J(XS, "azh/excluded_by_use/train"), XS + ":azh/excluded_by_use/train")
intval("ov.azh.unused", J(XS, "azh/n_kept_crops_of_unused_frames"), XS + ":azh/n_kept_crops_of_unused_frames")
intval("ov.azh.maxnon", J(OV, "azh_test/max_inliers_nonmatch"), OV + ":azh_test/max_inliers_nonmatch")
intval("ov.ctrl.v5", J(XS, "azh_train_slice/n_v5"), XS + ":azh_train_slice/n_v5")
intval("ov.ctrl.found", J(XS, "azh_train_slice/v5_found_again"), XS + ":azh_train_slice/v5_found_again")
intval("ov.ctrl.new", J(XS, "azh_train_slice/n_new_crops_excluded"), XS + ":azh_train_slice/n_new_crops_excluded")
intval("ov.ctrl.new.train", J(XS, "azh_train_slice/new_crops_by_use/train"), XS + ":azh_train_slice/new_crops_by_use/train")
intval("ov.ctrl.new.ref", J(XS, "azh_train_slice/new_crops_by_use/reference"), XS + ":azh_train_slice/new_crops_by_use/reference")
intval("ov.ctrl.azh", J(CO, "n_control_images_matched"), CO + ":n_control_images_matched")
intval("ov.ctrl.azh.added", J(XS, "azh_train_slice/n_copies_of_azh_test_added"), XS + ":azh_train_slice/n_copies_of_azh_test_added")
assert J(XS, "azh/n_excluded") == len(ax) and J(XS, "azh_train_slice/n_excluded") == len(cx)
assert J(XS, "azh_train_slice/v5_by_use/train") + J(XS, "azh_train_slice/new_crops_by_use/train") == F["n.excl.ctrl.train"]["raw"]
assert J(XS, "azh_train_slice/v5_by_use/reference") + J(XS, "azh_train_slice/new_crops_by_use/reference") == F["n.excl.ctrl.val"]["raw"]
assert F["ov.azh.used"]["raw"] + F["ov.azh.unused"]["raw"] == F["ov.azh.match"]["raw"] and F["ov.azh.valref"]["raw"] == 0
assert F["ov.ctrl.v5"]["raw"] + F["ov.ctrl.new"]["raw"] + F["ov.ctrl.azh.added"]["raw"] == len(cx)
ws = J(AU, "check9_log/hits/FAILED/0")
m = re.search(r"(\d+) of (\d+) masks differ in size", ws)
intval("ws.bad", int(m.group(1)), AU + ":check9_log/hits/FAILED/0")
intval("ws.n", int(m.group(2)), AU + ":check9_log/hits/FAILED/0")

# ------------------------------------------------------------------ method groups
for g in ["U", "F", "K", "M", "M11s", "M19", "N8s", "U+F", "U+K", "U+M", "U+M19", "ALL"]:
    num(f"n.{SETK[g]}", ST, f"definitions/method_sets/{g}/n", lambda x: str(x))
for g in ["P9", "N7", "C16"]:
    num(f"n.{g}", ST, f"seeds/dice|{g}/n", lambda x: str(x))
n_m19 = J(ST, "seeds/dice|C16/n")
put("n.pre", str(len(J(BE, "groups/P10s")) + len(J(BE, "groups/N8s"))),
    len(J(BE, "groups/P10s")) + len(J(BE, "groups/N8s")), BE + ":groups/P10s + groups/N8s")
put("n.P10", str(len(J(BE, "groups/P10s"))), len(J(BE, "groups/P10s")), BE + ":groups/P10s")
put("n.N8", str(len(J(BE, "groups/N8s"))), len(J(BE, "groups/N8s")), BE + ":groups/N8s")
TR = load("kaggle2/train_report_k2.json")
put("n.train", str({r["n_train"] for r in TR.values()}.pop()), 392, "kaggle2/train_report_k2.json:*/n_train")
assert len({r["n_train"] for r in TR.values()}) == 1 and len({r["n_sweep"] for r in TR.values()}) == 1
put("n.sweep", str({r["n_sweep"] for r in TR.values()}.pop()), 148, "kaggle2/train_report_k2.json:*/n_sweep")
put("ep.pre", str({r["epochs"] for r in TR.values()}.pop()), 40, "kaggle2/train_report_k2.json:*/epochs")
vers = {json.dumps(r["versions"], sort_keys=True) for r in TR.values()}
assert len(vers) == 1
vv = json.loads(vers.pop())
put("ver.torch", vv["torch"].split("+")[0], vv["torch"], "kaggle2/train_report_k2.json:*/versions/torch")
put("ver.smp", vv["smp"], vv["smp"], "kaggle2/train_report_k2.json:*/versions/smp")
put("ver.timm", vv["timm"], vv["timm"], "kaggle2/train_report_k2.json:*/versions/timm")
# run-2 trainings: 10 original x seeds 1, 2 and 8 new x seeds 0, 1, 2
n_run2 = 10 * 2 + 8 * 3
n_ok = len(TR)
intval("n.run2", n_run2, "design: 10 architectures x 2 seeds + 8 architectures x 3 seeds")
intval("n.run2.ok", n_ok, "kaggle2/train_report_k2.json (entries = saved models)")
intval("n.run2.fail", n_run2 - n_ok, "n.run2 minus n.run2.ok")
manet_run2 = 2 + 3
manet_ok = sum(1 for k in TR if k.startswith("manet"))
intval("n.manet.run2", manet_run2, "design: manet_r18 seeds 1, 2; manet_r50 seeds 0, 1, 2")
intval("n.manet.fail", manet_run2 - manet_ok, "kaggle2/train_report_k2.json (MA-Net entries)")
intval("n.other.run2", n_run2 - manet_run2, "design")
assert n_run2 - n_ok == manet_run2 - manet_ok
# held-out Dice of every individual pretrained model (run 1 and run 2)
ho = [v["sweep_dice"] for v in load("kaggle/train_report_pretrained.json").values()] + [r["sweep_dice"] for r in TR.values()]
put("ho.pre", ival(min(ho), max(ho)), [min(ho), max(ho)], "kaggle/train_report_pretrained.json + kaggle2/train_report_k2.json: sweep_dice")
intval("n.pre.models", len(ho), "train reports (individual pretrained models)")
num("ho.nn", RB, "heldout/mean_stored")
num("ho.nn.test", RB, "heldout/fuseg_test_r1/mean")
# nnU-Net re-prediction differences


def find_paths(obj, pred, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += find_paths(v, pred, path + "/" + k if path else k)
    elif pred(path, obj):
        out.append((path, obj))
    return out


# ------------------------------------------------------------------ agreement (stats_v7)
LEV = {"ctrl": "same dataset", "src": "same source", "dsrc": "different source, same modality",
       "across": "across modality", "wound": "wound pairs", "colon": "colonoscopy pairs",
       "mdiff": "modality_diff", "mdiff52": "modality_diff_excl_within_source",
       "w_in_out": "comp|wound|in x out", "w_out_same": "comp|wound|out, same modality",
       "w_out_across": "comp|wound|out, across modality",
       "c_in_out": "comp|colonoscopy|in x out", "c_out_same": "comp|colonoscopy|out, same modality",
       "c_out_across": "comp|colonoscopy|out, across modality", "fa": "fuseg~azh"}
for met in ["dice", "nsd"]:
    for g in ["U", "F", "K", "M", "M11s", "M19", "N8s", "U+F", "U+K", "U+M", "U+M19", "ALL"]:
        for q, lab in LEV.items():
            base = f"a.{met}.{SETK[g]}.{q}"
            num(base, ST, f"point/{met}|{g}/{lab}")
            ci(base + ".m", ST, f"bootstrap/{met}|{g}/{lab}/method_only_repro_B2000")
            ci(base + ".h", ST, f"bootstrap/{met}|{g}/{lab}/hierarchical")
        pval(f"p.{met}.{SETK[g]}", ST, f"exact/{met}|{g}/full_55/p")
        pval(f"p52.{met}.{SETK[g]}", ST, f"exact/{met}|{g}/excl_within_source_52/p")
intval("n.relabel", J(ST, "exact/dice|U/full_55/n_relabelings"), ST + ":exact/dice|U/full_55/n_relabelings")
put("p.min", fp(J(ST, "exact/dice|U/full_55/min_p")), J(ST, "exact/dice|U/full_55/min_p"), ST + ":exact/dice|U/full_55/min_p")
assert abs(J(ST, "exact/dice|U/full_55/min_p") - PMIN) < 1e-12
for g in ["U", "F", "K", "M", "M11s", "M19", "N8s", "U+F", "U+K", "U+M", "U+M19", "ALL"]:
    num(f"ne.{SETK[g]}.mdiff", ST, f"dice_noempty/{g}/modality_diff")
    pval(f"ne.{SETK[g]}.p", ST, f"dice_noempty/{g}/p")
    num(f"ne.{SETK[g]}.wound", ST, f"dice_noempty/{g}/wound pairs")
# ranges of p and differences over U, F, K (Dice)
for met in ["dice", "nsd"]:
    ds = [J(ST, f"point/{met}|{g}/modality_diff") for g in ["U", "F", "K"]]
    put(f"mdiff.{met}.UFK.range", ival(min(ds), max(ds)), [min(ds), max(ds)], ST + f":point/{met}|U,F,K/modality_diff")
    ps = [J(ST, f"exact/{met}|{g}/full_55/p") for g in ["U", "F", "K"]]
    put(f"p.{met}.UFK.max", fp(max(ps)), max(ps), ST + f":exact/{met}|U,F,K/full_55/p (max)")
ne = [J(ST, f"dice_noempty/{g}/modality_diff") for g in ["U", "F", "K"]]
put("ne.UFK.range", ival(min(ne), max(ne)), [min(ne), max(ne)], ST + ":dice_noempty/U,F,K/modality_diff")
nep = [J(ST, f"dice_noempty/{g}/p") for g in ["U", "F", "K"]]
put("ne.UFK.pmax", fp(max(nep)), max(nep), ST + ":dice_noempty/U,F,K/p (max)")
p52 = [J(ST, f"exact/dice|{g}/excl_within_source_52/p") for g in ["U", "F", "K"]]
put("p52.UFK.max", fp(max(p52)), max(p52), ST + ":exact/dice|U,F,K/excl_within_source_52/p (max)")
d52 = [J(ST, f"exact/dice|{g}/excl_within_source_52/diff") for g in ["U", "F", "K"]]
put("d52.UFK.range", ival(min(d52), max(d52)), [min(d52), max(d52)], ST + ":exact/dice|U,F,K/excl_within_source_52/diff")
# leave one dataset out, M19

# ------------------------------------------------------------------ stratified identity
for met in ["dice", "nsd"]:
    for g in ["U+F", "U+K", "U+M19"]:
        k = SETK[g]
        for q, lab in [("wound", "wound pairs"), ("mdiff", "modality_diff")]:
            num(f"s.{met}.{k}.{q}", ST, f"stratified/{met}|{g}/pearson_on_percentiles/{lab}")
            num(f"s.{met}.{k}.{q}.sp", ST, f"stratified/{met}|{g}/stratified_spearman/{lab}")
            num(f"s.{met}.{k}.{q}.sw", ST, f"stratified/{met}|{g}/size_weighted_average/{lab}")
            num(f"s.{met}.{k}.{q}.pool", ST, f"stratified/{met}|{g}/pooled_unstratified/{lab}")
mx = max(J(ST, f"stratified/dice|{g}/max_abs_pair_diff_stratified_vs_weighted") for g in ["U+F", "U+K", "U+M19"])
put("s.maxpair", f2(mx), mx, ST + ":stratified/dice|U+F,U+K,U+M19/max_abs_pair_diff_stratified_vs_weighted (max)")
mi = max(J(ST, f"stratified/{m}|{g}/max_abs_pair_diff_pearsonpct_vs_identity") for m in ["dice", "nsd"] for g in ["U+F", "U+K", "U+M19", "ALL"])
put("s.ident", f"{mi:.0e}".replace("e-", " × 10^-").replace("^-", "^−"), mi, ST + ":stratified/*/max_abs_pair_diff_pearsonpct_vs_identity (max)")

# ------------------------------------------------------------------ floor (group mean Dice)
for g in ["U", "F", "K", "M19", "N8s"]:
    mw = [J(ST, f"floor/mean_dice/{g}/{s}") for s in WOUND]
    mo = [J(ST, f"floor/mean_dice/{g}/{s}") for s in OTHER]
    put(f"md.{SETK[g]}.wound", ival(min(mw), max(mw)), [min(mw), max(mw)], ST + f":floor/mean_dice/{g}/<wound>")
    put(f"md.{SETK[g]}.other", ival(min(mo), max(mo)), [min(mo), max(mo)], ST + f":floor/mean_dice/{g}/<other 7>")
    for s in SETS12:
        num(f"md.{SETK[g]}.{SK[s]}", ST, f"floor/mean_dice/{g}/{s}")
mo = [J(ST, f"floor/mean_dice/M19/{s}") for s in OTHER]
put("md.M19.other.max", f2(max(mo)), max(mo), ST + ":floor/mean_dice/M19/<other 7> (max)")
colon = ["kvasir_disjoint", "kvasir_sessile", "cvc_clinicdb", "cvc_colondb", "etis_larib"]
for g in ["U", "K"]:
    mc = [J(ST, f"floor/mean_dice/{g}/{s}") for s in colon]
    put(f"md.{g}.colon", ival(min(mc), max(mc)), [min(mc), max(mc)], ST + f":floor/mean_dice/{g}/<colonoscopy>")

# ------------------------------------------------------------------ reliability
r_all = [J(ST, f"reliability/dice|ALL/{s}/spearman_brown_median") for s in WOUND + OTHER]
put("rel.ALL", ival(min(r_all), max(r_all)), [min(r_all), max(r_all)], ST + ":reliability/dice|ALL/<11 refs>/spearman_brown_median")
r_m = [J(ST, f"reliability/dice|M19/{s}/spearman_brown_median") for s in WOUND]
put("rel.M19.wound", ival(min(r_m), max(r_m)), [min(r_m), max(r_m)], ST + ":reliability/dice|M19/<wound>/spearman_brown_median")
r_m = [J(ST, f"reliability/dice|M19/{s}/spearman_brown_median") for s in OTHER]
put("rel.M19.other", ival(min(r_m), max(r_m)), [min(r_m), max(r_m)], ST + ":reliability/dice|M19/<other>/spearman_brown_median")

# ------------------------------------------------------------------ seeds
C = "seeds/dice|C16/summary"
for q, lab in [("rbar", "rbar_wound"), ("rbar.ctrl", "rbar_control"), ("rbar.colon", "rbar_colon"),
               ("single", "single|wound pairs"), ("avg", "avg|wound pairs"),
               ("avgminus", "avg_minus_single|wound pairs"), ("cross", "crossseed|wound pairs"),
               ("gap", "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"),
               ("s0", "seed0|wound pairs"), ("s1", "seed1|wound pairs"), ("s2", "seed2|wound pairs")]:
    num(f"sd.{q}", ST, f"{C}/{lab}/est")
    ci(f"sd.{q}.m", ST, f"{C}/{lab}/method_boot")
    ci(f"sd.{q}.h", ST, f"{C}/{lab}/hierarchical")
for g in ["P9", "N7"]:
    lab = "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"
    num(f"sd.gap.{g}", ST, f"seeds/dice|{g}/summary/{lab}/est")
    ci(f"sd.gap.{g}.m", ST, f"seeds/dice|{g}/summary/{lab}/method_boot")
    num(f"sd.rbar.{g}", ST, f"seeds/dice|{g}/summary/rbar_wound/est")
    num(f"sd.cross.{g}", ST, f"seeds/dice|{g}/summary/crossseed|wound pairs/est")
CN = "seeds/nsd|C16/summary"
for q, lab in [("rbar", "rbar_wound"), ("cross", "crossseed|wound pairs"),
               ("gap", "gap|wound: same-dataset cross-seed minus cross-dataset cross-seed"),
               ("single", "single|wound pairs"), ("avg", "avg|wound pairs"),
               ("avgminus", "avg_minus_single|wound pairs")]:
    num(f"sdn.{q}", ST, f"{CN}/{lab}/est")
    ci(f"sdn.{q}.m", ST, f"{CN}/{lab}/method_boot")
# per wound dataset r-bar (C16)
rb = []
for s in WOUND:
    pd_ = J(ST, f"seeds/dice|C16/per_dataset/{s}")
    rb.append(pd_["rbar"])
put("sd.rbar.range", ival(min(rb), max(rb)), [min(rb), max(rb)], ST + ":seeds/dice|C16/per_dataset/<wound>/rbar")
# V6 group M at each training seed
NG = "seeds/named_groups"
num("sdg.M.mdiff", ST, f"{NG}/dice|M (V6) = P10_s0 + nnunet_2d/modality_diff")
num("sdg.Ms1.mdiff", ST, f"{NG}/dice|M_s1 = P10_s1 + nnunet_2d/modality_diff")
num("sdg.Ms2.mdiff", ST, f"{NG}/dice|M_s2 = P10_s2 + nnunet_2d/modality_diff")
num("sdg.M.wound", ST, f"{NG}/dice|M (V6) = P10_s0 + nnunet_2d/wound pairs")
num("sdg.Ms1.wound", ST, f"{NG}/dice|M_s1 = P10_s1 + nnunet_2d/wound pairs")
num("sdg.Ms2.wound", ST, f"{NG}/dice|M_s2 = P10_s2 + nnunet_2d/wound pairs")
for s in [0, 1, 2]:
    num(f"sdg.M19s{s}.mdiff", ST, f"{NG}/dice|M19_s{s} = P10_s{s} + N8_s{s} + nnunet_2d/modality_diff")
    num(f"sdg.M19s{s}.wound", ST, f"{NG}/dice|M19_s{s} = P10_s{s} + N8_s{s} + nnunet_2d/wound pairs")
md19 = [J(ST, f"{NG}/dice|M19_s{s} = P10_s{s} + N8_s{s} + nnunet_2d/modality_diff") for s in [0, 1, 2]]
put("sdg.M19s.mdiff.range", ival(min(md19), max(md19)), [min(md19), max(md19)], ST + ":seeds/named_groups/M19_s0..2/modality_diff")
pval("p.Ms1", "v7/blind_v7.json", "exact_tests/M_s1/p")
pval("p.Ms2", "v7/blind_v7.json", "exact_tests/M_s2/p")
# spread of method means vs seed noise (Dice, wound datasets)
SP = "seeds/spread"
rng_, gap_, ssd, ratio_, icc1, icc3 = [], [], [], [], [], []
for s in WOUND:
    rng_.append(J(ST, f"{SP}/dice|{s}/M19_range"))
    gap_.append(J(ST, f"{SP}/dice|{s}/M19_median_adjacent_gap"))
    ssd.append(J(ST, f"{SP}/dice|{s}/C16_seed_sd_pooled"))
    ratio_.append(J(ST, f"{SP}/dice|{s}/ratio_between_sd_to_seed_sd"))
    icc1.append(J(ST, f"{SP}/dice|{s}/C16_icc_single"))
    icc3.append(J(ST, f"{SP}/dice|{s}/C16_icc_3seed_avg"))
src = ST + ":seeds/spread/dice|<wound>/"
put("sp.range", ival(min(rng_), max(rng_)), [min(rng_), max(rng_)], src + "M19_range")
put("sp.gap", ival(min(gap_), max(gap_), f3), [min(gap_), max(gap_)], src + "M19_median_adjacent_gap")
put("sp.seedsd", ival(min(ssd), max(ssd), f3), [min(ssd), max(ssd)], src + "C16_seed_sd_pooled")
put("sp.ratio", ival(min(ratio_), max(ratio_)), [min(ratio_), max(ratio_)], src + "ratio_between_sd_to_seed_sd")
put("sp.icc1", ival(min(icc1), max(icc1)), [min(icc1), max(icc1)], src + "C16_icc_single")
put("sp.icc3", ival(min(icc3), max(icc3)), [min(icc3), max(icc3)], src + "C16_icc_3seed_avg")
# disattenuation (C16, Dice, wound pairs, seed averages, additive rule)
DA = "seeds/disattenuation/dice|C16/categories/wound pairs (6)"
num("da.add.raw", ST, f"{DA}/dis_additive/raw_mean")
num("da.add.cap", ST, f"{DA}/dis_additive/capped_mean")

# ------------------------------------------------------------------ benchmark (bench_v7)
BSET = ["U", "F", "K", "M19", "U+F", "U+K", "U+M19", "M", "M11s", "N8s", "P10s"]
for g in BSET:
    k = SETK.get(g, g)
    for r in WOUND + ["mean"]:
        for q in ["bench", "other", "diff"]:
            base = f"b.{k}.{SK.get(r, r)}.{q}"
            num(base, BE, f"rel/{g}/40/{r}|{q}/est")
            put(base + ".m", ival(J(BE, f"rel/{g}/40/{r}|{q}/lo"), J(BE, f"rel/{g}/40/{r}|{q}/hi")),
                [J(BE, f"rel/{g}/40/{r}|{q}/lo"), J(BE, f"rel/{g}/40/{r}|{q}/hi")], BE + f":rel/{g}/40/{r}|{q}/lo,hi")
    num(f"b11.{k}.mean.bench", BE, f"rel/{g}/11/mean|bench/est")
for q in ["bench", "other", "diff"]:
    put(f"bh.M19.mean.{q}", ival(J(BE, f"rel_hier/M19/40/mean|{q}/lo"), J(BE, f"rel_hier/M19/40/mean|{q}/hi")),
        [J(BE, f"rel_hier/M19/40/mean|{q}/lo"), J(BE, f"rel_hier/M19/40/mean|{q}/hi")], BE + f":rel_hier/M19/40/mean|{q}")
for g in ["U", "F", "K", "M19", "U+F", "U+K", "U+M19", "M"]:
    k = SETK[g]
    vals = [J(BE, f"rel/{g}/40/{r}|bench/est") for r in WOUND]
    put(f"b.{k}.range", ival(min(vals), max(vals)), [min(vals), max(vals)], BE + f":rel/{g}/40/<wound>|bench/est")
# texture effect (amplitude 40 minus 11), mean over references
for g in ["U", "F", "K", "M19", "U+F", "U+K", "U+M19"]:
    k = SETK[g]
    num(f"tx.{k}", BE, f"texture/{g}/mean/est")
    put(f"tx.{k}.m", ival(J(BE, f"texture/{g}/mean/lo"), J(BE, f"texture/{g}/mean/hi")),
        [J(BE, f"texture/{g}/mean/lo"), J(BE, f"texture/{g}/mean/hi")], BE + f":texture/{g}/mean/lo,hi")
    for r in WOUND:
        num(f"tx.{k}.{SK[r]}", BE, f"texture/{g}/{r}/est")
        put(f"tx.{k}.{SK[r]}.m", ival(J(BE, f"texture/{g}/{r}/lo"), J(BE, f"texture/{g}/{r}/hi")),
            [J(BE, f"texture/{g}/{r}/lo"), J(BE, f"texture/{g}/{r}/hi")], BE + f":texture/{g}/{r}/lo,hi")
# count of within-group per-reference texture intervals excluding 0 (U, F, K, M19)
cnt, tot = 0, 0
for g in ["U", "F", "K", "M19"]:
    for r in WOUND:
        lo, hi = J(BE, f"texture/{g}/{r}/lo"), J(BE, f"texture/{g}/{r}/hi")
        tot += 1
        cnt += (lo > 0 or hi < 0)
put("tx.within.n", str(cnt), cnt, BE + ":texture/U,F,K,M19/<wound>/lo,hi (intervals excluding 0)")
put("tx.within.tot", str(tot), tot, BE + ":texture (number of within-group reference intervals)")
cnt = sum(1 for r in WOUND if J(BE, f"texture/U+M19/{r}/lo") > 0)
put("tx.UM19.n", str(cnt), cnt, BE + ":texture/U+M19/<wound>/lo > 0")
# single-seed groups
for nm, key in [("P10", "_P10_single_seed_bench_range"), ("N8", "_N8_single_seed_bench_range"), ("M19", "_M19_single_seed_bench_range")]:
    lo, hi = J(BE, f"seed_effect_amp40/{key}")
    put(f"bs.{nm}.range", ival(lo, hi), [lo, hi], BE + f":seed_effect_amp40/{key}")
for g in ["P10_s0", "P10_s1", "P10_s2", "M19_s0", "M19_s1", "M19_s2", "N8_s0", "N8_s1", "N8_s2"]:
    num(f"bs.{g}", BE, f"seed_effect_amp40/{g}/bench/est")
    num(f"bs.{g}.other", BE, f"seed_effect_amp40/{g}/other/est")
o_p10 = [J(BE, f"seed_effect_amp40/P10_s{s}/other/est") for s in [0, 1, 2]]
put("bs.P10.other.range", ival(min(o_p10), max(o_p10)), [min(o_p10), max(o_p10)], BE + ":seed_effect_amp40/P10_s0..2/other/est")
# leave one method out (M19)
lomo = J(BE, "lomo/M19")
lb = [v["bench40_mean"] for k, v in lomo.items() if k != "_all"]
assert len(lb) == 19
put("lomo.M19.bench", ival(min(lb), max(lb)), [min(lb), max(lb)], BE + ":lomo/M19/<dropped>/bench")

# ------------------------------------------------------------------ practical (practical_v7)
S19 = "selection/M19"
num("sel.off", PR, f"{S19}/offdiag_mean", f3)
num("sel.off.max", PR, f"{S19}/offdiag_max", f3)
num("sel.off.min", PR, f"{S19}/offdiag_min", f3)
num("sel.rand", PR, f"{S19}/random_mean_over_B", f3)
num("sel.syn", PR, f"{S19}/synthetic/amp40/mean_loss", f3)
for r in WOUND:
    num(f"sel.syn.{SK[r]}", PR, f"{S19}/synthetic/amp40/loss/{r}", f3)
    num(f"sel.rand.{SK[r]}", PR, f"{S19}/random_choice_expected_loss/{r}", f3)
    best = J(PR, f"{S19}/best_per_dataset/{r}/model")
    put(f"sel.best.{SK[r]}", disp(best), best, PR + f":{S19}/best_per_dataset/{r}/model")
    num(f"sel.best.{SK[r]}.dice", PR, f"{S19}/best_per_dataset/{r}/mean_dice")
pk = J(PR, f"{S19}/synthetic/amp40/chosen")
put("sel.syn.pick", disp(pk), pk, PR + f":{S19}/synthetic/amp40/chosen")
num("sel.syn.pick.dice", PR, f"{S19}/synthetic/amp40/chosen_synth_dice")
ru = J(PR, f"{S19}/synthetic/amp40/runner_up")
put("sel.syn.ru", disp(ru), ru, PR + f":{S19}/synthetic/amp40/runner_up")
num("sel.syn.ru.dice", PR, f"{S19}/synthetic/amp40/runner_up_synth_dice")
num("sel.syn.n01", PR, f"{S19}/synthetic/amp40/n_within_0.01", lambda x: str(x))
picks11 = {J(PR, f"{S19}/synthetic/amp{a}/chosen") for a in (11, 28, 40)}
assert len(picks11) == 1
for s in [0, 1, 2]:
    g = f"selection/M19_s{s}"
    pk = J(PR, f"{g}/synthetic/amp40/chosen")
    put(f"sel.s{s}.pick", disp(pk), pk, PR + f":{g}/synthetic/amp40/chosen")
    num(f"sel.s{s}.loss", PR, f"{g}/synthetic/amp40/mean_loss", f3)
    num(f"sel.s{s}.off", PR, f"{g}/offdiag_mean", f3)
    num(f"sel.s{s}.rand", PR, f"{g}/random_mean_over_B", f3)
    num(f"sel.s{s}.n02", PR, f"{g}/synthetic/amp40/n_within_0.02", lambda x: str(x))
num("sel.s0.pick.dice", PR, "selection/M19_s0/synthetic/amp40/chosen_synth_dice")
num("sel.s0.ru.dice", PR, "selection/M19_s0/synthetic/amp40/runner_up_synth_dice")
ru0 = J(PR, "selection/M19_s0/synthetic/amp40/runner_up")
put("sel.s0.ru", disp(ru0), ru0, PR + ":selection/M19_s0/synthetic/amp40/runner_up")
for ss in ["2024", "7", "101"]:
    pk = J(PR, f"selection/M19_s0/synthetic/amp40/chosen_per_synth_seed/{ss}")
    put(f"sel.s0.pick.syn{ss}", disp(pk), pk, PR + f":selection/M19_s0/synthetic/amp40/chosen_per_synth_seed/{ss}")
# EfficientNet-B3 U-Net synthetic Dice by training seed (amplitude 40)
sdb = J(PR, "synthetic_dice_by_training_seed/amp40")
for s_ in [0, 1, 2]:
    num(f"eff.s{s_}", PR, f"synthetic_dice_by_training_seed/amp40/unet_effb3/per_seed/{s_}")
num("eff.range", PR, "synthetic_dice_by_training_seed/amp40/unet_effb3/range")
# candidate-wise best per dataset changes with the seed
best_change = [r for r in WOUND if len({disp(J(PR, f"clinical_best_by_training_seed/{r}/M19_s{s}/model")) for s in [0, 1, 2]}) > 1]
put("sel.bestchange.n", ["none", "one", "two", "three", "four"][len(best_change)], len(best_change), PR + ":clinical_best_by_training_seed/<wound>/M19_s0..2/model (architecture differs)")
for s in [0, 1, 2]:
    for r in WOUND:
        b = J(PR, f"clinical_best_by_training_seed/{r}/M19_s{s}/model")
        put(f"sel.best.{SK[r]}.s{s}", disp(b), b, PR + f":clinical_best_by_training_seed/{r}/M19_s{s}/model")
# misses
MS = "miss/0.1"
for g, k in [("P18_archmean", "P18"), ("P10_archmean", "P10"), ("N8_archmean", "N8")]:
    for r in ["azh", "fuseg", "azh_train_slice"]:
        num(f"miss.{k}.{SK[r]}", PR, f"{MS}/{r}/groups/{g}/est", pct)
        ci(f"miss.{k}.{SK[r]}.ci", PR, f"{MS}/{r}/groups/{g}/ci", pct)
    ci(f"miss.{k}.azh.archrange", PR, f"{MS}/azh/groups/{g}/arch_mean_range", pct)
    ci(f"miss.{k}.azh.modelrange", PR, f"{MS}/azh/groups/{g}/single_model_range", pct)
    num(f"miss.{k}.azh.minusnn", PR, f"{MS}/azh/groups/{g}/minus_nnunet", pct)
    ci(f"miss.{k}.azh.minusnn.ci", PR, f"{MS}/azh/groups/{g}/minus_nnunet_ci", pct)
for r in ["azh", "fuseg", "azh_train_slice", "medetec", "co2wounds"]:
    num(f"miss.nn.{SK[r]}", PR, f"{MS}/{r}/per_architecture/nnunet_2d/mean", pct)
    ci(f"miss.nn.{SK[r]}.ci", PR, f"{MS}/{r}/per_architecture/nnunet_2d/ci_of_seed_mean", pct)
num("miss.b2.azh", PR, f"{MS}/azh/per_architecture/segformer_mitb2/mean", pct)
ci("miss.b2.azh.ci", PR, f"{MS}/azh/per_architecture/segformer_mitb2/ci_of_seed_mean", pct)
intval("miss.azh.nonempty", J(PR, f"{MS}/azh/n_nonempty"), PR + f":{MS}/azh/n_nonempty")
intval("miss.azh.units", J(PR, f"{MS}/azh/n_units_nonempty"), PR + f":{MS}/azh/n_units_nonempty")
intval("miss.nmodels", J(PR, "padding/azh/P18_all_seeds/n_models"), PR + ":padding/azh/P18_all_seeds/n_models")
# seed variation of an architecture's AZH miss rate (max over architectures of max - min)
pa = J(PR, f"{MS}/azh/per_architecture")
sv = {a: v["max"] - v["min"] for a, v in pa.items() if v["n_seeds"] > 1}
amax = max(sv, key=sv.get)
put("miss.seedvar.max", f1(100 * sv[amax]), sv[amax], PR + f":{MS}/azh/per_architecture/*/max-min")
put("miss.seedvar.arch", NAME[amax.replace('_in', '')] if amax.replace('_in', '') in NAME else disp(amax), amax, PR + f":{MS}/azh/per_architecture (argmax)")
PD = "padding/azh/P18_all_seeds"
num("pad.rho.black", PR, f"{PD}/spearman_black_share_vs_miss_share")
num("pad.rho.frac", PR, f"{PD}/spearman_frac_vs_miss_share")
num("pad.maj", PR, f"{PD}/n_majority_miss_nnunet_detect", lambda x: str(x))
num("pad.rev", PR, f"{PD}/n_nnunet_miss_majority_detect", lambda x: str(x))
for t in [0, 1, 2]:
    num(f"pad.t{t}.more", PR, f"{PD}/miss_by_frac_tercile_and_padding/{t}/miss_more_padding")
    num(f"pad.t{t}.less", PR, f"{PD}/miss_by_frac_tercile_and_padding/{t}/miss_less_padding")
num("pad.black.azh", PR, "padding/azh/black_share_mean_all_kept", pct)
num("pad.black.fus", PR, "padding/fuseg/black_share_mean_all_kept", pct)

# ------------------------------------------------------------------ robustness, modern sets (robust_v7)
num("thr.c50", RB, "threshold/common/M19|0.50/wound pairs/est")
ci("thr.c50.m", RB, "threshold/common/M19|0.50/wound pairs/ci")
num("thr.c50.d", RB, "threshold/delta_vs_default/M19|0.50|wound pairs/est")
ci("thr.c50.d.m", RB, "threshold/delta_vs_default/M19|0.50|wound pairs/ci")
num("thr.c90", RB, "threshold/common/M19|0.90/wound pairs/est")
num("thr.oracle", RB, "threshold/oracle/M19/wound pairs/est")
ci("thr.oracle.m", RB, "threshold/oracle/M19/wound pairs/ci")
num("thr.oracle.d", RB, "threshold/delta_vs_default/M19|oracle|wound pairs/est")
ci("thr.oracle.d.m", RB, "threshold/delta_vs_default/M19|oracle|wound pairs/ci")
num("thr.K.oracle", RB, "threshold/oracle/K11/wound pairs/est")
ab = J(RB, "common_threshold_wound_pairs_delta_ci_above0/M19")
put("thr.above.lo", f"{min(ab):.2f}", min(ab), RB + ":common_threshold_wound_pairs_delta_ci_above0/M19 (min)")
put("thr.above.hi", f"{max(ab):.2f}", max(ab), RB + ":common_threshold_wound_pairs_delta_ci_above0/M19 (max)")
cm = {k.split("|")[1]: v for k, v in J(RB, "threshold/common").items() if k.startswith("M19|")}
wp = {t: v["wound pairs"]["est"] for t, v in cm.items()}
tmax = max(wp, key=wp.get)
put("thr.max", f2(wp[tmax]), wp[tmax], RB + f":threshold/common/M19|{tmax}/wound pairs/est (max over grid)")
put("thr.max.t", tmax, tmax, RB + ":threshold/common (argmax)")
kw = {k.split("|")[1]: v["wound pairs"]["est"] for k, v in J(RB, "threshold/common").items() if k.startswith("K11|")}
put("thr.K.max", f2(max(kw.values())), max(kw.values()), RB + ":threshold/common/K11|*/wound pairs/est (max)")
num("cal.default", RB, "threshold/calibration_probe/M19|default/mean_rho_wound")
num("cal.c50", RB, "threshold/calibration_probe/M19|0.50/mean_rho_wound")
num("cal.c90", RB, "threshold/calibration_probe/M19|0.90/mean_rho_wound")
# M19 Dice wound-pair agreement over post-processing, resolution and oracle threshold
vals = [J(RB, f"postprocessing/summaries/M19|dice|{v}/wound pairs/est") for v in ("P0", "P1", "P2")]
vals.append(J(RB, "resolution/M19|dice/with_fuseg_r256/wound pairs/est"))
vals.append(J(RB, "threshold/oracle/M19/wound pairs/est"))
put("rob.M19.dice.range", ival(min(vals), max(vals)), [min(vals), max(vals)], RB + ":postprocessing P0-P2, resolution fuseg_r256, oracle threshold: M19 dice wound pairs")
nv = [J(RB, f"postprocessing/summaries/M19|nsd2|{v}/wound pairs/est") for v in ("P0", "P1", "P2")]
nv += [J(RB, f"tolerance/M19|{t}|P0/wound pairs/est") for t in ("nsd1", "nsd2", "nsd3", "nsd5", "nsd_diag1")]
nv.append(J(RB, "resolution/M19|nsd2/with_fuseg_r256/wound pairs/est"))
put("rob.M19.nsd.range", ival(min(nv), max(nv)), [min(nv), max(nv)], RB + ":postprocessing, tolerance, resolution: M19 nsd wound pairs")
lo, hi = J(RB, "modality_diff_all_conditions/est_range/M19")
put("rob.M19.mdiff.range", ival(lo, hi), [lo, hi], RB + ":modality_diff_all_conditions/est_range/M19")
num("rob.mdiff.ncond", RB, "modality_diff_all_conditions/n_conditions", lambda x: str(x))
num("rob.pp.minrho", RB, "postprocessing/min_rank_rho_on_nonfloor_sets/M19|dice/rho")
num("rob.pp.minrho.nsd", RB, "postprocessing/min_rank_rho_on_nonfloor_sets/M19|nsd2/rho")
num("res.M19.shift", RB, "resolution/M19|dice/shift_mean")
num("res.M19.nshift", RB, "resolution/M19|nsd2/shift_mean")
num("res.M19.rho", RB, "resolution/M19|dice/spearman_fuseg_vs_r256")
num("res.M19.nrho", RB, "resolution/M19|nsd2/spearman_fuseg_vs_r256")
for v in ("P1", "P2"):
    num(f"pp.{v}.d", RB, f"postprocessing/delta_vs_P0/M19|dice|{v}-P0|wound pairs/est")
    ci(f"pp.{v}.d.m", RB, f"postprocessing/delta_vs_P0/M19|dice|{v}-P0|wound pairs/ci")

# ------------------------------------------------------------------ robustness, pooled sets (robust_pooled_v7)
for g in ["U+M19", "U+F", "U+K11"]:
    k = {"U+M19": "UM19", "U+F": "UF", "U+K11": "UK11"}[g]
    c = f"composition_summary/{g}|wound"
    num(f"rp.{k}.holds", RP, f"{c}/n_holds", lambda x: str(x))
    num(f"rp.{k}.ncond", RP, f"{c}/n_conditions", lambda x: str(x))
    ci(f"rp.{k}.pooled", RP, f"{c}/range_pooled")
    ci(f"rp.{k}.gap", RP, f"{c}/range_pooled_minus_higher_within")
    ci(f"rp.{k}.vsother", RP, f"{c}/range_pooled_vs_other")
    ci(f"rp.{k}.strat", RP, f"{c}/range_stratified")
    num(f"rp.{k}.nC2", RP, f"{c}/n_C2", lambda x: str(x))
    lo, hi = J(RP, f"modality_diff_summary/{g}/est_range")
    put(f"rp.{k}.mdiff", ival(lo, hi), [lo, hi], RP + f":modality_diff_summary/{g}/est_range")
    num(f"rp.{k}.mdiff.nci", RP, f"modality_diff_summary/{g}/n_ci_above0", lambda x: str(x))
ci("rp.UM19.within.M19", RP, "composition_summary/U+M19|wound/range_within_M19")
ci("rp.UM19.within.U", RP, "composition_summary/U+M19|wound/range_within_U")
num("rp.UM19.strat.maxdev", RP, "composition_summary/U+M19|wound/max_abs_stratified_minus_size_weighted", f2)
for g, k in [("U+M19", "UM19"), ("U+F", "UF")]:
    base = f"results/{g}/dice|P0/composition/wound"
    num(f"rp.{k}.gap.P0", RP, f"{base}/pooled_minus_higher_within/est")
    ci(f"rp.{k}.gap.P0.m", RP, f"{base}/pooled_minus_higher_within/ci")
    num(f"rp.{k}.vso.P0", RP, f"{base}/pooled_vs_other/est")
    ci(f"rp.{k}.vso.P0.m", RP, f"{base}/pooled_vs_other/ci")
num("rp.UM19.changes", RP, "change_summary/U+M19/n_changes", lambda x: str(x))
num("rp.UM19.changes.sig", RP, "change_summary/U+M19/n_ci_excludes_0", lambda x: str(x))
num("rp.UK11.colon.holds", RP, "composition_summary/U+K11|colonoscopy/n_holds", lambda x: str(x))

# ------------------------------------------------------------------ nnU-Net re-prediction (audit)
NR = "check2_rescore_vs_scores/nnunet_detail"
num("nn.img.maxdiff", AU, f"{NR}/all/max_abs_dice", lambda x: f"{x:.4f}")
rk = J(AU, f"{NR}/rank_among_11_v6_modern")
md_ = max(v["abs_delta_dice"] for v in rk.values())
assert all(v["rank_dice_run1"] == v["rank_dice_run2"] and v["rank_nsd2_run1"] == v["rank_nsd2_run2"] for v in rk.values())
put("nn.mean.maxdiff", f"{md_:.5f}", md_, AU + f":{NR}/rank_among_11_v6_modern/*/abs_delta_dice (max)")


# ------------------------------------------------------------------ checks behind qualitative statements
for s_ in WOUND:
    assert J(ST, f"floor/mean_dice/M19/{s_}") > J(ST, f"floor/mean_dice/U/{s_}"), s_
for s_ in OTHER:
    assert J(ST, f"floor/mean_dice/M19/{s_}") < J(ST, f"floor/mean_dice/U/{s_}"), s_
nf = sum(J(ST, f"floor/mean_dice/F/{s_}") < 0.1 for s_ in OTHER)
put("md.F.nfloor", {6: "six", 5: "five", 7: "seven", 4: "four"}[nf], nf, ST + ":floor/mean_dice/F/<other> < 0.1 (count)")
assert all(J(ST, f"floor/mean_dice/M19/{s_}") < 0.1 for s_ in OTHER)
assert all(J(ST, f"floor/mean_dice/U/{s_}") >= 0.1 or s_ == "etis_larib" for s_ in OTHER)
assert F["tx.UM19.n"]["raw"] == 4
assert F["n.excl.ctrl.mininl"]["raw"] >= 25
put("miss.b2.name", disp("segformer_mitb2"), "segformer_mitb2", PR + ":miss/0.1/azh/per_architecture (lowest mean)")
pa_ = {a: v["mean"] for a, v in J(PR, "miss/0.1/azh/per_architecture").items() if a != "nnunet_2d"}
assert min(pa_, key=pa_.get) == "segformer_mitb2"
# the M19 synthetic pick is the same at amplitudes 11, 28 and 40, and for every synthetic seed
assert len({J(PR, f"selection/M19/synthetic/amp40/chosen_per_synth_seed/{x}") for x in ("2024", "7", "101")}) == 1
# within-group texture intervals: every U+M19 reference interval above 0 (checked above); U+F mean interval above 0
assert J(BE, "texture/U+F/mean/lo") > 0
# F and K benchmark intervals include 0
for g in ["F", "K"]:
    assert J(BE, f"rel/{g}/40/mean|bench/lo") < 0 < J(BE, f"rel/{g}/40/mean|bench/hi")
# M19 diff, U+M19 diff interval
assert J(BE, "rel/U+M19/40/mean|diff/hi") < 0
assert J(BE, "rel/M19/40/mean|diff/lo") < 0 < J(BE, "rel/M19/40/mean|diff/hi")
# gap intervals exclude 0; N7 gap interval includes 0
assert J(ST, "seeds/dice|C16/summary/gap|wound: same-dataset cross-seed minus cross-dataset cross-seed/hierarchical")[0] > 0
assert J(ST, "seeds/dice|N7/summary/gap|wound: same-dataset cross-seed minus cross-dataset cross-seed/method_boot")[0] < 0
assert J(ST, "seeds/dice|C16/summary/rbar_wound/est") < 0.7
assert J(ST, "bootstrap/dice|M19/wound pairs/hierarchical")[0] < 0
# no robustness change for M19 (except common thresholds) has an interval excluding 0
for k_, v_ in J(RB, "postprocessing/delta_vs_P0").items():
    if k_.startswith("M19") and "wound" in k_:
        assert v_["ci"][0] < 0 < v_["ci"][1], k_
for k_, v_ in J(RB, "tolerance_delta_vs_nsd2").items():
    if k_.startswith("M19|nsd") and "wound" in k_:
        assert v_["ci"][0] < 0 < v_["ci"][1], k_
lo_, hi_ = J(RB, "resolution/M19|dice/delta_r256_minus_native_wound_pairs/ci") if isinstance(J(RB, "resolution/M19|dice/delta_r256_minus_native_wound_pairs"), dict) else J(RB, "resolution/M19|dice/delta_r256_minus_native_wound_pairs")[1:]
assert lo_ < 0 < hi_
assert J(RB, "threshold/delta_vs_default/M19|oracle|wound pairs/ci")[0] < 0 < J(RB, "threshold/delta_vs_default/M19|oracle|wound pairs/ci")[1]

# ------------------------------------------------------------------ write
with open(OUT, "w") as f:
    json.dump(F, f, indent=1, ensure_ascii=False, sort_keys=True)
print(f"wrote {OUT}: {len(F)} facts")

#!/usr/bin/env python3
"""Build results/exclusions_v6.json from exclusions_v5.json and the overlap checks.

Rule (one rule for every evaluation set): an evaluation image is removed before
any analysis if it is a copy or a crop of a photograph that another part of the
study uses, that is
  (a) a FUSeg training image used to train a model (training indices 0 to 392)
      or to tune a threshold (400 to 549) (results/v7/fuseg_train_index.json), or
  (b) an image of another evaluation set: the FUSeg reference (the first 120
      FUSeg validation images, without those excluded as duplicates) or, for the
      control, the AZH test set.
Crops of FUSeg photographs that nothing in the study uses are kept: they are
not leakage and not shared between evaluation sets.

Inputs
  results/exclusions_v5.json               8 FUSeg duplicates, 47 control crops (kept as they are)
  results/v7/azh_overlap_v7.json           SIFT check of AZH test and control against all
                                           1,010 FUSeg photographs (frames tied at the best
                                           inlier count are all considered)
  results/v7/azh_control_overlap_v7.json   AZH test against control: pixel copies and crops
Output: results/exclusions_v6.json (same layout: set -> [[index, file, reason], ...]) and
        results/v7/exclusions_v6_summary.json (counts by reason, for the manuscript)
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")


def J(*p):
    return json.load(open(os.path.join(RES, *p)))


V5 = J("exclusions_v5.json")
OV = J("v7", "azh_overlap_v7.json")
CO = J("v7", "azh_control_overlap_v7.json")
IDX = J("v7", "fuseg_train_index.json")
POS = {f: i for i, f in enumerate(IDX["files"])}
# FUSeg reference images actually analysed: the first 120 validation images minus
# the ones excluded as duplicates (exclusions_v5.json, kept unchanged below)
REF = [os.path.basename(p) for i, p in enumerate(J("perimage32", "fuseg.json")["image_paths"])
       if i not in {e[0] for e in V5["fuseg"]}]
IDS = {s: J("perimage32", s + ".json")["ids"] for s in ("azh", "azh_train_slice")}


def use_of(frame):
    """How the study uses FUSeg frame 'split/file': train, threshold, reference or None."""
    split, f = frame.split("/")
    if split == "train":
        i = POS.get(f)
        if i is None or i in IDX["eval_duplicates"]:
            return None
        if 0 <= i <= 392:
            return "train"
        if 400 <= i <= 549:
            return "threshold"
        return None
    return "reference" if f in REF else None


def classify(m):
    frames = m.get("fuseg_tied") or [m["fuseg"]]
    used = [(f, use_of(f)) for f in frames if use_of(f)]
    return frames, used


def reason(m, frames, used):
    f, u = used[0]
    split, fn = f.split("/")
    r = f"crop of FUSeg {split} {fn} ({m['inliers']} SIFT inliers"
    other = [x for x in frames if x != f]
    if other:
        r += "; equally matched to " + ", ".join(o.replace("/", " ") for o in other)
    return r + ")"


out = {"fuseg": V5["fuseg"]}
summ = {"rule": __doc__.split("Inputs")[0].strip(), "fuseg": {"n": len(V5["fuseg"])}}

# AZH test
azh, cnt = [], {"train": 0, "threshold": 0, "reference": 0}
unused = 0
for k, m in sorted(OV["azh_test"]["matches"].items(), key=lambda kv: IDS["azh"].index(kv[0])):
    frames, used = classify(m)
    if used:
        azh.append([IDS["azh"].index(k), k + ".png", reason(m, frames, used)])
        cnt[used[0][1]] += 1
    else:
        unused += 1
out["azh"] = azh
summ["azh"] = {"n_checked": OV["azh_test"]["n"], "n_match": OV["azh_test"]["n_match"],
               "n_match_train_split": OV["azh_test"]["n_match_train"],
               "n_match_validation_split": OV["azh_test"]["n_match_validation"],
               "n_excluded": len(azh), "excluded_by_use": cnt, "n_kept_crops_of_unused_frames": unused,
               "n_match_validation_in_reference": sum(use_of(m["fuseg"]) == "reference"
                                                      for m in OV["azh_test"]["matches"].values()),
               "max_inliers_nonmatch": OV["azh_test"]["max_inliers_nonmatch"]}

# control: the recorded exclusions are re-checked with the same rule (the overlap
# check found all of them again); an image whose matched photographs are used
# nowhere in the study is kept
ctrl, v5_dropped = {}, []
for e in V5["azh_train_slice"]:
    k = e[1][:-4]
    frames, used = classify(OV["validation_control"]["matches"][k])
    if used:
        ctrl[e[0]] = e
    else:
        v5_dropped.append({"image": e[1], "v5_reason": e[2], "matched_frames": frames})
new_crop, new_crop_use, ctrl_unused = [], {"train": 0, "threshold": 0, "reference": 0}, 0
for k in OV["validation_control"]["new_hits_not_recorded"]:
    m = OV["validation_control"]["matches"][k]
    frames, used = classify(m)
    i = IDS["azh_train_slice"].index(k)
    if used:
        ctrl[i] = [i, k + ".png", reason(m, frames, used)]
        new_crop.append(k)
        new_crop_use[used[0][1]] += 1
    else:
        ctrl_unused += 1
dup = []
for t, cs in sorted(CO["pixel_identical"].items()):
    for c in cs:
        i = IDS["azh_train_slice"].index(c)
        if i not in ctrl:
            ctrl[i] = [i, c + ".png", f"pixel-identical to AZH test image {t}.png"]
            dup.append(c)
crop_t = sorted({x["control"] for v in CO["crop_matches"].values() for x in v} - {c for v in CO["pixel_identical"].values() for c in v})
for c in crop_t:
    i = IDS["azh_train_slice"].index(c)
    if i not in ctrl:
        t = [k for k, v in CO["crop_matches"].items() if any(x["control"] == c for x in v)][0]
        ctrl[i] = [i, c + ".png", f"crop of AZH test image {t}.png"]
        dup.append(c)
out["azh_train_slice"] = [ctrl[i] for i in sorted(ctrl)]
summ["azh_train_slice"] = {
    "n_checked": len(IDS["azh_train_slice"]), "n_v5": len(V5["azh_train_slice"]),
    "n_v5_kept_by_rule": len(v5_dropped), "v5_kept_by_rule": v5_dropped,
    "n_v5_confirmed": len(V5["azh_train_slice"]) - len(v5_dropped),
    "v5_found_again": OV["validation_control"]["found_again"],
    "n_new_crops_excluded": len(new_crop), "new_crops_by_use": new_crop_use,
    "n_new_crops_kept_unused_frames": ctrl_unused,
    "n_copies_of_azh_test_added": len(dup),
    "n_control_images_matching_azh_test": CO["n_control_images_matched"],
    "n_excluded": len(out["azh_train_slice"]),
    "n_kept": len(IDS["azh_train_slice"]) - len(out["azh_train_slice"]),
    "v5_by_use": {u: sum(1 for e in V5["azh_train_slice"] if e[0] in ctrl and
                         classify(OV["validation_control"]["matches"][e[1][:-4]])[1][0][1] == u)
                  for u in ("train", "threshold", "reference")},
}
summ["azh"]["n_kept"] = len(IDS["azh"]) - len(azh)
assert sum(summ["azh_train_slice"]["v5_by_use"].values()) == len(V5["azh_train_slice"]) - len(v5_dropped)
json.dump(out, open(os.path.join(RES, "exclusions_v6.json"), "w"), indent=1)
json.dump(summ, open(os.path.join(RES, "v7", "exclusions_v6_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))

#!/usr/bin/env python3
"""Re-run the design-property ablation on the chroma-corrected generator B.

Every earlier ablation number was produced with the uncorrected chroma texture
(roughly six times clinical). This re-runs the same one-factor-at-a-time table
on the corrected generator so the two are comparable.
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.generator_b import build as build_b            # noqa: E402
from woundbench.methods import METHOD_NAMES                     # noqa: E402
from woundbench.metrics import spearman_permutation             # noqa: E402
from rank_validity import OUT, score_benchmark                  # noqa: E402

ORDER = ["none", "grade_independent_size", "off_centre_placement", "clutter",
         "irregular_shape", "textured_background"]
OLD = {"none": None, "grade_independent_size": -0.154,
       "off_centre_placement": -0.130, "clutter": -0.091,
       "irregular_shape": -0.045, "textured_background": -0.316}


def main():
    reg = pickle.load(open(os.path.join(OUT, "models.pkl"), "rb"))["registry"]
    names = METHOD_NAMES
    c = json.load(open(os.path.join(OUT, "clinical_ranking_cache.json")))
    med, fus = c["medetec"], c["fuseg"]

    print("Ablation on the CHROMA-CORRECTED generator B", flush=True)
    hdr = ("ablated", "rho_med", "rho_fus", "mean", "change", "was")
    print("{:<26}{:>9}{:>9}{:>8}{:>9}{:>9}".format(*hdr))
    print("-" * 70)

    base = None
    out = {}
    for ab in ORDER:
        imgs, masks, _ = build_b(n_per_class=15, ablation=ab)
        s, _ = score_benchmark("x", imgs, masks, reg, names, log=None)
        v = [s[n]["mean_dice"] for n in names]
        rm = spearman_permutation(v, med, n_perm=8000)[0]
        rf = spearman_permutation(v, fus, n_perm=8000)[0]
        m = (rm + rf) / 2
        if base is None:
            base = m
            ch_s, was_s = "", ""
            change = None
        else:
            change = m - base
            ch_s = "{:+.3f}".format(change)
            was_s = "{:+.3f}".format(OLD[ab])
        out[ab] = {"rho_medetec": rm, "rho_fuseg": rf, "mean_rho": m,
                   "change": change, "change_uncorrected": OLD[ab]}
        print("{:<26}{:>9.3f}{:>9.3f}{:>8.3f}{:>9}{:>9}".format(
            ab, rm, rf, m, ch_s, was_s), flush=True)

    changes = {k: v["change"] for k, v in out.items() if v["change"] is not None}
    tex = changes["textured_background"]
    others = {k: v for k, v in changes.items() if k != "textured_background"}
    biggest_other = max(others, key=lambda k: abs(others[k]))
    print("\ntexture removal: {:+.3f}".format(tex))
    print("largest other effect: {} at {:+.3f}".format(
        biggest_other, others[biggest_other]))
    print("ratio: {:.1f}x  (uncorrected was 2.1x, manuscript reported 6.7x)".format(
        abs(tex) / max(abs(others[biggest_other]), 1e-9)))

    json.dump(out, open(os.path.join(OUT, "ablation_fixed.json"), "w"), indent=2)
    print("\nsaved " + os.path.join(OUT, "ablation_fixed.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does FRAMING explain why Medetec and FUSeg prefer opposite contrast regimes?

In the contrast sweep, agreement with Medetec FALLS as lesion-to-surround
contrast rises (0.573 -> 0.434) while agreement with FUSeg RISES (0.238 ->
0.504). The two cancel, which is why the mean is flat.

Hypothesis: it is framing, not contrast preference per se. Medetec is cropped
around the lesion (median lesion fraction 0.098); FUSeg is full-limb (0.0068), a
14-fold difference the manuscript itself flags.

Test: crop FUSeg around its own lesions to Medetec's framing, rebuild it as a
third clinical reference, and re-correlate the SAME cached generator-B contrast
levels against it. If cropped-FUSeg now behaves like Medetec, framing explains
the split and "clinical realism" has to be stated per framing convention.

Reuses the per-method Dice already cached in results/dose_response.json, so only
the cropped clinical set needs scoring.

Writes results/framing_test.json.
"""

import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from woundbench.methods import METHOD_NAMES                     # noqa: E402
from woundbench.metrics import spearman_permutation             # noqa: E402
from rank_validity import DATA, OUT, load_pairs, score_benchmark  # noqa: E402

TARGET_FRACTION = 0.098      # Medetec's median lesion fraction
OUT_SIZE = 224               # Medetec's resolution


def crop_to_fraction(img, mask, target_frac=TARGET_FRACTION, out=OUT_SIZE):
    """Square crop centred on the lesion, sized so lesion area / crop area
    equals the target, then resized to Medetec's resolution."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None, None
    area = float(len(ys))
    side = int(round(np.sqrt(area / target_frac)))
    h, w = mask.shape
    side = max(32, min(side, min(h, w)))
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    ic = img[y0:y0 + side, x0:x0 + side]
    mc = mask[y0:y0 + side, x0:x0 + side]
    if mc.sum() < 20:
        return None, None
    return (cv2.resize(ic, (out, out)),
            cv2.resize(mc, (out, out), interpolation=cv2.INTER_NEAREST))


def main():
    with open(os.path.join(OUT, "models.pkl"), "rb") as f:
        reg = pickle.load(f)["registry"]
    names = METHOD_NAMES

    dr = json.load(open(os.path.join(OUT, "dose_response.json")))
    contrast = dr["contrast_sweep"]
    med_v = dr["clinical"]["medetec"]
    fus_v = dr["clinical"]["fuseg"]

    print("loading FUSeg and cropping to Medetec framing", flush=True)
    fu_i, fu_m = load_pairs(
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "images"),
        os.path.join(DATA, "Foot Ulcer Segmentation Challenge", "validation",
                     "labels"), 0, 120)
    ci, cm = [], []
    for im, ms in zip(fu_i, fu_m):
        a, b = crop_to_fraction(im, ms)
        if a is not None:
            ci.append(a)
            cm.append(b)
    frac_before = float(np.median([m.mean() for m in fu_m]))
    frac_after = float(np.median([m.mean() for m in cm]))
    print(f"  {len(ci)} images; median lesion fraction "
          f"{frac_before:.4f} -> {frac_after:.4f} "
          f"(Medetec is {TARGET_FRACTION})", flush=True)

    print("scoring 22 methods on cropped FUSeg", flush=True)
    summ, _ = score_benchmark("fuseg_cropped", ci, cm, reg, names, log=None)
    cropped_v = [summ[n]["mean_dice"] for n in names]

    print(f"\n  cropped-FUSeg vs Medetec  rho "
          f"{spearman_permutation(cropped_v, med_v)[0]:+.3f}")
    print(f"  cropped-FUSeg vs FUSeg    rho "
          f"{spearman_permutation(cropped_v, fus_v)[0]:+.3f}")

    print("\n" + "=" * 78)
    print("CONTRAST SWEEP re-correlated against the cropped reference")
    print("=" * 78)
    print(f"{'sat delta':>11}{'vs Medetec':>13}{'vs FUSeg':>11}"
          f"{'vs FUSeg-cropped':>19}")
    rows = []
    for r in contrast:
        v = [r["per_method_dice"][n] for n in names]
        rc = spearman_permutation(v, cropped_v, n_perm=20000)[0]
        rows.append({"measured_sat_delta": r["measured_sat_delta"],
                     "rho_medetec": r["rho_medetec"],
                     "rho_fuseg": r["rho_fuseg"],
                     "rho_fuseg_cropped": rc})
        print(f"{r['measured_sat_delta']:>11.1f}{r['rho_medetec']:>13.3f}"
              f"{r['rho_fuseg']:>11.3f}{rc:>19.3f}")

    x = [r["measured_sat_delta"] for r in rows]
    tm = spearman_permutation(x, [r["rho_medetec"] for r in rows])
    tf = spearman_permutation(x, [r["rho_fuseg"] for r in rows])
    tc = spearman_permutation(x, [r["rho_fuseg_cropped"] for r in rows])
    print(f"\ntrend of agreement against contrast:")
    print(f"  vs Medetec         Spearman {tm[0]:+.3f}  (p {tm[1]:.4f})")
    print(f"  vs FUSeg           Spearman {tf[0]:+.3f}  (p {tf[1]:.4f})")
    print(f"  vs FUSeg-cropped   Spearman {tc[0]:+.3f}  (p {tc[1]:.4f})")

    flipped = (tf[0] > 0) and (tc[0] < 0)
    print(f"\n  FUSeg trend was {'positive' if tf[0] > 0 else 'negative'}; "
          f"after cropping it is {'positive' if tc[0] > 0 else 'negative'}")
    print(f"  Medetec trend is {'positive' if tm[0] > 0 else 'negative'}")
    if flipped:
        print("\n  VERDICT: cropping FUSeg to Medetec's framing REVERSES its")
        print("  contrast preference. Framing, not contrast, drives the split.")
    else:
        print("\n  VERDICT: cropping does not reverse the preference. Framing")
        print("  alone does not explain the split.")

    with open(os.path.join(OUT, "framing_test.json"), "w") as fh:
        json.dump({"rows": rows,
                   "lesion_fraction_before": frac_before,
                   "lesion_fraction_after": frac_after,
                   "n_cropped": len(ci),
                   "cropped_per_method_dice": {n: summ[n]["mean_dice"]
                                               for n in names},
                   "trends": {"medetec": {"rho": tm[0], "p": tm[1]},
                              "fuseg": {"rho": tf[0], "p": tf[1]},
                              "fuseg_cropped": {"rho": tc[0], "p": tc[1]}},
                   "reversed": bool(flipped)}, fh, indent=2)
    print(f"\nsaved {os.path.join(OUT, 'framing_test.json')}")


if __name__ == "__main__":
    main()

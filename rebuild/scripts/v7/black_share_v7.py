#!/usr/bin/env python3
"""Per-image black-pixel share and image size for the wound references and the
control, computed from the raw images with the perimage32 loaders (the same
pixels the Kaggle kit refs/<set>/NNNN_img.png hold).

black share = share of pixels with every channel <= 8 (as practical_v7).
All images in perimage32 order, before exclusions (exclusions are applied by
the reader). CO2Wounds is not in the public wound-segmentation repository and
is not computed when data/extra/co2wounds is absent; its set means (it has no
exclusions, so they cannot change) are then carried from practical_v7.json at
commit cf270c0, where they were computed from the Kaggle kit refs.

Output: results/v7/black_share_v7.json
"""
import json
import subprocess
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import perimage32 as P                                            # noqa: E402

PER = os.path.join(ROOT, "results", "perimage32")
CARRY = "cf270c0"
SETS = ["medetec", "co2wounds", "fuseg", "azh", "azh_train_slice"]


def main():
    loaders = dict(P.LOADERS)
    loaders["extra"] = P.make_extra_loader(127)
    out = {"definition": __doc__.strip(), "sets": {}}
    for s in SETS:
        d = json.load(open(os.path.join(PER, s + ".json")))
        if not os.path.exists(os.path.join(ROOT, d["image_paths"][0])):
            assert s not in json.load(open(os.path.join(ROOT, "results", "exclusions_v6.json"))), s
            old = json.loads(subprocess.run(["git", "show", f"{CARRY}:rebuild/results/v7/practical_v7.json"],
                                            cwd=ROOT, capture_output=True, check=True).stdout)["padding"][s]
            out["sets"][s] = {"carried_from": f"practical_v7.json at commit {CARRY}",
                              "black_share_mean_all_kept": old["black_share_mean_all_kept"],
                              "black_share_mean_nonempty": old["black_share_mean_nonempty"],
                              "n_kept": old["n_kept"]}
            print(f"{s}: raw images absent, set means carried from {CARRY}")
            continue
        bs, size = [], []
        for k, (ip, mp) in enumerate(zip(d["image_paths"], d["mask_paths"])):
            im, m = loaders[d["loader"]](os.path.join(ROOT, ip), os.path.join(ROOT, mp))
            assert abs(float(m.mean()) - d["gt_fraction"][k]) < 1e-9, (s, k)
            bs.append(float((im.max(2) <= 8).mean()))
            size.append(list(im.shape[:2]))
        out["sets"][s] = {"ids": d["ids"], "black_share": bs, "size": size}
        print(f"{s}: {len(bs)} images, mean black share {np.mean(bs):.4f}")
    json.dump(out, open(os.path.join(ROOT, "results", "v7", "black_share_v7.json"), "w"), indent=0)


if __name__ == "__main__":
    main()

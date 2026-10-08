#!/usr/bin/env python3
"""Every pair of references, Spearman agreement by pool, for Appendix A.

Writes ../manuscript_v4/appendix_pairs.json sorted by level, then by the
all-32 agreement, high to low.
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hierarchy_v4 as H                                          # noqa: E402
from fig_v4 import LABEL                                          # noqa: E402
from woundbench.combined import ALL_METHOD_FAMILY                 # noqa: E402

LABEL = dict(LABEL, azh_train_slice="AZH train slice")


def main():
    data, names = H.load()
    tr = lambda n: (ALL_METHOD_FAMILY[n] in ("learned", "CNN")
                    or ALL_METHOD_FAMILY[n].startswith("deep"))
    T = np.array([i for i, n in enumerate(names) if tr(n)])
    U = np.array([i for i, n in enumerate(names) if not tr(n)])
    pairs = list(itertools.combinations(list(H.REFS), 2)) + [H.CONTROL]
    out = []
    for metric in ("dice", "nsd"):
        vec = {k: H.means(data[k], metric) for k in data}
        rows = []
        for a, b in pairs:
            va, vb = vec[a], vec[b]
            rows.append({"metric": metric, "a": LABEL[a], "b": LABEL[b],
                         "level": H.level_of(a, b),
                         "all": float(spearmanr(va, vb).statistic),
                         "untrained": float(spearmanr(va[U], vb[U]).statistic),
                         "trained": float(spearmanr(va[T], vb[T]).statistic)})
        rows.sort(key=lambda r: (H.LEVELS.index(r["level"]), -r["all"]))
        out += rows
    p = os.path.join(ROOT, "..", "manuscript_v4", "appendix_pairs.json")
    json.dump(out, open(p, "w"), indent=0)
    print(f"{len(out)} rows -> {p}")


if __name__ == "__main__":
    main()

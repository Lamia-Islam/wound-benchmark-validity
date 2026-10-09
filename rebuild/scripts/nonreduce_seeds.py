#!/usr/bin/env python3
"""Score the 32 local methods on the Table 8 benchmarks at seeds 7 and 101.

Reads the synthetic PNGs exported for the Kaggle kit (identical to the
generator output, checked by fingerprint) and appends per-method mean Dice to
results/nonreduce32_seeds.jsonl. Resumable.
"""
import glob, json, os, pickle, sys
import cv2
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from woundbench import deep as D                                  # noqa: E402
from woundbench.combined import CombinedRegistry                  # noqa: E402
from woundbench.metrics import dice                               # noqa: E402

SYN = sys.argv[1]
OUT = os.path.join(ROOT, "results", "nonreduce32_seeds.jsonl")
SETS = [f"{b}_s{s}" for s in (7, 101) for b in ("nrA", "nrB0", "nrC0", "nrC34")]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["set"] for l in open(OUT) if l.strip()}
todo = [s for s in SETS if s not in done]
if todo:
    import torch
    torch.set_num_threads(2)
    classical = pickle.load(open(os.path.join(ROOT, "results", "models.pkl"), "rb"))["registry"]
    zoo = D.DeepZoo(os.path.join(ROOT, "results", "deep")).load()
    thr = json.load(open(os.path.join(ROOT, "results", "deep_thresholds.json")))
    reg = CombinedRegistry(classical, zoo, thr)
    names = reg.available()
    assert len(names) == 32
    for s in todo:
        files = sorted(glob.glob(os.path.join(SYN, s, "*_img.png")))
        acc = {n: [] for n in names}
        for f in files:
            im = cv2.imread(f)
            gt = (cv2.imread(f.replace("_img", "_mask"), 0) > 0).astype(np.uint8)
            out = reg.run_all(im, names=names)
            for n in names:
                acc[n].append(dice(out[n], gt))
        open(OUT, "a").write(json.dumps({"set": s, "dice": {n: float(np.mean(acc[n])) for n in names}}) + "\n")
        print(s, len(files), flush=True)

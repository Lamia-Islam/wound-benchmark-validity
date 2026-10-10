#!/usr/bin/env python3
"""Phase-2 GPU run: entry point. python k2_run.py STAGE [STAGE ...]

Stages, in the intended order:
  restore    copy run-1 checkpoints, thresholds and the nnU-Net model into OUT
  woundseg   prepare the fifth wound reference (WoundSeg, Hugging Face)
  seeds      original ten pretrained architectures at seeds 1 and 2
  modern2    eight more modern architectures at seeds 0, 1 and 2
  rescore    post-processing, tolerance, resolution and threshold-grid variants
  sam2       MedSAM with official preprocessing; SAM/MedSAM with nnU-Net boxes
  pack       zip the results into wb_results2.zip next to OUT

Every stage is resumable and contained: a failure is logged with its traceback
and the next stage still runs. A DONE_<stage> file marks a clean finish.
"""
import os
import sys
import time
import traceback
import zipfile

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import k2_common as C                                             # noqa: E402


def stage_restore():
    C.restore_prev()


def stage_woundseg():
    import k2_woundseg
    k2_woundseg.stage_woundseg()


def stage_seeds():
    import k2_train
    k2_train.stage_seeds()


def stage_modern2():
    import k2_train
    k2_train.stage_modern2()


def stage_rescore():
    import k2_rescore
    k2_rescore.stage_rescore()


def stage_sam2():
    import k2_sam
    k2_sam.stage_sam2()


KEEP_DIRS = ("scores", "rescore", "woundseg")
SKIP_IN_WOUNDSEG = ("raw",)


def stage_pack():
    """Zip scores/, rescore/, woundseg/ (selected images, masks, meta; not the
    raw download) and every top-level json/txt/DONE file. Checkpoints, the
    nnU-Net work folders, regenerated synthetic images and extra_sets are left
    out because they are large and reproducible."""
    out = C.OUT.rstrip("/")
    dst = os.path.join(os.path.dirname(out), "wb_results2.zip")
    tmp = dst + ".tmp"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(out):
            p = os.path.join(out, f)
            if os.path.isfile(p) and (f.endswith((".json", ".txt")) or f.startswith("DONE_")):
                z.write(p, f); n += 1
        for d in KEEP_DIRS:
            base = os.path.join(out, d)
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                if d == "woundseg":
                    dirs[:] = [x for x in dirs if x not in SKIP_IN_WOUNDSEG]
                for f in files:
                    if f.endswith(".tmp"):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, out)); n += 1
    os.replace(tmp, dst)
    C.log(f"  wrote {dst}: {n} files, {os.path.getsize(dst) / 1e6:.1f} MB")


STAGES = {"restore": stage_restore, "woundseg": stage_woundseg,
          "seeds": stage_seeds, "modern2": stage_modern2,
          "rescore": stage_rescore, "sam2": stage_sam2, "pack": stage_pack}
ORDER = ["restore", "woundseg", "seeds", "modern2", "rescore", "sam2", "pack"]

if __name__ == "__main__":
    for st in sys.argv[1:] or ORDER:
        C.log(f"stage {st} on {C.DEV}{' (DRY)' if C.DRY else ''}")
        t0 = time.time()
        try:
            STAGES[st]()
            C.log(f"stage {st} done in {time.time() - t0:.0f}s")
            open(os.path.join(C.OUT, f"DONE_{st}"), "w").write(str(time.time() - t0))
        except Exception:
            C.log(f"stage {st} FAILED:\n{traceback.format_exc()}")

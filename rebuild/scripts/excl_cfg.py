"""Which exclusion list the analysis uses, and where its results go.

WB_EXCL selects the list (default exclusions_v6.json, the submission analysis).
WB_EXCL=exclusions_v5.json reruns the same code under the V6 manuscript's
exclusions; results then go to results/v7_excl_v5/ so they never overwrite the
submission results. Every check that a value equals V6 or the earlier V7
implementation is run in that mode (the old values were computed under the old
exclusions), and the two folders together give the old-versus-new comparison.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
EXCL_NAME = os.environ.get("WB_EXCL", "exclusions_v6.json")
assert EXCL_NAME in ("exclusions_v5.json", "exclusions_v6.json"), EXCL_NAME
EXCL_PATH = os.path.join(RES, EXCL_NAME)
EXCL = json.load(open(EXCL_PATH))
OLD = EXCL_NAME == "exclusions_v5.json"
V7OUT = os.path.join(RES, "v7_excl_v5" if OLD else "v7")
V7OUT_REL = "results/" + ("v7_excl_v5" if OLD else "v7")
V7IN = os.path.join(RES, "v7")          # exclusion-independent inputs (rescore_local, black shares, overlap check)

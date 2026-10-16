#!/usr/bin/env python3
"""Check that the current code, run under the V6 exclusions
(WB_EXCL=exclusions_v5.json, outputs in results/v7_excl_v5), reproduces the
results committed for manuscript V6 (git commit REF, results/v7).

Every numeric and text leaf of stats, bench, practical, robust, robust_pooled,
blind and audit is compared (blind: reported values at their printed decimals and item labels,
checked inside blind_v7.py). Leaves that record the run itself (runtime,
exclusion name, check bookkeeping added with the exclusion switch, the texture
source, file paths) are listed and skipped.

Output: results/v7/compare_excl_v7.json
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = "cf270c0"
FILES = ["stats_v7.json", "bench_v7.json", "practical_v7.json", "robust_v7.json",
         "robust_pooled_v7.json", "blind_v7.json", "audit/audit_v7.json"]
SKIP_KEYS = {"runtime_s", "exclusions", "enforced", "note", "first_differing_leaves", "texture_source",
             "engine_vs_stats_v7", "crosscheck_old_v7_implementation", "crosscheck_old_v7",
             "check_vs_earlier_v7_attempt", "definitions", "elapsed_s"}


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP_KEYS:
                continue
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}/{i}")
    else:
        yield p, x


def main():
    out = {"reference_commit": REF, "skipped_keys": sorted(SKIP_KEYS), "files": {}}
    ok = True
    for f in FILES:
        old = json.loads(subprocess.run(["git", "show", f"{REF}:rebuild/results/v7/{f}"], cwd=ROOT,
                                        capture_output=True, check=True).stdout)
        new = json.load(open(os.path.join(ROOT, "results", "v7_excl_v5", f)))
        lo, ln = dict(leaves(old)), dict(leaves(new))
        if f == "blind_v7.json":
            # V6 typed the reported values at their printed decimals; they are now read
            # unrounded from the result files, and blind_v7.py asserts in this mode that
            # both agree at those decimals. Compare everything else (replicated values,
            # PASS/FAIL, labels) exactly.
            for d in (lo, ln):
                for k in [k for k in d if k.endswith(("/reported", "/abs_diff", "/quantity"))]:
                    del d[k]
        common = set(lo) & set(ln)
        diff = sorted(k for k in common if lo[k] != ln[k] and not (
            isinstance(lo[k], float) and isinstance(ln[k], float) and lo[k] != lo[k] and ln[k] != ln[k]))
        r = {"n_old": len(lo), "n_new": len(ln), "n_common": len(common), "n_identical": len(common) - len(diff),
             "differing": diff[:30], "only_old": sorted(set(lo) - set(ln))[:30],
             "only_new": sorted(set(ln) - set(lo))[:30]}
        out["files"][f] = r
        ok &= not diff and not r["only_old"]
        print(f, {k: v for k, v in r.items() if k.startswith("n_")}, "differing:", diff[:5], "only_old:", r["only_old"][:5])
    out["all_identical"] = bool(ok)
    json.dump(out, open(os.path.join(ROOT, "results", "v7", "compare_excl_v7.json"), "w"), indent=1)
    print("all identical:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

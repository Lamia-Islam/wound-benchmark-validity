#!/usr/bin/env bash
# Full V7 pipeline under the submission exclusions (exclusions_v6.json), then the
# check that the same code reproduces V6 under the V6 exclusions.
# Prerequisites: raw data in rebuild/data/wound-segmentation (public), and for the
# audit a partial Kaggle kit: python3 scripts/v7/kit_partial_v7.py data/kaggle_kit
# The SIFT overlap checks (azh_overlap_v7.py, about 3 min on 4 CPUs, and
# azh_control_overlap_v7.py) are inputs to exclusions_v6.py and are not rerun here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export WB_KIT="${WB_KIT:-$PWD/data/kaggle_kit}"
run() { echo "== $*"; nice -n 10 python3 "$@" > "/tmp/run_all_$(basename "$1" .py).log" 2>&1 || { tail -20 "/tmp/run_all_$(basename "$1" .py).log"; exit 1; }; }
run scripts/v7/exclusions_v6.py
for s in stats_v7 bench_v7 practical_v7 write_md_v7 robust_v7 robust_pooled_v7 blind_v7 audit_v7 review_v7 fig_v6 online_resources_v6 facts_v6; do
  run scripts/v7/$s.py
done
grep -E "^PASS" /tmp/run_all_blind_v7.log
(cd ../manuscript_v6 && node build_v6.js | head -2 && node build_v6.js --unblinded | head -1)
if [ "${WB_CHECK_V6:-1}" = "1" ]; then
  export WB_EXCL=exclusions_v5.json
  for s in stats_v7 bench_v7 practical_v7 write_md_v7 robust_v7 robust_pooled_v7 blind_v7 audit_v7; do
    run scripts/v7/$s.py
  done
  python3 scripts/v7/compare_excl_v7.py | tail -1
fi

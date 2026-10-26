#!/bin/bash
# P3-0a end to end: frame cache (lmee env, Habitat) -> model queries (onecanvas env) -> report + docs/p3_0_decision.md.
# Usage (repo root): bash analysis/p3_0_run_all.sh          # every stage, skips finished work (both stages resume)
#                    REPORT_ONLY=1 bash analysis/p3_0_run_all.sh   # just rescore from analysis/out_p3_0/*.jsonl (CPU, seconds)
set -e
cd "$(dirname "$0")/.."
LMEE_PY=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
OC_PY=${OC_PY:-/home/quang/miniconda3/envs/onecanvas/bin/python}
LOG=analysis/runs/p3_0
mkdir -p $LOG
RUNS="p2_E0_fix_s77 p2_E0_fix_s78 p2_E0_fix_s79"
if [ -z "$REPORT_ONLY" ]; then
  i=0
  for r in $RUNS; do CUDA_VISIBLE_DEVICES=$i $LMEE_PY analysis/p3_0_cache.py --run $r > $LOG/cache_$r.log 2>&1 & i=$((i+1)); done
  wait
  $OC_PY rcmem/test_onecanvas_conversion.py > analysis/out_p3_0/conversion_test.txt 2>&1
  q() { for r in $2; do CUDA_VISIBLE_DEVICES=$1 $OC_PY analysis/p3_0a.py --run $r --model $3 >> $LOG/${3}_$r.log 2>&1; done; }
  q 0 "$RUNS" qwen25 &
  q 1 "$RUNS" qwen3 &
  q 2 "p2_E0_fix_s77 p2_E0_fix_s78" onecanvas &
  (q 3 "p2_E0_fix_s79" onecanvas; q 3 "$RUNS" qwen3_B) &
  wait
  CUDA_VISIBLE_DEVICES=0 $OC_PY analysis/p3_0_oc_sanity.py > analysis/out_p3_0/onecanvas_sanity.txt 2>/dev/null
fi
$OC_PY analysis/p3_0_report.py

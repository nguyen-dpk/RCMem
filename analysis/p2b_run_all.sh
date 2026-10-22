#!/bin/bash
# Phase 2b (DEV_PLAN.md) in one command, from the repo root:   bash analysis/p2b_run_all.sh
# Default: rebuild the tables and docs/phase2b_decision.md from saved outputs (CPU, ~1 min).
# RERUN=1 also re-runs the GPU stages through the queue: F0 (depth re-rendered at the recorded poses) and F1 / F2 / F2b on
# the 58-task subset (two scene chunks each; any GPU with >= 22 GB free is used; GPUS="0 1 2 3" by default).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
if [ "${RERUN:-0}" = "1" ]; then
  {
    echo "f0"
    for x in F1 F2 F2b; do
      echo "$x p2b_${x}_s77 0.0 0.5 --seed 77"
      echo "$x p2b_${x}_s77 0.5 1.0 --seed 77"
    done
  } > analysis/runs/p2b_jobs.txt
  analysis/runs/p2b_queue.sh ${GPUS:-0 1 2 3}
fi
(cd analysis && $P p2b_report.py)   # -> analysis/out_p2b/p2b_results.json, docs/phase2b_decision.md

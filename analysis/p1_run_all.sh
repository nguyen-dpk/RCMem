#!/bin/bash
# Phase 1 (DEV_PLAN.md) in one command, from the repo root:   bash analysis/p1_run_all.sh
# Default: rebuild the tables from saved runs (CPU, ~1 min). RERUN=1 also re-runs every experiment through the
# GPU queue (E0 x2 more, E1, E2, E4, E5 on the 58-task subset; ~30 GPU-hours; any GPU with >= 22 GB free is used).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
if [ "${RERUN:-0}" = "1" ]; then
  printf '%s\n' "p1 E0 p1_E0_r2 0.0 0.5" "p1 E0 p1_E0_r2 0.5 1.0" "p1 E4 p1_E4 0.0 0.5" "p1 E4 p1_E4 0.5 1.0" \
    "p1 E1 p1_E1 0.0 0.5" "p1 E1 p1_E1 0.5 1.0" "p1 E2 p1_E2 0.0 0.5" "p1 E2 p1_E2 0.5 1.0" \
    "p1 E0 p1_E0_r3 0.0 0.5" "p1 E0 p1_E0_r3 0.5 1.0" "p1 E5 p1_E5 0.0 0.5" "p1 E5 p1_E5 0.5 1.0" > analysis/runs/p1_jobs.txt
  analysis/runs/p1_queue.sh ${GPUS:-0 1 2 3}
fi
cd analysis && $P p1_report.py      # -> analysis/out_p1/p1_results.json, analysis/out_p1/tables.md

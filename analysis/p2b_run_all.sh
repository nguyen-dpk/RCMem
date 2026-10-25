#!/bin/bash
# Phase 2b (DEV_PLAN.md) in one command, from the repo root:   bash analysis/p2b_run_all.sh
# Default: rebuild the tables and docs/phase2b_decision.md from saved outputs (CPU, ~2 min).
# RERUN=1 also re-runs the GPU stages through the queue: F0 (depth re-rendered at the recorded poses) and F1 / F2 / F2b at
# seeds 77-79 with the empty-context rule (p2b_ec_*; two scene chunks each; GPUS="0 1 2 3" by default, a GPU is used once it
# has >= 22 GB free, "a+b" splits one job over a pair). The fixed baseline p2_E0_fix_s77-79 comes from Phase 2.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
if [ "${RERUN:-0}" = "1" ]; then
  {
    echo "f0"
    for sd in 77 78 79; do
      for x in F2 F1 F2b; do
        echo "$x p2b_ec_${x}_s$sd 0.0 0.5 --seed $sd"
        echo "$x p2b_ec_${x}_s$sd 0.5 1.0 --seed $sd"
      done
    done
  } > analysis/runs/p2b_jobs.txt
  analysis/runs/p2b_queue.sh ${GPUS:-0 1 2 3}
fi
(cd analysis && $P p2b_report.py)   # -> analysis/out_p2b/p2b_results.json, docs/phase2b_decision.md

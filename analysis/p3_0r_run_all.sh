#!/bin/bash
# P3-0 re-check (DEV_PLAN.md), step 3 in one command, from the repo root:   bash analysis/p3_0r_run_all.sh
# Default: rescore and regenerate docs/p3_0_decision.md from saved outputs (CPU). RERUN=1 also runs the navigation jobs
# through the Phase 2b GPU queue (analysis/runs/p2b_queue.sh): PA (A1-band panoramas), PB (region markers) and F1, all
# on base Qwen3-VL-8B (the backbone the re-check chose), seeds 77-79, two scene chunks each; a GPU is used once it has
# >= 22 GB free (GPUS="0 1 2 3" by default). Steps 1-2 (offline backbone control): bash analysis/p3_0_run_all.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
VLM=${VLM:-Qwen/Qwen3-VL-8B-Instruct}
if [ "${RERUN:-0}" = "1" ]; then
  for sd in 77 78 79; do
    for x in PA PB F1; do
      echo "$x p3r_${x}_s$sd 0.0 0.5 --seed $sd --vlm $VLM"
      echo "$x p3r_${x}_s$sd 0.5 1.0 --seed $sd --vlm $VLM"
    done
  done > analysis/runs/p3r_jobs.txt
  QUEUE=analysis/runs/p3r_jobs.txt WORKER=analysis/p3_0_nav.py analysis/runs/p2b_queue.sh ${GPUS:-0 1 2 3}
fi
/home/quang/miniconda3/envs/onecanvas/bin/python analysis/p3_0_report.py   # -> p3_0_results.json, docs/p3_0_decision.md

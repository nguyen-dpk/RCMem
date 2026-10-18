#!/bin/bash
# Reproduce every number in docs/gap_report.md.  Run from the repo root:   bash analysis/run_all.sh
# Uses existing outputs when present; set RERUN=1 to also redo the GPU stages (instrumented re-run ~2.5 h on
# 4x24 GB GPUs; GT question views ~100 min on one GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
LMEE_PY=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}                                                   # env with habitat-sim + LMEE deps
ARROW_PY=${ARROW_PY:-/home/quang/miniconda3/envs/memoryexplorer/bin/python}  # any env with pandas + pyarrow
GPU=${GPU:-0}
CFG=$PWD/analysis/cfg/instr_choice_sub.yaml

if [ "${RERUN:-0}" = "1" ]; then
  # 1. instrumented re-run of the baseline on the 58-task subset (5 scene splits; resumable, retries on OOM)
  for r in "0.0 0.25" "0.25 0.5" "0.5 0.625" "0.625 0.75" "0.75 1.0"; do
    analysis/runs/run_split.sh "$GPU" "$CFG" $r
  done
  # 2. recover the ground-truth view each QA was generated from
  CUDA_VISIBLE_DEVICES=$GPU $LMEE_PY analysis/gt_question_views.py
fi

$LMEE_PY analysis/check_baseline_facts.py           # baseline facts (retrieval size, step cap, start pose, QA letters)
$ARROW_PY analysis/check_train_memory.py            # training-memory construction (hand-inserted goal frames)
$LMEE_PY analysis/analyze.py > analysis/out/analyze_stdout.json   # taxonomy + diagnostics a-e -> analysis/out/
for px in 200 8192; do                              # visibility-threshold sensitivity
  MIN_PX=$px $LMEE_PY analysis/analyze.py > /dev/null && cp analysis/out/summary.json analysis/out/summary_minpx$px.json
done
$LMEE_PY analysis/analyze.py > /dev/null            # restore the default (MIN_PX=1638) outputs
echo "done: analysis/out/summary.json, nav_items.csv, qa_items.csv"

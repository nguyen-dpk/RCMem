#!/bin/bash
# P3 (DEV_PLAN.md) in one command, from the repo root:   bash analysis/p3_run_all.sh
# CPU stages always run: region-builder offline accuracy, parser accuracy, pilot sample, then docs/p3_report.md.
# PILOT=1 runs the pilot (RP, --viz, analysis/cfg/p3_pilot.yaml) on GPUs 0-3 and writes its qualitative report.
# RERUN=1 runs the subset runs (seed 77, base Qwen3-VL-8B) through the Phase 2b GPU queue (analysis/runs/p2b_queue.sh):
#   RP (RCMem, predicted regions), RG (RCMem, GT regions), the RP variants RPin / RPpt / RPcu, and RP on Qwen2.5-VL-7B-Instruct
#   (F2b and MemoryExplorer's pipeline on Qwen3 were dropped by the user's decisions; p3_nav.py --exp F2b / E0 still runs them);
#   F1 on Qwen3-VL-8B is P3-0's p3r_F1_s77 (bash analysis/p3_0r_run_all.sh). REPORT_ONLY=1 skips the CPU stages too.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
VLM=${VLM:-Qwen/Qwen3-VL-8B-Instruct}
if [ "${REPORT_ONLY:-0}" != "1" ]; then
  $P analysis/p3_regions.py                # -> analysis/out_p3/regions_offline.json
  $P rcmem/instruction.py                  # -> analysis/out_p3/parser_eval.json
  $P analysis/p3_pilot.py                  # -> analysis/out_p3/pilot.json, analysis/cfg/p3_pilot.yaml
fi
if [ "${PILOT:-0}" = "1" ]; then
  rm -rf analysis/runs/p3_pilot_RP
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${PILOT_GPUS:-0,1,2,3} \
    P2B_VLM_MAXMEM=${PILOT_MAXMEM:-0:3GiB,1:7GiB,2:8GiB,3:7GiB} \
    $P analysis/p3_nav.py --exp RP --vlm $VLM --run_name p3_pilot_RP --cfg_file $PWD/analysis/cfg/p3_pilot.yaml --seed 77 --viz
  $P analysis/p3_qual.py --run p3_pilot_RP   # -> docs/qual/p3_RP_<date>.html
fi
if [ "${RERUN:-0}" = "1" ]; then
  for x in "RP p3_RP_s77 $VLM" "RG p3_RG_s77 $VLM" "RPin p3_RPin_s77 $VLM" "RPpt p3_RPpt_s77 $VLM" "RPcu p3_RPcu_s77 $VLM" \
           "RP p3_RP25_s77 Qwen/Qwen2.5-VL-7B-Instruct"; do
    set -- $x
    echo "$1 $2 0.0 0.5 --seed 77 --vlm $3"
    echo "$1 $2 0.5 1.0 --seed 77 --vlm $3"
  done > analysis/runs/p3_jobs.txt
  QUEUE=analysis/runs/p3_jobs.txt WORKER=analysis/p3_nav.py analysis/runs/p2b_queue.sh ${GPUS:-0+1 2+3}
fi
$P analysis/p3_report.py                   # -> analysis/out_p3/p3_results.json, docs/p3_report.md

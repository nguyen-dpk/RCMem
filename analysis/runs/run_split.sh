#!/bin/bash
# usage: run_split.sh <CUDA_VISIBLE_DEVICES> <cfg> <start> <end>; retries after OOM, resuming per episode
cd /home/quang/nguyendpk/LMEE
for i in 1 2 3 4 5 6; do
  /home/quang/miniconda3/envs/lmee/bin/python analysis/clean_partial.py $3 $4
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 /home/quang/miniconda3/envs/lmee/bin/python analysis/instrumented_run.py --cfg_file $2 --start_ratio $3 --end_ratio $4 --split 1 >> analysis/runs/stdout_split_$3.txt 2>&1 && grep -q "All scenes finish" analysis/runs/instr_choice_sub/log_$(printf %.2f $3)_$(printf %.2f $4)_1.log && break
done

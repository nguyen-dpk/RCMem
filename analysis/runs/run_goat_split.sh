#!/bin/bash
# usage: run_goat_split.sh <CUDA_VISIBLE_DEVICES> <start> <end>; GOAT episode 0 of every scene, retries after OOM
cd /home/quang/nguyendpk/LMEE
for i in 1 2 3 4 5 6; do
  /home/quang/miniconda3/envs/lmee/bin/python analysis/clean_partial.py $2 $3 goat_ep0
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 /home/quang/miniconda3/envs/lmee/bin/python analysis/goat_runner.py --episodes 0 --start_ratio $2 --end_ratio $3 >> analysis/runs/stdout_goat_$2.txt 2>&1 && grep -q "All scenes finish" analysis/runs/goat_ep0/log_$(printf %.2f $2)_$(printf %.2f $3)_1.log && break
done

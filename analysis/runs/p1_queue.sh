#!/bin/bash
# Experiment job queue (Phases 1-2). Each line of analysis/runs/p1_jobs.txt:
#   "<kind> <exp> <run_name> <start> <end> [extra p1_experiments.py args, e.g. --fix_start --seed 78 --data full]"
#   "cmd <script> <args>" runs a one-off analysis script on the GPU (kind = p1 | goat | cmd).
# One worker per GPU (args: GPU ids); a worker takes the next job only when its GPU has >= 22 GB free; each job
# resumes per episode and retries up to 6 times after OOM. Usage: nohup analysis/runs/p1_queue.sh 0 1 2 3 &
cd /home/quang/nguyendpk/LMEE
P=/home/quang/miniconda3/envs/lmee/bin/python
Q=analysis/runs/p1_jobs.txt
pop() { flock analysis/runs/p1_jobs.lock bash -c "head -n1 $Q; sed -i 1d $Q"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
worker() {
  g=$1
  while true; do
    [ -s $Q ] || break
    if [ "$(free_mb $g)" -lt 22000 ]; then sleep 120; continue; fi
    job=$(pop); [ -z "$job" ] && break
    read kind exp run s e extra <<< "$job"
    echo "$(date) GPU$g start $job" >> analysis/runs/p1_queue.log
    if [ "$kind" = cmd ]; then  # "cmd <python script and args>": run once with the lmee interpreter on this GPU
      CUDA_VISIBLE_DEVICES=$g $P ${job#cmd } >> analysis/runs/stdout_cmd_$(echo ${job#cmd } | tr -c 'A-Za-z0-9_\n' _ | cut -c1-80).txt 2>&1 \
        && echo "$(date) GPU$g done $job" >> analysis/runs/p1_queue.log || echo "$(date) GPU$g FAILED $job" >> analysis/runs/p1_queue.log
      continue
    fi
    for i in 1 2 3 4 5 6; do
      if [ "$kind" = goat ]; then
        $P analysis/clean_partial.py $s $e goat_ep0
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $P analysis/goat_runner.py --episodes 0 --start_ratio $s --end_ratio $e >> analysis/runs/stdout_goat_$s.txt 2>&1
        log=analysis/runs/goat_ep0/log_$(printf %.2f $s)_$(printf %.2f $e)_1.log
      else
        $P analysis/clean_partial.py $s $e $run
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $P analysis/p1_experiments.py --exp $exp --run_name $run --start_ratio $s --end_ratio $e $extra >> analysis/runs/stdout_${run}_$s.txt 2>&1
        log=analysis/runs/$run/log_$(printf %.2f $s)_$(printf %.2f $e)_1.log
      fi
      grep -q "All scenes finish" $log && break
      echo "$(date) GPU$g retry $i $job" >> analysis/runs/p1_queue.log
      sleep 180
    done
    grep -q "All scenes finish" $log && echo "$(date) GPU$g done $job" >> analysis/runs/p1_queue.log || echo "$(date) GPU$g FAILED $job" >> analysis/runs/p1_queue.log
  done
}
for g in "$@"; do worker $g & sleep 5; done
wait

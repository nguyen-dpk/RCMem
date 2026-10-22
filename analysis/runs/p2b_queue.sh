#!/bin/bash
# Phase 2b GPU job queue. Each line of analysis/runs/p2b_jobs.txt:
#   "<exp> <run_name> <start> <end> [extra p2b_experiments.py args]"   (exp = F1 | F2 | F2b)
#   "f0" runs analysis/p2b_f0.py
# One worker per GPU (args: GPU ids); a worker takes the next job only when its GPU has >= 22 GB free (the GPUs are
# shared); each job resumes per episode and retries up to 6 times. Usage: nohup analysis/runs/p2b_queue.sh 0 1 2 3 &
cd "$(dirname "$0")/../.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
Q=analysis/runs/p2b_jobs.txt
L=analysis/runs/p2b_logs
mkdir -p $L
pop() { flock analysis/runs/p2b_jobs.lock bash -c "head -n1 $Q; sed -i 1d $Q"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
worker() {
  g=$1
  while true; do
    [ -s $Q ] || break
    if [ "$(free_mb $g)" -lt 22000 ]; then sleep 120; continue; fi
    job=$(pop); [ -z "$job" ] && break
    read exp run s e extra <<< "$job"
    echo "$(date) GPU$g start $job" >> $L/queue.log
    if [ "$exp" = f0 ]; then
      CUDA_VISIBLE_DEVICES=$g $P analysis/p2b_f0.py >> $L/f0.log 2>&1 \
        && echo "$(date) GPU$g done $job" >> $L/queue.log || echo "$(date) GPU$g FAILED $job" >> $L/queue.log
      continue
    fi
    log=analysis/runs/$run/log_$(printf %.2f $s)_$(printf %.2f $e)_1.log
    for i in 1 2 3 4 5 6; do
      $P analysis/clean_partial.py $s $e $run
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $P analysis/p2b_experiments.py --exp $exp \
        --run_name $run --start_ratio $s --end_ratio $e $extra >> $L/stdout_${run}_$s.txt 2>&1
      grep -q "All scenes finish" $log && break
      echo "$(date) GPU$g retry $i $job" >> $L/queue.log
      sleep 180
    done
    grep -q "All scenes finish" $log && echo "$(date) GPU$g done $job" >> $L/queue.log || echo "$(date) GPU$g FAILED $job" >> $L/queue.log
  done
}
for g in "$@"; do worker $g & sleep 5; done
wait

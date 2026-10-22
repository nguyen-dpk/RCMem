#!/bin/bash
# Phase 2b GPU job queue. Each line of analysis/runs/p2b_jobs.txt:
#   "<exp> <run_name> <start> <end> [extra p2b_experiments.py args]"   (exp = F1 | F2 | F2b)
#   "f0" runs analysis/p2b_f0.py
# One worker per GPU (args: GPU ids); a worker takes the next job once its GPU has >= 22 GB free (the GPUs are shared);
# a failed attempt is requeued at the front (any GPU may resume it, per episode), up to 6 attempts per job. Usage: nohup analysis/runs/p2b_queue.sh 0 1 2 3 &
cd "$(dirname "$0")/../.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
Q=analysis/runs/p2b_jobs.txt
L=analysis/runs/p2b_logs
mkdir -p $L
pop() { flock analysis/runs/p2b_jobs.lock bash -c "head -n1 $Q; sed -i 1d $Q"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
# the GPUs are shared: wait until the GPU has >= 22 GB free twice, 30 s apart (lets a job that just started elsewhere allocate)
wait_free() { while true; do
  if [ "$(free_mb $1)" -ge 22000 ]; then sleep 30; [ "$(free_mb $1)" -ge 22000 ] && return; fi; sleep 20; done; }
worker() {
  g=$1
  while true; do
    [ -s $Q ] || break
    wait_free $g
    job=$(pop); [ -z "$job" ] && break
    read exp run s e extra <<< "$job"
    echo "$(date) GPU$g start $job" >> $L/queue.log
    if [ "$exp" = f0 ]; then
      CUDA_VISIBLE_DEVICES=$g $P analysis/p2b_f0.py >> $L/f0.log 2>&1 \
        && echo "$(date) GPU$g done $job" >> $L/queue.log || echo "$(date) GPU$g FAILED $job" >> $L/queue.log
      continue
    fi
    log=analysis/runs/$run/log_$(printf %.2f $s)_$(printf %.2f $e)_1.log
    $P analysis/clean_partial.py $s $e $run
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $P analysis/p2b_experiments.py --exp $exp \
      --run_name $run --start_ratio $s --end_ratio $e $extra >> $L/stdout_${run}_$s.txt 2>&1
    if grep -q "All scenes finish" $log; then echo "$(date) GPU$g done $job" >> $L/queue.log; continue; fi
    # failed (usually another process took the GPU): put the job back at the front so any GPU can resume it
    n=$(( $(cat $L/attempts_${run}_$s 2>/dev/null || echo 0) + 1 )); echo $n > $L/attempts_${run}_$s
    if [ $n -lt 6 ]; then
      flock analysis/runs/p2b_jobs.lock sed -i "1i $job" $Q
      echo "$(date) GPU$g retry $n (requeued) $job" >> $L/queue.log
      sleep 60
    else
      echo "$(date) GPU$g FAILED $job" >> $L/queue.log
    fi
  done
}
for g in "$@"; do worker $g & sleep 5; done
wait

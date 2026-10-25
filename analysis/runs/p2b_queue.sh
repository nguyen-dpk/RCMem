#!/bin/bash
# Phase 2b GPU job queue. Each line of analysis/runs/p2b_jobs.txt:
#   "<exp> <run_name> <start> <end> [extra p2b_experiments.py args]"   (exp = F1 | F2 | F2b)
#   "f0" (first line only) runs analysis/p2b_f0.py
# One worker per GPU spec (args): "g" = one GPU with >= 22 GB free; "a+b" = a GPU pair with >= 14 GB free on each (the
# VLM is split over the pair, perception runs on a). The GPUs are shared, so a worker waits until its GPUs are free
# enough, then starts one `p2b_experiments.py --worker` process that loads the model once and runs queued jobs back to
# back, keeping the GPUs until the queue is empty; a failed job is requeued at the front (resumes per episode, max 6
# attempts).  Usage: nohup analysis/runs/p2b_queue.sh 0 1+2 3 &
cd "$(dirname "$0")/../.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
Q=analysis/runs/p2b_jobs.txt
L=analysis/runs/p2b_logs
mkdir -p $L
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
enough() {  # enough <spec>: are the GPUs of the spec free enough now?
  if [[ $1 == *+* ]]; then [ "$(free_mb ${1%+*})" -ge 14000 ] && [ "$(free_mb ${1#*+})" -ge 14000 ]
  else [ "$(free_mb $1)" -ge 22000 ]; fi; }
# wait until the spec is free enough twice, 30 s apart (lets a job that just started elsewhere allocate)
wait_free() { while true; do if enough $1; then sleep 30; enough $1 && return; fi; sleep 20; done; }
worker() {
  g=$1
  while [ -s $Q ]; do
    wait_free $g
    if [ "$(head -n1 $Q)" = f0 ]; then
      flock analysis/runs/p2b_jobs.lock sed -i 1d $Q
      echo "$(date) GPU$g start f0" >> $L/queue.log
      CUDA_VISIBLE_DEVICES=${g%+*} $P analysis/p2b_f0.py >> $L/f0.log 2>&1 \
        && echo "$(date) GPU$g done f0" >> $L/queue.log || echo "$(date) GPU$g FAILED f0" >> $L/queue.log
      continue
    fi
    if [[ $g == *+* ]]; then
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${g/+/,} P2B_VLM_MAXMEM="0:6GiB,1:14GiB" \
        $P analysis/p2b_experiments.py --worker $Q >> $L/worker_gpu$g.txt 2>&1
    else
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $P analysis/p2b_experiments.py --worker $Q \
        >> $L/worker_gpu$g.txt 2>&1
    fi
  done
}
for g in "$@"; do worker $g & sleep 5; done
wait

#!/bin/bash
# Phase 2 (DEV_PLAN.md) in one command, from the repo root:   bash analysis/p2_run_all.sh
# Default: rebuild everything from saved runs (CPU, a few minutes). RERUN=1 also re-runs the GPU jobs through the queue:
# baseline with/without the start fix (3 seeds), E1/E1b on the fixed baseline (3 seeds), the 166-task baseline, and the
# QA choice-permutation replays (~95 GPU-hours; any GPU with >= 22 GB free is used).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${LMEE_PY:-/home/quang/miniconda3/envs/lmee/bin/python}
if [ "${RERUN:-0}" = "1" ]; then
  $P - <<'PY'
jobs = []
def add(exp, run, extra, chunks=((0.0, 0.5), (0.5, 1.0))):
    jobs.extend(f"p1 {exp} {run} {s} {e} {extra}" for s, e in chunks)
for sd in (77, 78, 79): add("E0", f"p2_E0_fix_s{sd}", f"--fix_start --seed {sd}")
for sd in (78, 79): add("E0", f"p2_E0_nofix_s{sd}", f"--seed {sd}")
for sd in (77, 78, 79): add("E1", f"p2_E1_fix_s{sd}", f"--fix_start --seed {sd}")
for sd in (77, 78, 79): add("E1b", f"p2_E1b_fix_s{sd}", f"--fix_start --seed {sd}")
add("E0", "p2_full_E0_fix_s77", "--fix_start --seed 77 --data full", ((0.0, .25), (.25, .5), (.5, .75), (.75, 1.0)))
jobs += [f"cmd analysis/p2_qa_permute.py --run {r} --perms 3" for r in ("instr_choice_sub", "p2_E0_fix_s77", "p2_E0_fix_s78", "p2_E0_fix_s79")]
open("analysis/runs/p1_jobs.txt", "w").write("\n".join(jobs) + "\n")
PY
  analysis/runs/p1_queue.sh ${GPUS:-0 1 2 3}
fi
python3 analysis/room_types.py                         # region room types for E1b -> analysis/out_p2/room_types.json
(cd analysis && $P p2_verifier.py > /dev/null)         # E4 verifier vs ground truth -> analysis/out_p2/verifier.json
(cd analysis && $P p2_report.py)                       # -> analysis/out_p2/p2_results.json, docs/phase2_report.md

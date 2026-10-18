"""Numbers backing the 'verified baseline facts' section, computed from the user's baseline run (read-only)
and the benchmark files.  Run (lmee env, repo root): python analysis/check_baseline_facts.py
"""
import collections
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
from src.utils import get_pts_angle_goatbench  # noqa: E402
from src.habitat import get_quaternion  # noqa: E402

BASE = os.path.join(REPO, "evaluation/results/exp_eval_lmee_choice")
BENCH = os.path.join(REPO, "data/LMEE-Bench/lmee_bench_sub")
log = open(os.path.join(BASE, "log_0.00_1.00_1.log")).read()

# 1. retrieval returns the union of text top-3 and image top-3 -> up to 6 memories
nf = json.load(open(os.path.join(BASE, "n_filtered_snapshots.json")))
v = np.array(list(nf.values()))
pre = [(int(a), int(b)) for a, b in re.findall(r"Prefiltering memory: (\d+) -> (\d+)", log)]
print("[retrieval] memories passed to the MLLM per decision (logged 'Prefiltering memory a -> b'):",
      f"n={len(pre)}, max={max(b for _, b in pre)}, share>3={np.mean([b > 3 for _, b in pre]):.3f},",
      f"mean={np.mean([b for _, b in pre]):.2f}")
print("[retrieval] n_filtered_snapshots.json: mean", round(v.mean(), 2), "max", v.max())

# 2. step budget per subtask = max(sqrt(scene area) * 2, 50)
ms = [int(x) for x in re.findall(r"Scene max step: (\d+)", log)]
print("[step cap] 'Scene max step' over episodes:", dict(collections.Counter(ms)), "| share == 50:", round(np.mean([m == 50 for m in ms]), 3))

# 3. start orientation: LMEE start rotations parse to NaN -> identity rotation for every episode
angles = []
for f in sorted(os.listdir(BENCH)):
    for e in json.load(open(os.path.join(BENCH, f))):
        with np.errstate(all="ignore"):
            angles.append(get_pts_angle_goatbench(e["Start_position"][0], e["Start_rotation"][0])[1])
nan = sum(np.isnan(a) for a in angles)
print(f"[start pose] episodes whose start yaw parses to NaN: {nan}/{len(angles)};",
      "get_quaternion(NaN) =", get_quaternion(float('nan'), 0), "(identity: all initial look-around views identical)")
print("[start pose] example stored rotation:", json.load(open(os.path.join(BENCH, sorted(os.listdir(BENCH))[0])))[0]["Start_rotation"][0],
      "(y-axis yaw stored as [w, x, y, z]; code reads [x, y, z, w])")

# 4. QA: one memory is chosen and the answer is parsed from the last line of the reply
ans = json.load(open(os.path.join(BASE, "lmee_answer.json")))
print("[QA] questions:", len(ans), "| pred letters:", dict(collections.Counter(a["pred_answer"] for a in ans)),
      "| gold letters:", dict(collections.Counter(a["answer"] for a in ans)))
print("[QA] accuracy:", round(100 * np.mean([a["pred_answer"] == a["answer"] for a in ans]), 2))
n_qa_steps = collections.Counter(a["task_step"] for a in ans)
print("[QA] task_step of QA subtasks (steps walked after answering):", dict(sorted(n_qa_steps.items())))

"""P3: time and effort per subtask, RCMem vs LMEE's MemoryExplorer (58-task subset, choice mode). CPU only.

Per run, over navigation subtasks:
- agent effort (hardware-independent): agent steps (task_step), metres walked (explore_dist), decisions;
- VLM seconds per decision: RCMem logs them (p3.step1_s + p2b.latency_s, plus p3.render_s for panoramas); MemoryExplorer
  does not, so they are read from its log: time from "Input prompt:" to the next "Response:" line (1 s resolution);
- wall-clock per subtask: sum over the run's log segments (halves, and restarts after a crash) of their elapsed time / navigation subtasks (includes
  perception, simulation, QA and model loading; GPUs were shared differently across phases, so compare with care).
Run (repo root, lmee env): python analysis/p3_latency.py   -> analysis/out_p3/latency.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze as A  # noqa: E402

RUNS = [("MemoryExplorer E0 (fixed start)", "p2_E0_fix_s77"), ("MemoryExplorer E0 (fixed start)", "p2_E0_fix_s78"),
        ("MemoryExplorer E0 (fixed start)", "p2_E0_fix_s79"), ("F1 (Qwen3, no regions)", "p3r_F1_s77"),
        ("RCMem RP (Qwen3)", "p3_RP_s77"), ("RCMem RPcu (Qwen3)", "p3_RPcu_s77")]
TS = re.compile(r"^(\d\d):(\d\d):(\d\d) - ")


def secs(line):
    m = TS.match(line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def log_stats(run):
    vlm, wall = [], 0
    for f in glob.glob(os.path.join(REPO, "analysis/runs", run, "log_*.log")):
        t_in, last = None, 0
        for line in open(f, errors="ignore"):
            t = secs(line)
            if t is None:
                continue
            if t < last:  # a resumed job restarts the elapsed clock: bank the previous segment
                wall += last
            last = t
            if line[11:].startswith("Input prompt:"):
                t_in = t
            elif line[11:].startswith("Response:") and t_in is not None:
                vlm.append(t - t_in)
                t_in = None
        wall += last
    return vlm, wall


def main():
    out = {}
    for label, run in RUNS:
        _, st, dec = A.load_records(os.path.join(REPO, "analysis/runs", run))
        nav = [s for s in st.values() if s["goal_type"] != "QA"]
        ds = [d for s in nav for d in dec.get(s["subtask"], [])]
        if "p2b" in (ds[0] if ds else {}):
            called = [d for d in ds if d["p2b"]["prompt_tokens"] > 0 or d.get("p3", {}).get("step1_tokens", 0) > 0]
            v = [d["p2b"]["latency_s"] + d.get("p3", {}).get("step1_s", 0.0) for d in called]
            r = [d.get("p3", {}).get("render_s", 0.0) for d in ds]
            src = "records"
        else:
            v, r, src = log_stats(run)[0], [0.0], "log timestamps"
        _, wall = log_stats(run)
        steps = [s["task_step"] for s in nav]
        dist = [s["explore_dist"] for s in nav if np.isfinite(s["explore_dist"])]
        out[run] = {"label": label, "nav_subtasks": len(nav), "decisions_per_subtask": len(ds) / len(nav),
                    "vlm_s_per_decision": float(np.mean(v)), "vlm_source": src, "render_s_per_decision": float(np.mean(r)),
                    "vlm_s_per_subtask": float(np.sum(v)) / len(nav) + float(np.sum(r)) / len(nav),
                    "steps_per_subtask": float(np.mean(steps)), "metres_per_subtask": float(np.mean(dist)),
                    "wall_s_per_subtask": wall / len(nav), "success": float(np.mean([s["success_by_distance"] for s in nav]))}
    json.dump(out, open(os.path.join(REPO, "analysis/out_p3/latency.json"), "w"), indent=1)
    print("| Run | SR | Decisions / subtask | VLM s / decision | VLM (+render) s / subtask | Agent steps / subtask | Metres walked / subtask | Wall-clock s / subtask |")
    print("|---|---|---|---|---|---|---|---|")
    for run, m in out.items():
        print(f"| {m['label']} `{run}` | {100 * m['success']:.1f} | {m['decisions_per_subtask']:.1f} | {m['vlm_s_per_decision']:.1f} ({m['vlm_source']}) | "
              f"{m['vlm_s_per_subtask']:.0f} | {m['steps_per_subtask']:.1f} | {m['metres_per_subtask']:.1f} | {m['wall_s_per_subtask']:.0f} |")


if __name__ == "__main__":
    main()

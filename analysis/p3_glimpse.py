"""P3 diagnostic: where small or brief goal sightings are lost in RCMem's pipeline (offline, CPU, ground truth only).

Over the navigation decisions of a logged run (default: the fixed baseline p2_E0_fix_s77, whose trajectories the P3-0
cache and the P3-0a tables were built from), at each decision: has the goal been visible in any captured view so far,
with at least T pixels of a 1280 x 1280 view (records' per-view GT instance counts, logged from 200 px)? For those
decisions, follow the sighting through RCMem's stages and report the first stage that loses it:
  pruned        no KEPT frame shows the goal with >= T px (P3-0 cache: kept = >= 10% new voxels at 0.05 m, more lenient
                than the online rule of 25%, so this is a lower bound)
  room hidden   the goal's GT room is not shown in step 1 (no kept frame has >= 5% of its pixels in it; P3-0a's list)
  tiny in pano  the goal covers < 1024 px (one 32 x 32 Qwen3-VL token) of its room's 1024 x 512 panorama (P3-0a goal_px)
  not detected  the detector never mapped a box to the goal (decision record's goal_obj_ids_mapping is empty)
  available     every stage keeps it
Each decision is also typed by the sighting: "small" if the goal's largest view so far is < SMALL px (0.25% of the
frame, about 64 x 64 px), and "from another room" if every view showing it with >= T px is dominated (pixel-weighted
GT region, analysis/analyze.dominant_region) by a room other than the goal's.
Run (repo root, lmee env): python analysis/p3_glimpse.py [--run p2_E0_fix_s77]   -> analysis/out_p3/glimpse.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze as A  # noqa: E402

CACHE = os.environ.get("P3_CACHE", "/ssd1/quang/embodied/outputs/p3_0_cache")
THRESH = [200, 1638, 4096]   # px of a 1280 x 1280 view: 0.012% (smallest logged), 0.1% (analyze.MIN_PX), 0.25%
SMALL = 4096
TOKEN_PX = 1024              # one Qwen3-VL image token = 32 x 32 px of the panorama
STAGES = ["pruned", "room hidden", "tiny in pano", "not detected", "available"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p2_E0_fix_s77")
    a = ap.parse_args()
    scenes, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", a.run))
    p3a = {}
    for l in open(os.path.join(REPO, f"analysis/out_p3_0/{a.run}__qwen3_band.jsonl")):
        d = json.loads(l)
        p3a[(d["subtask"], d["global_step"])] = d
    kept = {}
    res = {t: collections.Counter() for t in THRESH}
    split = {t: collections.defaultdict(collections.Counter) for t in THRESH}
    n_dec, examples = 0, collections.defaultdict(list)
    for sid, st in subtasks.items():
        if st["goal_type"] == "QA" or not decisions.get(sid):
            continue
        ep, objs = st["episode"], scenes[A.scene_of(sid)]
        if ep not in kept:
            f = os.path.join(CACHE, a.run, ep + ".npz")
            kept[ep] = set(np.load(f)["name"].tolist()) if os.path.exists(f) else None
        if kept[ep] is None:
            continue
        G = {int(x) for x in st["goal_obj_ids"]}
        groom = {objs[g]["region"] for g in G if g in objs} - {None, "_-1"}
        frames = st["frames"]
        for d in decisions[sid]:
            row = p3a.get((sid, d["global_step"]))
            if row is None:
                continue
            n_dec += 1
            upto = [n for n in frames if int(n.split("-")[0]) <= d["global_step"]]
            gpx = {n: sum(c for k, c in frames[n]["vis"].items() if int(k) in G) for n in upto}
            mx = max(gpx.values(), default=0)
            detected = any(d.get("goal_obj_ids_mapping", {}).values())
            pano_px = max([p for r, p in zip(row["regions"], row["goal_px"]) if r in groom] or [0])
            for t in THRESH:
                seen = [n for n, c in gpx.items() if c >= t]
                if not seen:
                    continue
                if not any(n in kept[ep] for n in seen):
                    stage = "pruned"
                elif not set(row["regions"]) & groom:
                    stage = "room hidden"
                elif pano_px < TOKEN_PX:
                    stage = "tiny in pano"
                elif not detected:
                    stage = "not detected"
                else:
                    stage = "available"
                other = all(A.dominant_region(frames, n, objs) not in groom for n in seen)
                kind = ("small" if mx < SMALL else "large") + (", from another room" if other else ", from its room")
                res[t][stage] += 1
                split[t][kind][stage] += 1
                if t == THRESH[0] and stage != "available" and len(examples[stage]) < 5:
                    examples[stage].append({"subtask": sid, "global_step": d["global_step"], "goal": st["class"],
                                            "max_goal_px": mx, "pano_px": pano_px, "kind": kind})
    out = {"run": a.run, "decisions": n_dec, "thresholds_px_of_1280x1280": THRESH, "small_px": SMALL,
           "stages": {t: dict(res[t]) for t in THRESH}, "by_kind": {t: {k: dict(v) for k, v in split[t].items()} for t in THRESH},
           "examples": examples}
    os.makedirs(os.path.join(REPO, "analysis/out_p3"), exist_ok=True)
    json.dump(out, open(os.path.join(REPO, "analysis/out_p3/glimpse.json"), "w"), indent=1)
    print(f"{n_dec} navigation decisions ({a.run}); decisions where the goal was visible so far, by the first stage that loses it:")
    print("| visible with >= px (of 1280²) | decisions | " + " | ".join(STAGES) + " |")
    for t in THRESH:
        n = sum(res[t].values())
        print(f"| {t} | {n} | " + " | ".join(f"{res[t][s]} ({100 * res[t][s] / max(n, 1):.0f}%)" for s in STAGES) + " |")
    t = THRESH[0]
    print(f"\nby sighting type (>= {t} px):")
    print("| sighting | decisions | " + " | ".join(STAGES) + " |")
    for k in sorted(split[t]):
        n = sum(split[t][k].values())
        print(f"| {k} | {n} | " + " | ".join(f"{split[t][k][s]} ({100 * split[t][k][s] / max(n, 1):.0f}%)" for s in STAGES) + " |")


if __name__ == "__main__":
    main()

"""F0 (DEV_PLAN.md P2b): token budget of a context-as-memory prompt, from existing run records.

For every navigation/QA decision of the recorded E0 runs, the context is every frame the agent had captured up to that
decision (memory persists across subtasks). Depth is re-rendered at each recorded camera pose (habitat-sim, 256x256,
same hfov/height/tilt as the runs) because the records keep poses, not depth. Settings:
  (a) all frames
  (b) frame-level voxel pruning (p2b_context.FramePruner) at 3 voxel sizes
  (c) token-level voxel pruning (p2b_context.TokenPruner) at the same voxel sizes
  (d) 2x2 token pooling of every frame (169 -> 49 tokens), alone and on top of (b)
Prompt length = the F1 prompt (p2b_context.nav_prompt / qa_prompt) with that decision's question, image goal and
frontiers, counted exactly with the Qwen2.5-VL tokenizer; images are counted with smart_resize. The per-frame object
list is the one part estimated: its length uses the frame's visible ground-truth instances as the object count
(F1 logs the exact count, and analysis/p2b_report.py compares the two).

Run (repo root, lmee env, one GPU for rendering):  CUDA_VISIBLE_DEVICES=1 python analysis/p2b_f0.py
Output: analysis/out_p2b/f0.json (+ f0_decisions.csv)
"""
import argparse
import collections
import csv
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import analyze as A  # noqa: E402
import p2b_context as C  # noqa: E402

OUT = os.path.join(REPO, "analysis/out_p2b")
RUNS = ["instr_choice_sub", "p1_E0_r2", "p1_E0_r3"]  # the three complete E0 runs of the 58-task subset (seed 77)
VOXELS = [float(x) for x in os.environ.get("F0_VOXELS", "0.05 0.1 0.2").split()]
RENDER = 256
GOAL_IMG = (1280, 1280)  # image goals are rendered at the run resolution and passed unresized
BUDGETS = [16000, 32000]


def frame_order(name):
    s, v = name.split("-view_")
    return int(s), int(v.split(".")[0])


def make_sim(scene_id):
    import habitat_sim
    from src.habitat import make_semantic_cfg
    split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
    base = os.path.join(REPO, "data/hm3d-0.2/hm3d", split, scene_id, scene_id.split("-")[1])
    s = {"scene": base + ".basis.glb", "default_agent": 0, "sensor_height": 1.5, "width": RENDER, "height": RENDER,
         "hfov": C.HFOV, "scene_dataset_config_file": os.path.join(REPO, "data/hm3d_annotated_basis.scene_dataset_config.json"),
         "camera_tilt": -30 * np.pi / 180}
    sim = habitat_sim.Simulator(make_semantic_cfg(s))
    return sim, sim.initialize_agent(0)


def render_depth(sim, agent, frame):
    """Re-render the depth of a recorded frame; returns (depth, cam_pose, position error vs the record)."""
    import habitat_sim
    import quaternion
    from src.habitat import get_quaternion
    fwd = np.asarray(frame["fwd"])
    angle = math.atan2(-fwd[0], -fwd[2])  # agent yaw: forward of yaw a is (-sin a, 0, -cos a)
    st = habitat_sim.AgentState()
    st.position = np.asarray(frame["pts"], dtype=np.float32)
    st.rotation = get_quaternion(angle, 0)
    agent.set_state(st)
    obs = sim.get_sensor_observations()
    ss = agent.get_state().sensor_states["depth_sensor"]
    pose = np.eye(4)
    pose[:3, :3] = quaternion.as_rotation_matrix(ss.rotation)
    pose[:3, 3] = ss.position
    err = float(np.linalg.norm(pose[:3, 3] - np.asarray(frame["cam"])) + np.linalg.norm(-pose[:3, 2] - fwd))
    return obs["depth_sensor"], pose, err


def per_frame_cost(counter):
    """Exact token cost of one frame entry with no object line, and the mean cost of one object line (common names)."""
    hw = lambda x: (C.PROMPT_PX, C.PROMPT_PX)
    s0, c0 = C.nav_prompt("q", None, [], [])
    n0 = counter.count(s0, c0, hw)
    s1, c1 = C.nav_prompt("q", None, [(12, "img", [])], [])
    none_line = len(counter.p.tokenizer("No detected object.\n")["input_ids"])
    frame0 = counter.count(s1, c1, hw) - n0 - none_line
    names = ["chair", "table", "picture", "cabinet", "sofa", "bed", "lamp", "potted plant", "towel", "kitchen counter"]
    s2, c2 = C.nav_prompt("q", None, [(12, "img", names)], [])
    per_obj = (counter.count(s2, c2, hw) - n0 - frame0) / len(names)
    return frame0, per_obj


def total(r, s, obj_ratio=1.0):
    """Prompt tokens of a setting; obj_ratio rescales the ground-truth object count to detected objects."""
    return r["overhead"] + r[f"img_{s}"] + r[f"n_{s}"] * r["frame_text"] + r["obj_line"] * obj_ratio * r[f"k_{s}"]


def summarize(rows, settings, obj_ratio=1.0):
    summ = {}
    for s in settings:
        x = np.array([total(r, s, obj_ratio) for r in rows], float)
        n = np.array([r[f"n_{s}"] for r in rows])
        summ[s] = {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                   "max": float(x.max()), **{f"share_le_{b // 1000}k": float(100 * np.mean(x <= b)) for b in BUDGETS},
                   "frames_mean": float(n.mean()), "frames_max": int(n.max())}
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=RUNS)
    p.add_argument("--limit_scenes", type=int, default=0, help="smoke test: only the first N scenes, no output files")
    args = p.parse_args()
    from transformers import AutoProcessor
    counter = C.TokenCounter(AutoProcessor.from_pretrained(os.path.join(REPO, "data/MemoryExplorer")))
    frame0, per_obj = per_frame_cost(counter)
    pooled = math.ceil(C.FRAME_GRID / 2) ** 2

    # ---- load records: per episode frames (union over subtasks) and decisions
    eps = {}  # (run, episode) -> {"scene", "frames", "decisions"}
    scenes = {}
    for run in args.runs:
        sc, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", run))
        scenes.update(sc)
        for sid, st in subtasks.items():
            e = eps.setdefault((run, st["episode"]), {"scene": A.scene_of(sid), "frames": {}, "decisions": []})
            e["frames"].update(st["frames"])
            for d in decisions.get(sid, []):
                e["decisions"].append((d, st))
    by_scene = collections.defaultdict(list)
    for k, e in eps.items():
        by_scene[e["scene"]].append(k)

    rows, render_err = [], []
    items = sorted(by_scene.items())[:args.limit_scenes or None]
    for si, (scene_id, keys) in enumerate(items):
        sim, agent = make_sim(scene_id)
        objs = scenes[scene_id]
        print(f"[{si + 1}/{len(by_scene)}] {scene_id}: {len(keys)} episodes", flush=True)
        for key in keys:
            e = eps[key]
            names = sorted(e["frames"], key=frame_order)
            fp = {v: C.FramePruner(v) for v in VOXELS}
            tp = {v: C.TokenPruner(v) for v in VOXELS}
            info = {}
            for n in names:
                depth, pose, err = render_depth(sim, agent, e["frames"][n])
                render_err.append(err)
                pts, _ = C.backproject(depth, pose, stride=1)
                tpts, tok = C.token_points(depth, pose)
                k = sum(1 for s, c in e["frames"][n]["vis"].items()
                        if c >= A.MIN_PX and objs.get(int(s), {}).get("cat", "Unknown") not in A.STRUCT)
                info[n] = {"step": frame_order(n)[0], "k": k,
                           "kept": {v: fp[v].add(pts)[0] for v in VOXELS},
                           "tok": {v: tp[v].add(tpts, tok) for v in VOXELS}}
            for d, st in e["decisions"]:
                g = int(d["global_step"])
                avail = [n for n in names if info[n]["step"] <= g]
                qa = st["goal_type"] == "QA"
                if qa:
                    s0, c0 = C.qa_prompt(st["question"], None, [], ego_img="ego")
                else:
                    s0, c0 = C.nav_prompt(st["question"], "goal" if st["goal_type"] == "image" else None, [],
                                          ["f"] * int(d["n_frontiers"]))
                over = counter.count(s0, c0, lambda x: GOAL_IMG if x == "goal" else (C.PROMPT_PX, C.PROMPT_PX))
                r = {"run": key[0], "episode": key[1], "subtask": d["subtask"], "global_step": g, "qa": qa,
                     "overhead": over, "frame_text": frame0 - C.FRAME_TOKENS, "obj_line": per_obj}
                def put(s, frames, img):
                    r[f"n_{s}"], r[f"k_{s}"], r[f"img_{s}"] = len(frames), sum(info[n]["k"] for n in frames), img
                put("a_all", avail, C.FRAME_TOKENS * len(avail))
                put("d_pool_all", avail, pooled * len(avail))
                G = {int(x) for x in st["goal_obj_ids"]} if not qa else {int(st["QA"]["object_id"].split("_")[-1])}
                shows = lambda n: bool(G & A.vis_of(e["frames"], n))
                r["goal_in_avail"] = any(shows(n) for n in avail)
                for v in VOXELS:
                    kept = [n for n in avail if info[n]["kept"][v]]
                    r[f"goal_kept_v{v}"] = any(shows(n) for n in kept)
                    put(f"b_frame_v{v}", kept, C.FRAME_TOKENS * len(kept))
                    put(f"d_pool_b{v}", kept, pooled * len(kept))
                    tk = [n for n in avail if info[n]["tok"][v]]
                    put(f"c_token_v{v}", tk, sum(info[n]["tok"][v] for n in tk))
                rows.append(r)
        sim.close()

    settings = ["a_all"] + [f"b_frame_v{v}" for v in VOXELS] + [f"c_token_v{v}" for v in VOXELS] + ["d_pool_all"] + \
               [f"d_pool_b{v}" for v in VOXELS]
    summ = summarize(rows, settings)
    ga = [r for r in rows if r["goal_in_avail"]]
    goal_ret = {f"b_frame_v{v}": float(100 * np.mean([r[f"goal_kept_v{v}"] for r in ga])) for v in VOXELS}
    res = {"goal_view_retention_pct": goal_ret, "decisions_goal_in_avail": len(ga), "runs": args.runs, "decisions": len(rows), "nav_decisions": sum(not r["qa"] for r in rows),
           "episodes": len(eps), "voxel_sizes": VOXELS, "novelty": C.NOVELTY, "max_depth": C.MAX_DEPTH,
           "frame_tokens": C.FRAME_TOKENS, "pooled_frame_tokens": pooled, "settings_list": settings,
           "frame_entry_text_tokens": frame0 - C.FRAME_TOKENS, "object_line_tokens": per_obj,
           "render_pose_error_max": float(max(render_err)), "settings": summ}
    if args.limit_scenes:
        print(json.dumps(res, indent=1))
        return
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "f0.json"), "w"), indent=1)
    with open(os.path.join(OUT, "f0_decisions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=1))
    for s, v in summ.items():
        print(f"{s:14s} " + " ".join(f"{k}={v[k]:.1f}" for k in v))


if __name__ == "__main__":
    main()

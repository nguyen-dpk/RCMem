"""P3-0a stage 1: re-render the views of a logged run and keep the frame-level voxel-pruned ones (Habitat only).

For every episode of a run, the views the agent saw (records' `frames`, logged poses) are rendered again with the
evaluation camera (1.5 m, hfov 120, tilt -30) at RES x RES: RGB, z-depth and HM3D semantic instances. A view is kept
when at least MIN_NEW of its 0.05 m voxels are new (frame-level pruning, as in P2b's F runs). Image-goal subtasks also
get their target image (first GOAT viewpoint). Everything the model side needs is written to one npz per episode, so
P3-0a (analysis/p3_0a.py, py3.11 env) never touches the simulator.

Run (lmee env, repo root): CUDA_VISIBLE_DEVICES=0 python analysis/p3_0_cache.py --run p2_E0_fix_s77
Output: $P3_CACHE/<run>/<episode>.npz (default /ssd1/quang/embodied/outputs/p3_0_cache)
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import analyze as A  # noqa: E402
import gt_question_views as g  # noqa: E402
from rcmem.geometry import HFOV, TILT, backproject_habitat, voxel_keys  # noqa: E402

CACHE = os.environ.get("P3_CACHE", "/ssd1/quang/embodied/outputs/p3_0_cache")
RES = 448
VOXEL, MIN_NEW = 0.05, 0.10


def yaw_of(fr):
    fx, _, fz = fr["fwd"]
    return float(np.arctan2(-fx, -fz))


def render(agent, sim, pos, rot):
    import habitat_sim
    st = habitat_sim.AgentState()
    st.position, st.rotation = np.asarray(pos, dtype=np.float32), rot
    agent.set_state(st)
    o = sim.get_sensor_observations()
    return o["color_sensor"][..., :3], o["depth_sensor"], o["semantic_sensor"]


def goal_image(sim, agent, scene, st):
    """Target image of an image-goal subtask: the render from the goal's first GOAT viewpoint."""
    import quaternion
    goals = json.load(open(os.path.join(g.GOAT, scene.split("-")[1] + ".json")))["goals"]
    by_num = {int(x["object_id"].split("_")[-1]): x for v in goals.values() for x in v}
    goal = by_num.get(st["goal_obj_ids"][0])
    if not goal or not goal["view_points"]:
        return None
    a = goal["view_points"][0]["agent_state"]
    r = a["rotation"]  # GOAT stores (x, y, z, w)
    return render(agent, sim, a["position"], quaternion.quaternion(r[3], r[0], r[1], r[2]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (smoke test)")
    args = ap.parse_args()
    out_dir = os.path.join(CACHE, args.run)
    os.makedirs(out_dir, exist_ok=True)
    g.RES = RES  # make_sim reads it at call time
    scenes, subtasks, _ = A.load_records(os.path.join(REPO, "analysis/runs", args.run))
    eps = collections.OrderedDict()
    for sid, st in subtasks.items():
        eps.setdefault(st["episode"], []).append(sid)
    sim_scene, sim, agent = None, None, None
    for n_ep, (ep, sids) in enumerate(eps.items()):
        if args.limit and n_ep >= args.limit:
            break
        f_out = os.path.join(out_dir, ep + ".npz")
        if os.path.exists(f_out):
            continue
        scene = A.scene_of(sids[0])
        if scene != sim_scene:
            if sim is not None:
                sim.close()
            sim, agent = g.make_sim(scene)
            sim_scene = scene
        t0 = time.time()
        frames = {}
        for s in sids:  # frames accumulate over the episode; union keeps every logged view
            frames.update(subtasks[s]["frames"])
        order = sorted(frames, key=lambda n: (int(n.split("-")[0]), n))
        seen, keep = set(), []
        rgbs, depths, sems, cams, yaws, steps, names, new_frac = [], [], [], [], [], [], [], []
        for name in order:
            fr = frames[name]
            yaw = yaw_of(fr)
            rgb, depth, sem = render(agent, sim, fr["pts"], g.yaw_q(yaw))
            pts, ok = backproject_habitat(depth, fr["cam"], yaw)
            keys = voxel_keys(pts[ok], VOXEL)
            nf = len(keys - seen) / max(len(keys), 1)
            if not keys or nf < MIN_NEW:
                continue
            seen |= keys
            assert sem.max() < 65535
            rgbs.append(rgb)
            depths.append(np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))  # millimetres
            sems.append(sem.astype(np.uint16))
            cams.append(fr["cam"])
            yaws.append(yaw)
            steps.append(int(name.split("-")[0]))
            names.append(name)
            new_frac.append(nf)
        extra = {}
        for s in sids:
            st = subtasks[s]
            if st["goal_type"] == "image":
                gi = goal_image(sim, agent, scene, st)
                if gi is not None:
                    extra["goal_" + s] = gi
        np.savez_compressed(
            f_out, rgb=np.stack(rgbs), depth_mm=np.stack(depths), sem=np.stack(sems), cam=np.asarray(cams),
            yaw=np.asarray(yaws), step=np.asarray(steps), name=np.asarray(names), new_frac=np.asarray(new_frac),
            all_steps=np.asarray([int(n.split("-")[0]) for n in order]), hfov=HFOV, tilt=TILT, res=RES, **extra)
        print(f"{ep}: kept {len(names)}/{len(order)} views, {time.time() - t0:.0f}s", flush=True)
    if sim is not None:
        sim.close()


if __name__ == "__main__":
    main()

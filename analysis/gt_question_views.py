"""Recover the ground-truth view each LMEE QA was generated from, and the HM3D instances visible in it.

For every QA in the 58-task subset: search agent poses around the QA's goal object (GOAT viewpoints plus navigable
points on a 0.5 m grid within 6 m x 12 yaws ranked by CLIP image similarity, then local refinement of the 8 best) with the evaluation camera (1.5 m, hfov 120,
tilt -30, 360x360), and keep the render that best matches the QA's source image (task_test/.../front.png) by mean
absolute RGB difference.
Output: analysis/out/gt_question_views.json  (one entry per QA: best match error, visible instance pixel counts)
        analysis/out/gt_view_pairs/<idx>.png   (source image | best render, for manual checking)

Run (lmee env, from repo root): CUDA_VISIBLE_DEVICES=1 python analysis/gt_question_views.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import habitat_sim  # noqa: E402
import quaternion  # noqa: E402
from src.habitat import make_semantic_cfg  # noqa: E402

BENCH = os.path.join(REPO, "data/LMEE-Bench/lmee_bench_sub")
TASKS = os.path.join(REPO, "data/LMEE-Bench/task_test")
GOAT = os.path.join(REPO, "data/goat_bench/val_unseen/content")
HM3D = os.path.join(REPO, "data/hm3d-0.2/hm3d")
OUT = os.path.join(REPO, "analysis/out")
RES = 360
MIN_FRAC = 200 / (1280 * 1280)  # same visibility threshold as instrumented_run.py (200 px at 1280^2)


def make_sim(scene_full):
    split = "train" if int(scene_full.split("-")[0]) < 800 else "val"
    sid = scene_full.split("-")[1]
    settings = {
        "scene": os.path.join(HM3D, split, scene_full, sid + ".basis.glb"),
        "default_agent": 0, "sensor_height": 1.5, "width": RES, "height": RES, "hfov": 120,
        "scene_dataset_config_file": os.path.join(REPO, "data/hm3d_annotated_basis.scene_dataset_config.json"),
        "camera_tilt": -30 * np.pi / 180,
    }
    sim = habitat_sim.Simulator(make_semantic_cfg(settings))
    return sim, sim.initialize_agent(0)


def render(agent, sim, pos, rot_q):
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = rot_q
    agent.set_state(st)
    obs = sim.get_sensor_observations()
    return obs["color_sensor"][..., :3], obs["semantic_sensor"]


def yaw_q(yaw):
    return quaternion.from_rotation_vector([0, yaw, 0])


def small(img):
    return np.asarray(Image.fromarray(img).resize((96, 96)), dtype=np.float32)


_CLIP = {}


def clip_embed(imgs):
    """L2-normalised CLIP ViT-L/14 image embeddings (same model the baseline uses for memory retrieval)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if not _CLIP:
        _CLIP["m"] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
        _CLIP["p"] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            x = _CLIP["p"](images=imgs[i:i + 64], return_tensors="pt")["pixel_values"].cuda()
            f = _CLIP["m"].get_image_features(pixel_values=x)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def search_poses(sim, agent, goal, src):
    """Global search: navigable points on a 0.5 m grid within 6 m of the goal (same floor) x 12 yaws, ranked by
    CLIP image similarity to the source image; then local refinement of the 8 best (+-0.25 m, +-15 deg) by mean
    absolute RGB error. Returns [(err, position, yaw, clip_sim)] for the best refined pose."""
    if goal is None:
        return []
    y = goal["view_points"][0]["agent_state"]["position"][1] if goal["view_points"] else goal["position"][1]
    gx, gz = goal["position"][0], goal["position"][2]
    pts = [vp["agent_state"]["position"] for vp in goal["view_points"]]
    for dx in np.arange(-6, 6.01, 0.5):
        for dz in np.arange(-6, 6.01, 0.5):
            p = np.asarray([gx + dx, y, gz + dz], dtype=np.float32)
            if dx * dx + dz * dz <= 36 and sim.pathfinder.is_navigable(p):
                pts.append(p.tolist())
    poses, imgs = [], []
    for p in pts:
        for k in range(12):
            yaw = 2 * np.pi * k / 12
            rgb, _ = render(agent, sim, p, yaw_q(yaw))
            poses.append((p, yaw))
            imgs.append(Image.fromarray(rgb))
    sims = clip_embed(imgs) @ clip_embed([Image.fromarray(src.astype(np.uint8))])[0]
    refined = []
    for i in np.argsort(-sims)[:8]:
        p, yaw = poses[i]
        for dx in (-0.25, 0, 0.25):
            for dz in (-0.25, 0, 0.25):
                for dy in np.deg2rad([-15, -7.5, 0, 7.5, 15]):
                    q = [p[0] + dx, p[1], p[2] + dz]
                    rgb, _ = render(agent, sim, q, yaw_q(yaw + dy))
                    refined.append((float(np.abs(rgb.astype(np.float32) - src).mean()), q, yaw + dy, float(sims[i])))
    refined.sort(key=lambda t: t[0])
    return refined[:1]


def main():
    os.makedirs(os.path.join(OUT, "gt_view_pairs"), exist_ok=True)
    results = []
    for f in sorted(os.listdir(BENCH)):
        scene_full = f.split(".")[0]
        sid = scene_full.split("-")[1]
        goals = json.load(open(os.path.join(GOAT, sid + ".json")))["goals"]
        by_id = {x["object_id"]: x for v in goals.values() for x in v}
        episodes = json.load(open(os.path.join(BENCH, f)))
        sim, agent = make_sim(scene_full)
        cats = {int(o.semantic_id): o.category.name() for o in sim.semantic_scene.objects if o is not None and o.category is not None}
        for ep_idx, ep in enumerate(episodes):
            for qa in ep["QAs"]:
                src = np.asarray(Image.open(os.path.join(TASKS, qa["image_path"])).convert("RGB").resize((RES, RES)), dtype=np.float32)
                goal = by_id.get(qa["object_id"])
                best = (np.inf, None, None)
                cands = search_poses(sim, agent, goal, src)
                for err, p, yaw, csim in cands:
                    rgb, sem = render(agent, sim, p, yaw_q(yaw))
                    if err < best[0]:
                        best = (err, (p, yaw), (rgb, sem))
                        clip_sim = csim
                entry = {"scene": scene_full, "episode": ep_idx, "object_id": qa["object_id"], "question": qa["question"],
                         "category": qa["category"], "image_path": qa["image_path"], "match_err": best[0], "vis": {}}
                if best[2] is not None:
                    rgb, sem = best[2]
                    ids, cnt = np.unique(sem, return_counts=True)
                    entry["vis"] = {int(i): {"px_frac": float(c) / sem.size, "cat": cats.get(int(i), "?")}
                                    for i, c in zip(ids, cnt) if c / sem.size >= MIN_FRAC and i != 0}
                    entry["viewpoint"] = {"position": [float(v) for v in best[1][0]], "yaw": float(best[1][1])}
                    entry["clip_sim"] = clip_sim
                    pair = np.concatenate([src.astype(np.uint8), rgb], axis=1)
                    Image.fromarray(pair).save(os.path.join(OUT, "gt_view_pairs", f"{len(results):03d}.png"))
                entry["idx"] = len(results)
                results.append(entry)
                print(entry["idx"], scene_full, qa["object_id"], f"err={best[0]:.1f}", flush=True)
        sim.close()
    json.dump(results, open(os.path.join(OUT, "gt_question_views.json"), "w"), indent=1)
    errs = np.array([r["match_err"] for r in results])
    print(f"QAs: {len(results)}  match err median {np.median(errs):.1f}  p90 {np.percentile(errs, 90):.1f}")


if __name__ == "__main__":
    main()

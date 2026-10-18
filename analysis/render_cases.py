"""Render panels for manual labelling of failure cases (the run deletes its images, so frames are re-rendered
from the poses recorded by instrumented_run.py).

Nav case : [goal view (GOAT viewpoint) | memory view the agent chose | first stored view that shows the goal, if any]
QA case  : [QA source image | GT-matched view | memory view the agent answered from]
Run (lmee env): CUDA_VISIBLE_DEVICES=1 python analysis/render_cases.py <subtask_id> [<subtask_id> ...]
Output: analysis/out/cases/<subtask_id>.png
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze  # noqa: E402
import gt_question_views as g  # noqa: E402
import quaternion  # noqa: E402


def frame_pose(fr):
    fx, _, fz = fr["fwd"]
    return fr["pts"], float(np.arctan2(-fx, -fz))


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([0, 0, 359, 14], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((2, 1), text[:60], fill=(255, 255, 0))
    return np.asarray(im)


def main(ids):
    scenes, subtasks, decisions = analyze.load_records()
    gp = os.path.join(g.OUT, "gt_question_views.json")
    gtv = {(x["scene"], x["episode"], x["object_id"], x["question"]): x
           for x in (json.load(open(gp)) if os.path.exists(gp) else [])}
    os.makedirs(os.path.join(g.OUT, "cases"), exist_ok=True)
    by_scene = {}
    for sid in ids:
        by_scene.setdefault(analyze.scene_of(sid), []).append(sid)
    for scene, sids in by_scene.items():
        sim, agent = g.make_sim(scene)
        goals = json.load(open(os.path.join(g.GOAT, scene.split("-")[1] + ".json")))["goals"]
        by_id = {x["object_id"]: x for v in goals.values() for x in v}
        for sid in sids:
            st = subtasks[sid]
            frames = st["frames"]
            decs = decisions.get(sid, [])
            panels = []
            if st["goal_type"] == "QA":
                qa = st["QA"]
                ep = int(st["episode"].split("_ep_")[1])
                src = np.asarray(Image.open(os.path.join(g.TASKS, qa["image_path"])).convert("RGB").resize((360, 360)))
                panels.append(label(src, "QA source  ans=" + qa["answer"] + " pred=" + str(st["pred_answer"])))
                v = gtv.get((scene, ep, qa["object_id"], qa["question"]))
                if v and "viewpoint" in v:
                    rgb, _ = g.render(agent, sim, v["viewpoint"]["position"], g.yaw_q(v["viewpoint"]["yaw"]))
                    panels.append(label(rgb, f"GT-matched view err={v['match_err']:.0f}"))
                ch = decs[0]["choice"] if decs else None
                if ch and ch["image"] in frames:
                    p, yaw = frame_pose(frames[ch["image"]])
                    rgb, _ = g.render(agent, sim, p, g.yaw_q(yaw))
                    panels.append(label(rgb, "answered from " + ch["image"]))
            else:
                goal = by_id.get(f"{st['class']}_{st['goal_obj_ids'][0]}") or next(
                    (x for x in by_id.values() if int(x["object_id"].split("_")[-1]) == st["goal_obj_ids"][0]), None)
                if goal and goal["view_points"]:
                    a = goal["view_points"][0]["agent_state"]
                    r = a["rotation"]
                    rgb, _ = g.render(agent, sim, a["position"], quaternion.quaternion(r[3], r[0], r[1], r[2]))
                    panels.append(label(rgb, "goal " + goal["object_id"]))
                ch = decs[-1]["choice"] if decs else None
                if ch and ch["type"] == "memory" and ch["image"] in frames:
                    p, yaw = frame_pose(frames[ch["image"]])
                    rgb, _ = g.render(agent, sim, p, g.yaw_q(yaw))
                    o = decs[-1]["objects"].get(str(ch["cluster"][0]), {})
                    panels.append(label(rgb, f"chosen {ch['image']} obj={o.get('cls')}"))
                seen = [n for n in frames if st["goal_obj_ids"][0] in analyze.vis_of(frames, n)]
                if seen:
                    p, yaw = frame_pose(frames[seen[0]])
                    rgb, _ = g.render(agent, sim, p, g.yaw_q(yaw))
                    panels.append(label(rgb, "first view showing goal " + seen[0]))
            if panels:
                Image.fromarray(np.concatenate(panels, axis=1)).save(os.path.join(g.OUT, "cases", sid + ".png"))
                print("saved", sid)
        sim.close()


if __name__ == "__main__":
    main(sys.argv[1:])

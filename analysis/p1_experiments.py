"""Phase-1 experiments (DEV_PLAN.md, P1) as runtime wrappers on the instrumented LMEE run. LMEE code is not modified.

  E0  unmodified baseline (only the logging of instrumented_run.py)
  E1  oracle region gate: navigation decisions only see memories/objects in the current goal's GT region (none left
      -> the agent must pick a frontier); QA only sees memories whose view lies in the question goal's GT region
      (falls back to all memories if none does)
  E2  room name in prompt: append the current subtask's room (task config) to the navigation question
  E4  verify on arrival: after arriving at a chosen object, one extra VLM call checks the view against the subtask
      (question + room, + goal image for image goals); if rejected the object is hidden for the rest of the subtask
      and the agent decides again
  E5  oracle earlier-goal exclusion: navigation decisions do not see objects whose nearest GT instance was the goal
      of an earlier subtask of the episode

Ground truth (HM3D regions/instances) is used here only for the E1/E5 oracles, as DEV_PLAN.md allows.
Usage (repo root): CUDA_VISIBLE_DEVICES=1 python analysis/p1_experiments.py --exp E1 --run_name p1_E1 --start_ratio 0 --end_ratio 1
Output: analysis/runs/<run_name>/records_*.jsonl (+ LMEE result files, log).
"""
import argparse
import base64
import collections
import copy
import glob
import inspect
import json
import logging
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import instrumented_run as ir  # noqa: E402  (applies the logging wrappers on import)
from src import eval_utils_qwen_lmee as eu  # noqa: E402
from src.tsdf_planner import TSDFPlanner, SnapShot  # noqa: E402
from src.utils import resize_image  # noqa: E402

STRUCT = {"wall", "floor", "ceiling", "Unknown", "unknown", "door frame", "window frame", "stair", "stairs", "beam",
          "pillar", "column", "baseboard", "ceiling lamp", "light", "ceiling light", "unknown object"}
MATCH_DIST = 1.5  # m, detection -> nearest GT instance (same rule as analyze.py)
MIN_PX = 1638     # 0.1% of a 1280x1280 frame (same as analyze.py)
EXP = os.environ.get("P1_EXP", "E0")
ST = {"gt": {}, "room": None, "goal_sids": set(), "reject": set(), "earlier": collections.defaultdict(set),
      "episode": None}


# ---------------------------------------------------------------- ground truth helpers (oracles only)
def gt_scene(scene):
    sid = scene._gap_scene_id
    if sid not in ST["gt"]:
        objs = [(int(o.semantic_id), o.category.name(), np.array(o.aabb.center), o.region.id if o.region else None)
                for o in scene.simulator.semantic_scene.objects if o is not None and o.category is not None]
        keep = [o for o in objs if o[1] not in STRUCT]
        ST["gt"][sid] = {"sid": np.array([o[0] for o in keep]), "c": np.stack([o[2] for o in keep]),
                         "region": {o[0]: o[3] for o in objs}, "cat": {o[0]: o[1] for o in objs}}
    return ST["gt"][sid]


def nearest_gt_sid(scene, center):
    g = gt_scene(scene)
    d = np.linalg.norm(g["c"] - np.asarray(center)[None], axis=1)
    i = int(np.argmin(d))
    return int(g["sid"][i]) if d[i] <= MATCH_DIST else None


def obj_region(scene, oid):
    s = nearest_gt_sid(scene, scene.objects[oid]["bbox"].center)
    return gt_scene(scene)["region"].get(s) if s is not None else None


def view_region(scene, rgb_id):
    """Pixel-weighted majority GT region of the non-structural instances visible in a stored view."""
    g, w = gt_scene(scene), collections.Counter()
    for k, c in ir.CUR["frames"].get(rgb_id, {}).get("vis", {}).items():
        if c >= MIN_PX and g["cat"].get(int(k)) not in STRUCT and g["region"].get(int(k)) not in (None, "_-1"):
            w[g["region"][int(k)]] += c
    return w.most_common(1)[0][0] if w else None


def task_rooms():
    out = {}
    for f in glob.glob(os.path.join(REPO, "data/LMEE-Bench/task_test/*/*/config.json")):
        c = json.load(open(f))
        out[(c["Scene"], c["Task instruction"].strip())] = {int(o.split("_")[-1]): reg for o, reg in c["Object"]}
    return out


ROOMS = task_rooms()


def room_text(label):
    """'Region 5: Tie: Dining room & Living room' -> 'dining room or living room'."""
    t = label.split(":", 1)[-1].strip()
    t = t[4:].strip() if t.lower().startswith("tie:") else t
    return " or ".join(x.strip().lower() for x in t.split("&")) if t else None


# ---------------------------------------------------------------- subtask bookkeeping
_prev_init = ir.Logger.init_subtask


def init_subtask(self, *a, **k):
    md = _prev_init(self, *a, **k)
    scene = k["scene"]
    if ir.CUR["episode"] != ST["episode"]:
        ST["episode"] = ir.CUR["episode"]
    ep = ST["episode"]
    ST["goal_sids"] = set(md["goal_obj_ids"])
    ST["reject"] = set()
    label = ROOMS.get((scene._gap_scene_id, k["task_instruct"].strip()), {}).get(md["goal_obj_ids"][0], "")
    ST["room"] = room_text(label) if label else None
    if EXP == "E2" and md["task_type"] != "QA" and ST["room"]:
        md["question"] += f"\nThe target object is located in the {ST['room']}."
    if md["task_type"] != "QA":
        md["_earlier_goals"] = set(ST["earlier"][ep])
        ST["earlier"][ep] |= set(md["goal_obj_ids"])
    return md


ir.Logger.init_subtask = init_subtask


# ---------------------------------------------------------------- memory filtering at each decision
def filtered(scene, keep_obj=None, keep_view=None):
    new = {}
    for rid, s in scene.snapshots.items():
        if keep_view is not None and not keep_view(rid):
            continue
        cl = [o for o in s.cluster if keep_obj is None or keep_obj(o)]
        if cl:
            c = copy.copy(s)  # keeps dynamic attributes (visual_prompt); scene.frames still holds the original
            c.cluster = cl
            new[rid] = c
    return new


def decision_filter(md, scene):
    nav = md["task_type"] != "QA"
    if EXP == "E1":
        if nav:
            g = gt_scene(scene)
            goal_regions = {g["region"].get(x) for x in md["goal_obj_ids"]}
            return filtered(scene, keep_obj=lambda o: obj_region(scene, o) in goal_regions), "nav_region_gate"
        goal = int(md["QA"]["object_id"].split("_")[-1])
        greg = gt_scene(scene)["region"].get(goal)
        snaps = filtered(scene, keep_view=lambda rid: view_region(scene, rid) == greg)
        return (snaps, "qa_region_gate") if snaps else (None, "qa_region_gate_fallback_all")
    if EXP == "E4" and nav and ST["reject"]:
        return filtered(scene, keep_obj=lambda o: o not in ST["reject"]), "hide_rejected"
    if EXP == "E5" and nav and md.get("_earlier_goals"):
        earlier = md["_earlier_goals"] - set(md["goal_obj_ids"])
        return filtered(scene, keep_obj=lambda o: nearest_gt_sid(scene, scene.objects[o]["bbox"].center) not in earlier), "hide_earlier_goals"
    return None, None


_inner_q = ir._orig_q


def query_with_filter(subtask_metadata, scene, *a, **k):
    snaps, mode = decision_filter(subtask_metadata, scene)
    if mode is not None:
        ir.emit({"type": "p1_filter", "exp": EXP, "subtask": ir.CUR["subtask"], "mode": mode,
                 "n_before": len(scene.snapshots), "n_after": len(snaps) if snaps is not None else len(scene.snapshots)})
    if snaps is None:
        return _inner_q(subtask_metadata, scene, *a, **k)
    orig = scene.snapshots
    scene.snapshots = snaps
    try:
        return _inner_q(subtask_metadata, scene, *a, **k)
    finally:
        scene.snapshots = orig


ir._orig_q = query_with_filter  # instrumented_run's logging wrapper calls this (so records show what the agent saw)

# ---------------------------------------------------------------- E4: verify on arrival
_orig_step = TSDFPlanner.agent_step


def verify(model, processor, md, rgb):
    sys_prompt = ("You check whether an embodied agent has reached the exact target object of its current subtask. "
                  "Answer with 'Yes' or 'No' on the first line, then one short reason.")
    content = [(f"Subtask: {md['question']}\n" + (f"The target object is located in the {ST['room']}.\n" if ST["room"] else ""),)]
    if md.get("image"):
        with open(md["image"], "rb") as f:
            content.append(("Target image:", base64.b64encode(f.read()).decode("utf-8")))
    content.append(("Current view of the agent (the chosen object should be in front of it):",
                    eu.encode_tensor2base64(resize_image(rgb, 360, 360))))
    content.append(("Is the object in front of the agent the exact target of the subtask, in the right room? Answer Yes or No.",))
    reply = eu.call_qwen25(sys_prompt, content, model, processor) or ""
    first = reply.strip().split()[0].lower().strip(".,:!") if reply.strip() else ""
    return first != "no", reply  # unparsable -> accept (baseline behaviour)


def agent_step(self, *a, **k):
    ret = _orig_step(self, *a, **k)
    if EXP != "E4" or ret[0] is None or not ret[5]:
        return ret
    c = inspect.currentframe().f_back.f_locals  # run_lmee.main
    mpc, md = c.get("max_point_choice"), c.get("subtask_metadata")
    if not isinstance(mpc, SnapShot) or md is None or md["task_type"] == "QA":
        return ret
    pts, angle = ret[0], ret[1]
    obs, _ = c["scene"].get_observation(pts, angle=angle)
    ok, reply = verify(c["mllm_model"], c["processor"], md, obs["color_sensor"])
    ir.emit({"type": "p1_verify", "subtask": ir.CUR["subtask"], "object": int(mpc.cluster[0]), "accepted": ok,
             "reply": reply[:300], "cnt_step": c.get("cnt_step")})
    logging.info(f"[E4] verify object {mpc.cluster[0]}: {'accepted' if ok else 'rejected'} | {reply[:120]}")
    if ok:
        return ret
    ST["reject"].add(int(mpc.cluster[0]))
    self.max_point, self.target_point = None, None  # main loop queries the VLM again at the next step
    return ret[:5] + (False,)


TSDFPlanner.agent_step = agent_step

if __name__ == "__main__":
    import time
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["E0", "E1", "E2", "E4", "E5"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--cfg_file", default=os.path.join(REPO, "analysis/cfg/instr_choice_sub.yaml"))
    p.add_argument("--start_ratio", type=float, default=0.0)
    p.add_argument("--end_ratio", type=float, default=1.0)
    args = p.parse_args()
    EXP = args.exp
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.exp_name = args.run_name
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    ir.OUT = open(os.path.join(cfg.output_dir, f"records_{args.start_ratio}_{args.end_ratio}.jsonl"), "a")
    t0 = time.time()

    class F(logging.Formatter):
        def formatTime(self, r, datefmt=None):
            s = int(r.created - t0)
            return f"{s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}"

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(os.path.join(cfg.output_dir, f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_1.log"), mode="a"),
        logging.StreamHandler()])
    for h in logging.getLogger().handlers:
        h.setFormatter(F(fmt="%(asctime)s - %(message)s"))
    logging.info(f"***** P1 experiment {EXP} -> {cfg.output_dir} *****")
    model, processor = ir.run_lmee.load_model(cfg)
    ir.run_lmee.main(cfg, model, processor, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=1, answer_type="choice")

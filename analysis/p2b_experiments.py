"""Phase-2b experiments (DEV_PLAN.md P2b) as runtime wrappers on the instrumented LMEE run. LMEE code is not modified.

  F1   context-as-memory: every decision sees the episode's voxel-pruned frames (p2b_context.FramePruner, frame level)
       in viewing order instead of the retrieved memory snapshots; no retrieval step. Same model (MemoryExplorer,
       Qwen2.5-VL-7B), same decision prompt structure; the answer is "Frame f, Object o", where the objects are the
       detector's boxes in frame f (drawn and numbered in the frame). The chosen object becomes the same single-object
       SnapShot the baseline navigates to, so frontiers, navigation and stopping are unchanged.
  F2   F1 with the context restricted to frames whose ground-truth region is the goal's region (hard mask; QA falls
       back to all frames when none qualifies, as in E1)
  F2b  F1 with the context restricted to frames whose ground-truth region TYPE matches the subtask's region type
       (analysis/out_p2/room_types.json; unknown types are kept, as in E1b)

Every run uses the start-orientation fix unless --no_fix_start. Ground truth is used only for the F2/F2b oracles.
Each decision record keeps the analysis fields of instrumented_run.py ("snapshots" = the frames in context, with the
objects listed for each) plus a "p2b" block: context size, exact prompt tokens, latency, raw reply.

Usage (repo root): CUDA_VISIBLE_DEVICES=1 python analysis/p2b_experiments.py --exp F1 --run_name p2b_F1_s77
Output: analysis/runs/<run_name>/records_*.jsonl (+ LMEE result files, log).
"""
import argparse
import base64
import inspect
import logging
import os
import random
import re
import sys
import time

import cv2
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import p1_experiments as P1  # noqa: E402  (instrumented logging + GT helpers; P1_EXP stays E0, so no P1 filter)
import p2b_context as C  # noqa: E402

ir = P1.ir
eu = P1.eu
from qwen_vl_utils import process_vision_info  # noqa: E402
from src.tsdf_planner import SnapShot  # noqa: E402
from src.utils import resize_image  # noqa: E402

EXP = "F1"
VOXEL = 0.05          # frame-level pruning voxel size, chosen from F0 (analysis/out_p2b/f0.json, see p2b_report.py)
MAX_CTX = 32000       # prompt-token cap; the oldest frames are dropped beyond it (logged; F0 says it rarely binds)
CTX = {"episode": None, "pruner": None, "kept": [], "cam": {}, "frac": {}}


# ---------------------------------------------------------------- frames: voxel pruning as they are captured
_prev_usg = ir.Scene.update_scene_graph


def update_scene_graph(self, *a, **k):
    out = _prev_usg(self, *a, **k)
    pts, _ = C.backproject(k["depth"], k["cam_pos"], stride=C.DEPTH_STRIDE)
    kept, frac = CTX["pruner"].add(pts)
    CTX["frac"][k["img_path"]] = frac
    CTX["cam"][k["img_path"]] = np.asarray(k["cam_pos"])[:3, 3]
    if kept:
        CTX["kept"].append(k["img_path"])
    return out


ir.Scene.update_scene_graph = update_scene_graph

_prev_ie = ir.Logger.init_episode


def init_episode(self, episode_id):
    CTX.update(episode=episode_id, pruner=C.FramePruner(VOXEL), kept=[], cam={}, frac={})
    return _prev_ie(self, episode_id)


ir.Logger.init_episode = init_episode


# ---------------------------------------------------------------- context masks (oracles)
def context_frames(md, scene):
    """Kept frames (viewing order) after the experiment's mask; returns (names, mask label)."""
    frames = list(CTX["kept"])
    nav = md["task_type"] != "QA"
    if EXP == "F1":
        return frames, "none"
    g = P1.gt_scene(scene)
    if nav:
        goals = [g["region"].get(x) for x in md["goal_obj_ids"]]
    else:
        goals = [g["region"].get(int(md["QA"]["object_id"].split("_")[-1]))]
    if EXP == "F2":
        keep = [n for n in frames if P1.view_region(scene, n) in set(goals)]
        label = "goal_region"
    else:  # F2b
        target = set().union(*[P1.rtypes(scene, r) for r in goals])
        def ok(n):
            t = P1.rtypes(scene, P1.view_region(scene, n))
            return not t or not target or bool(t & target)
        keep = [n for n in frames if ok(n)]
        label = "goal_region_type"
    if not nav and not keep:
        return frames, label + "_fallback_all"
    return keep, label


# ---------------------------------------------------------------- prompt pieces
def frame_objects(scene, name):
    """Detector boxes of a frame that still map to live scene objects: [(obj_id, class, xyxy)], first box per object."""
    fr = scene.frames.get(name)
    if fr is None or getattr(fr, "visual_prompt", None) is None:
        return []
    out, seen = [], set()
    for oid, box in zip(fr.visual_prompt.data["obj_id"], fr.visual_prompt.xyxy):
        oid = int(oid)
        if oid in seen or oid not in scene.objects or oid not in fr.full_obj_list:
            continue
        seen.add(oid)
        out.append((oid, scene.objects[oid]["class_name"], np.asarray(box, float)))
    return out


def draw_frame(rgb, objs, size=C.PROMPT_PX):
    img = np.ascontiguousarray(resize_image(rgb, size, size))
    H, W = rgb.shape[:2]
    for j, (_, _, b) in enumerate(objs):
        x0, y0, x1, y1 = (b * [size / W, size / H, size / W, size / H]).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
        t = str(j)
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x0, y0), (x0 + tw + 2, y0 + th + 3), (255, 0, 0), -1)
        cv2.putText(img, t, (x0 + 1, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def call_vlm(sys_prompt, content, model, processor):
    """LMEE's call_qwen25 (same messages, max_new_tokens=512), also returning prompt length and latency."""
    msgs = C.to_messages(sys_prompt, content)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    n = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512)
    torch.cuda.synchronize()
    dt = time.time() - t0
    reply = processor.batch_decode([out[0][n:]], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return reply, n, dt


def parse_nav(reply):
    """-> ("frame", f, o) | ("frontier", i, None) | None. Reads the ANSWER line (falls back to the last match)."""
    lines = [l.strip() for l in reply.split("\n") if l.strip().lower().startswith("answer:")]
    for src in (lines[:1], [reply]):
        s = " ".join(src).lower()
        m = list(re.finditer(r"frame\s*(\d+)\s*,?\s*object\s*(\d+)|frame\s*(\d+)|frontier\s*(\d+)", s))
        if m:
            x = m[-1] if src == [reply] else m[0]
            if x.group(1) is not None:
                return "frame", int(x.group(1)), int(x.group(2))
            if x.group(3) is not None:
                return "frame", int(x.group(3)), None
            return "frontier", int(x.group(4)), None
    return None


# ---------------------------------------------------------------- the decision
def f_query(subtask_metadata, scene, tsdf_planner, rgb_egocentric_views, cfg, model, processor, verbose=False):
    md, nav = subtask_metadata, subtask_metadata["task_type"] != "QA"
    names, mask = context_frames(md, scene)
    objs = {n: frame_objects(scene, n) for n in names}
    goal_b64 = None
    if md["task_type"] == "image":
        with open(md["image"], "rb") as f:
            goal_b64 = base64.b64encode(f.read()).decode("utf-8")
    frontier_b64 = [eu.encode_tensor2base64(f.feature) for f in tsdf_planner.frontiers] if nav else []
    ego = eu.encode_tensor2base64(rgb_egocentric_views[-1]) if (not nav and rgb_egocentric_views) else None

    def build(ns):
        fr = [(CTX["kept"].index(n), eu.encode_tensor2base64(draw_frame(scene.all_observations[n], objs[n])),
               [c for _, c, _ in objs[n]]) for n in ns]
        return C.nav_prompt(md["question"], goal_b64, fr, frontier_b64) if nav else C.qa_prompt(md["question"], None, fr, ego)

    hw = {}
    counter = C.TokenCounter(processor)
    size = lambda b: hw.setdefault(b, (1280, 1280) if b == goal_b64 else (C.PROMPT_PX, C.PROMPT_PX))
    sys_prompt, content = build(names)
    n_est, dropped = counter.count(sys_prompt, content, size), 0
    while n_est > MAX_CTX and names:  # cap: drop the oldest frames
        drop = max(1, len(names) // 10)
        names, dropped = names[drop:], dropped + drop
        sys_prompt, content = build(names)
        n_est = counter.count(sys_prompt, content, size)

    reply, n_tok, dt = call_vlm(sys_prompt, content, model, processor)
    logging.info(f"[{EXP}] context {len(names)}/{len(CTX['kept'])} frames ({mask}), {n_tok} tokens, {dt:.1f}s\n{reply}")
    by_index = {CTX["kept"].index(n): n for n in names}
    choice, ret, pred_answer, parse = None, None, None, "invalid"

    if nav:
        p = parse_nav(reply)
        if p and p[0] == "frame" and p[1] in by_index and objs[by_index[p[1]]]:
            n = by_index[p[1]]
            j = p[2] if p[2] is not None and 0 <= p[2] < len(objs[n]) else 0  # baseline: bad object index -> object 0
            oid = objs[n][j][0]
            parse = "frame" if j == p[2] else "frame_object_defaulted"
            snap = SnapShot(image=n, color=(random.random(), random.random(), random.random()), obs_point=np.empty(3),
                            full_obj_list={oid: scene.objects[oid]["conf"]}, cluster=[oid])
            ret = (snap, len(names))
            choice = {"type": "memory", "image": n, "cluster": [oid], "frame_index": p[1], "object_index": j}
            logging.info(f"Next choice: Frame {p[1]} ({n}), Object {j}: {oid} {scene.objects[oid]['class_name']}")
        elif p and p[0] == "frontier" and 0 <= p[1] < len(tsdf_planner.frontiers):
            fr = tsdf_planner.frontiers[p[1]]
            ret, parse = (fr, len(names)), "frontier"
            choice = {"type": "frontier", "image": fr.image, "position": np.asarray(fr.position).tolist()}
            logging.info(f"Next choice: Frontier {p[1]} at {fr.position}")
        else:
            logging.info(f"Invalid answer {p}, subtask fails (same as an invalid baseline reply)")
    else:
        lines = [l.strip() for l in reply.strip().split("\n") if l.strip()]
        m = re.search(r"frame\s*(\d+)", lines[0].lower()) if lines else None
        if m and int(m.group(1)) in by_index:
            n = by_index[int(m.group(1))]
            pred_answer = lines[-1] if len(lines) > 1 else ""
            cl = [o for o, _, _ in objs[n]]
            if not cl and scene.objects:  # frame without a detected object: navigate to the object nearest its camera
                cam = CTX["cam"][n]
                cl = [min(scene.objects, key=lambda o: np.linalg.norm(np.asarray(scene.objects[o]["bbox"].center) - cam))]
            if cl:
                snap = SnapShot(image=n, color=(random.random(), random.random(), random.random()), obs_point=np.empty(3),
                                full_obj_list={o: scene.objects[o]["conf"] for o in cl}, cluster=cl)
                ret, parse = (snap, pred_answer, len(names)), "frame"
                choice = {"type": "memory", "image": n, "cluster": cl, "frame_index": int(m.group(1))}
        if ret is None:
            logging.info("Invalid QA answer, subtask fails (same as an invalid baseline reply)")

    caller = inspect.currentframe().f_back.f_locals  # run_lmee.main
    ir.emit({
        "type": "decision", "episode": ir.CUR["episode"], "subtask": ir.CUR["subtask"],
        "cnt_step": caller.get("cnt_step"), "global_step": caller.get("global_step"),
        "pts": np.asarray(caller.get("pts")).tolist(),
        "goal_obj_ids_mapping": {str(g): [int(x) for x in v] for g, v in caller.get("goal_obj_ids_mapping", {}).items()},
        "snapshots": {n: [o for o, _, _ in objs[n]] for n in names},  # the memory = the frames in context
        "objects": {int(oid): {"cls": o["class_name"], "center": np.asarray(o["bbox"].center).tolist(), "image": o.get("image")}
                    for oid, o in scene.objects.items()},
        "n_frontiers": len(tsdf_planner.frontiers),
        "retrieval": [{"query": "context", "mode": "context", "top_k": None, "text_rank": [], "text_dist": [],
                       "image_rank": [], "image_dist": [], "index_aligned": True, "returned": list(names)}],
        "choice": choice, "pred_answer": pred_answer,
        "p2b": {"exp": EXP, "mask": mask, "voxel": VOXEL, "ctx_frames": len(names), "kept_frames": len(CTX["kept"]),
                "captured_frames": len(CTX["frac"]), "dropped_by_cap": dropped, "prompt_tokens": n_tok,
                "prompt_tokens_counted": n_est, "objects_listed": sum(len(objs[n]) for n in names),
                "latency_s": dt, "parse": parse, "reply": reply[:2000]},
    })
    return ret


ir.run_lmee.query_mllm_for_response = f_query

if __name__ == "__main__":
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["F1", "F2", "F2b"])
    p.add_argument("--no_fix_start", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--voxel", type=float, default=VOXEL)
    p.add_argument("--max_ctx", type=int, default=MAX_CTX)
    p.add_argument("--device_map", default=None, help="e.g. 'auto' to split the VLM over all visible GPUs")
    p.add_argument("--run_name", required=True)
    p.add_argument("--cfg_file", default=os.path.join(REPO, "analysis/cfg/instr_choice_sub.yaml"))
    p.add_argument("--start_ratio", type=float, default=0.0)
    p.add_argument("--end_ratio", type=float, default=1.0)
    args = p.parse_args()
    EXP, VOXEL, MAX_CTX = args.exp, args.voxel, args.max_ctx
    if EXP == "F2b":
        P1.load_room_types()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device_map:
        cfg.model.device = args.device_map
    if not args.no_fix_start:
        _orig_pts_angle = ir.run_lmee.get_pts_angle_goatbench

        def fixed_pts_angle(init_pos, init_rot):  # LMEE stores [w,x,y,z]; habitat's quat_from_coeffs wants [x,y,z,w]
            w, x, y, z = init_rot
            return _orig_pts_angle(init_pos, [x, y, z, w])
        ir.run_lmee.get_pts_angle_goatbench = fixed_pts_angle
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
    logging.info(f"***** experiment {EXP} voxel={VOXEL} max_ctx={MAX_CTX} fix_start={not args.no_fix_start} seed={cfg.seed} "
                 f"-> {cfg.output_dir} *****")
    model, processor = ir.run_lmee.load_model(cfg)
    ir.run_lmee.main(cfg, model, processor, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=1, answer_type="choice")

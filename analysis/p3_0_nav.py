"""P3-0 re-check, step 3 (DEV_PLAN.md): the region unit in navigation, with a two-step commit, on the Phase 2b wrapper
(analysis/p2b_experiments.py: voxel-pruned frames, "Frame f, Object o" answers, empty-context rule, start fix).

  PA  step 1 sees one A1-band panorama per observed ground-truth region (rcmem/panorama.py, built online from the kept
      frames' pixels in that region, rendered at the region's centre, cropped to the seen latitude band) plus the
      frontiers, and answers "Region k" or "Frontier i"
  PB  step 1 sees `Region k: <label>` + the region's 4 views with the largest share of its pixels (label = the VLM's
      room type from those views, cached per view set) plus the frontiers, same answers
  Step 2 (both): the Phase 2b decision prompt with only the chosen region's frames -> "Frame f, Object o" (or a
  frontier). A region is observed once a kept frame has >= MIN_SHARE of its pixels in it (as in P3-0a); its frames
  are the kept frames with >= MIN_SHARE in it. No region observed yet -> nearest frontier, no VLM call (empty-context
  rule). An invalid step-1 reply ends the subtask, as an invalid baseline reply does. QA decisions see all kept frames
  (as F1). Ground truth is used only for the regions (as F2), never for the choice.
Each decision record gets a "p3" block: observed regions in order, the goal's regions, step-1 reply / choice / tokens /
seconds (labels included for PB), and whether step 2 ran.

Usage (repo root, lmee env): CUDA_VISIBLE_DEVICES=0 python analysis/p3_0_nav.py --exp PA --vlm Qwen/Qwen3-VL-8B-Instruct --run_name x
       or as a queue worker: python analysis/p3_0_nav.py --worker analysis/runs/p3r_jobs.txt  (also runs F1/F2/F2b lines)
"""
import base64
import collections
import os
import re
import sys

if __name__ == "__main__" and os.environ.get("PYTHONUTF8") != "1":
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONUTF8": "1"})

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import p2b_experiments as F  # noqa: E402  (applies the Phase 2b hooks: frame pruning, decision override, start fix)
from rcmem.geometry import habitat_pose_to_onecanvas, intrinsics, to_onecanvas  # noqa: E402
from rcmem.panorama import RegionPanorama  # noqa: E402

ir, eu, P1, C = F.ir, F.eu, F.P1, F.C
MIN_SHARE = 0.05   # P3-0a: region observed once a kept frame has this share of its pixels in it
PANO_RES = 448     # frames are downsampled to this before splatting (P3-0a's cache resolution)
PANO_W = 1024
B_VIEWS = 4
REG = {"scene": None, "centres": {}, "region_of": None, "panos": {}, "share": collections.defaultdict(dict),
       "first_seen": {}, "labels": {}}

SYS1 = (
    "You are an embodied agent exploring an indoor environment.\n"
    "Your goal is to decide which room of the house, as seen so far (Regions), holds the target object described in the task "
    "instruction, or which unexplored region (Frontiers) to explore if none of them does.\n"
    "RULES:\n"
    "1) Your whole reply MUST contain a THOUGHT section that begins with 'THOUGHT:' (at most 10 short lines), then a single "
    "ANSWER line that begins with 'ANSWER:'.\n"
    "2) The ANSWER line MUST be one of these exact formats, where i is a plain integer:\n"
    "  ANSWER: Region i\n"
    "  ANSWER: Frontier i\n"
    "3) Choose a Region only if the target is in that room; after it you will see that room's frames and pick the object.\n"
    "4) Do NOT invent indices.\n"
)
INTRO = {
    "PA": ("Regions: each image below is one room of the house as seen so far, rendered over 360 degrees from the room's "
           "centre and cropped to the band of heights seen so far; grey areas have not been seen yet.\n"),
    "PB": "Regions: the views below are grouped by room; each group starts with the room's number and type.\n",
}
ASK_LABEL = "These views are from one room of a house. What type of room is it? Answer with the room type only, e.g. kitchen."


# ---------------------------------------------------------------- regions, built as frames are kept
def scene_regions(scene):
    """GT instance id -> region id, and region centres (P3-0a's rule: GT object box centre, lowest bottom + 1.5 m)."""
    if REG["scene"] == scene._gap_scene_id:
        return
    objs = [o for o in scene.simulator.semantic_scene.objects if o is not None]
    rid = np.full(max(int(o.semantic_id) for o in objs) + 2, "", dtype=object)
    lo, hi = {}, {}
    for o in objs:
        r = o.region.id if o.region is not None else None
        if r in (None, "_-1"):
            continue
        rid[int(o.semantic_id)] = r
        c, s = np.asarray(o.aabb.center), np.asarray(o.aabb.sizes) / 2
        lo[r], hi[r] = np.minimum(lo.get(r, c - s), c - s), np.maximum(hi.get(r, c + s), c + s)
    REG.update(scene=scene._gap_scene_id, region_of=rid,
               centres={r: np.array([(lo[r][0] + hi[r][0]) / 2, lo[r][1] + 1.5, (lo[r][2] + hi[r][2]) / 2]) for r in lo})


_prev_usg = ir.Scene.update_scene_graph


def update_scene_graph(self, *a, **k):
    n_before = len(F.CTX["kept"])
    out = _prev_usg(self, *a, **k)
    if F.EXP not in ("PA", "PB") or len(F.CTX["kept"]) == n_before or k.get("semantic_obs") is None:
        return out
    scene_regions(self)
    name = F.CTX["kept"][-1]
    sem = cv2.resize(np.asarray(k["semantic_obs"]).astype(np.int32), (PANO_RES, PANO_RES), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(np.asarray(k["depth"], np.float32), (PANO_RES, PANO_RES), interpolation=cv2.INTER_NEAREST) * 1000
    reg = REG["region_of"][np.clip(sem, 0, len(REG["region_of"]) - 1)]
    valid = (depth > 100) & (depth < 10000)
    rgb = cv2.resize(np.asarray(k["image_rgb"])[..., :3], (PANO_RES, PANO_RES), interpolation=cv2.INTER_AREA)
    pose, K = habitat_pose_to_onecanvas(k["cam_pos"]), intrinsics(PANO_RES, PANO_RES)
    for r in set(reg[valid].tolist()) - {""}:
        if r not in REG["centres"]:
            continue
        m = valid & (reg == r)
        if F.EXP == "PA":
            if r not in REG["panos"]:
                REG["panos"][r] = RegionPanorama(to_onecanvas(REG["centres"][r]), PANO_W, "cuda:0")
            REG["panos"][r].add(rgb, depth, m, pose, K)
        REG["share"][r][name] = m.sum() / m.size
        if REG["share"][r][name] >= MIN_SHARE and r not in REG["first_seen"]:
            REG["first_seen"][r] = len(REG["first_seen"])
    return out


ir.Scene.update_scene_graph = update_scene_graph
_prev_ie = ir.Logger.init_episode


def init_episode(self, episode_id):
    REG.update(panos={}, share=collections.defaultdict(dict), first_seen={}, labels={})
    return _prev_ie(self, episode_id)


ir.Logger.init_episode = init_episode


def region_frames(r):
    """Kept frames (viewing order) with >= MIN_SHARE of their pixels in region r."""
    return [n for n in F.CTX["kept"] if REG["share"][r].get(n, 0) >= MIN_SHARE]


# ---------------------------------------------------------------- step 1
def parse_step1(reply, n_regions, n_frontiers):
    lines = [l for l in reply.split("\n") if l.strip().lower().startswith("answer:")]
    s = (lines[0] if lines else reply).lower()
    m = list(re.finditer(r"(region|frontier)\s*(\d+)", s))
    if not m:
        return None
    x = m[0] if lines else m[-1]
    k = int(x.group(2))
    if x.group(1) == "region":
        return ("region", k) if 0 <= k < n_regions else None
    return ("frontier", k) if 0 <= k < n_frontiers else None


def step1(md, scene, tsdf_planner, regions, model, processor):
    """-> (parsed choice, block for the record)."""
    toks, secs, labels = 0, 0.0, []
    content = []
    q = f"Task Instruction: {md['question']}"
    if md["task_type"] == "image":
        content += [(f"{q}\nImage goal:", base64.b64encode(open(md["image"], "rb").read()).decode("utf-8")), ("\n",)]
    else:
        content.append((q + "\n",))
    content.append((INTRO[F.EXP],))
    for k, r in enumerate(regions):
        if F.EXP == "PA":
            content.append((f"Region {k}: ", eu.encode_tensor2base64(REG["panos"][r].band_image())))
            content.append(("\n",))
        else:
            views = sorted(REG["share"][r], key=lambda n: (-REG["share"][r][n], n))[:B_VIEWS]
            imgs = [eu.encode_tensor2base64(np.ascontiguousarray(F.resize_image(scene.all_observations[n], C.PROMPT_PX, C.PROMPT_PX)))
                    for n in views]
            key = (r, tuple(sorted(views)))
            if key not in REG["labels"]:
                rep, n_tok, dt = F.call_vlm("You describe rooms.", [("", im) for im in imgs] + [(ASK_LABEL,)], model, processor)
                REG["labels"][key] = rep.strip().split("\n")[0].rstrip(".").lower()[:40]
                toks, secs = toks + n_tok, secs + dt
            labels.append(REG["labels"][key])
            content.append((f"Region {k}: {labels[-1]}\n",))
            content += [("", im) for im in imgs]
    content.append(("The followings are all the Frontiers that you can explore:\n",))
    if not tsdf_planner.frontiers:
        content.append(("No Frontier is available.\n",))
    for i, f in enumerate(tsdf_planner.frontiers):
        content += [(f"Frontier {i}: ", eu.encode_tensor2base64(f.feature)), ("\n",)]
    content.append(("You MUST output the THOUGHT section followed by ONE valid ANSWER line:\n"
                    "   ANSWER: Region i  (the room that holds the target)\n"
                    "   ANSWER: Frontier i  (when no room shown holds it)\n",))
    reply, n_tok, dt = F.call_vlm(SYS1, content, model, processor)
    p = parse_step1(reply, len(regions), len(tsdf_planner.frontiers))
    block = {"step1_reply": reply[:2000], "step1_tokens": toks + n_tok, "step1_s": secs + dt, "step1_labels": labels,
             "step1_choice": None if p is None else (regions[p[1]] if p[0] == "region" else "explore"),
             "step1_frontier": p[1] if p and p[0] == "frontier" else None}
    return p, block


# ---------------------------------------------------------------- the decision
P3 = {}
_emit = ir.emit


def emit(rec):
    if rec.get("type") == "decision" and P3:
        rec["p3"] = dict(P3)
    return _emit(rec)


ir.emit = emit
_ctx_f = F.context_frames
_call_vlm = F.call_vlm


def query(subtask_metadata, scene, tsdf_planner, rgb_egocentric_views, cfg, model, processor, verbose=False):
    import inspect
    caller = inspect.currentframe().f_back.f_locals  # run_lmee.main
    args = (subtask_metadata, scene, tsdf_planner, rgb_egocentric_views, cfg, model, processor, verbose)
    P3.clear()
    md = subtask_metadata
    if F.EXP not in ("PA", "PB"):
        return F.f_query(*args, caller=caller)
    if md["task_type"] == "QA":  # QA: all kept frames, as F1
        F.context_frames = lambda md_, sc: (list(F.CTX["kept"]), "qa_all_frames")
        try:
            return F.f_query(*args, caller=caller)
        finally:
            F.context_frames = _ctx_f
    scene_regions(scene)
    regions = sorted(REG["first_seen"], key=REG["first_seen"].get)
    g = P1.gt_scene(scene)
    P3.update(exp=F.EXP, regions=regions, goal_regions=sorted({g["region"].get(x) for x in md["goal_obj_ids"]} - {None, "_-1"}),
              step1_choice=None, step2=False, step1_tokens=0, step1_s=0.0)
    if not regions:  # empty-context rule: no region seen yet
        F.context_frames = lambda md_, sc: ([], "no_region_seen")
        try:
            return F.f_query(*args, caller=caller)
        finally:
            F.context_frames = _ctx_f
    p, block = step1(md, scene, tsdf_planner, regions, model, processor)
    P3.update(block)
    if p is not None and p[0] == "region":
        r = regions[p[1]]
        P3["step2"] = True
        F.context_frames = lambda md_, sc: (region_frames(r), f"region {r}")
        try:
            return F.f_query(*args, caller=caller)
        finally:
            F.context_frames = _ctx_f
    # step 1 chose a frontier or was invalid: let the Phase 2b decision code act on that reply (same records/returns)
    canned = f"ANSWER: Frontier {p[1]}" if p is not None else ""
    F.context_frames, F.EMPTY_EXPLORE = (lambda md_, sc: ([], "step1_" + ("frontier" if p else "invalid"))), False
    F.call_vlm = lambda *a_, **k_: (canned, 0, 0.0)
    try:
        return F.f_query(*args, caller=caller)
    finally:
        F.context_frames, F.EMPTY_EXPLORE, F.call_vlm = _ctx_f, True, _call_vlm


ir.run_lmee.query_mllm_for_response = query

if __name__ == "__main__":
    a = F.parse(sys.argv[1:])
    if a.worker:
        F.worker(a.worker if os.path.isabs(a.worker) else os.path.join(REPO, a.worker))
    else:
        F.run_job(a)

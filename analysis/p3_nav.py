"""P3 (DEV_PLAN.md): RCMem zero-shot in the agent loop, on the P3-0 wrapper (analysis/p3_0_nav.py -> p2b_experiments.py:
voxel-pruned frames, "Frame f, Object o" answers, empty-context rule, start fix).

  RP  predicted regions: rcmem/regions.py segments the map built from every captured depth frame at narrow passages;
      each kept frame's pixels take the region of their 3D point's cell; one A1-band panorama per observed region,
      rendered at the region's maximum-visibility centre (rcmem/panorama.py, rebuilt at every decision since regions
      change as the map grows)
  RG  the same with ground-truth HM3D regions (pixel -> instance -> region; centre = box of the region's GT objects at
      floor + 1.5 m, as P3-0)
  E0  MemoryExplorer's own pipeline (bank, CLIP retrieval, its prompts) with the decision VLM given by --vlm
  F1 / F2 / F2b / PA / PB run as in p3_0_nav.py.
RG and RP: a region is observed once a kept frame has >= MIN_SHARE of its pixels in it; its frames are those kept
frames. Step 1 sees the subtask markers (rcmem/instruction.py: `<subtask i: goal, room>` for the subtasks so far, with
the region holding each earlier subtask's committed target, and "same room as subtask j" / "a different <room> from
subtask j" when the parse says so), then one panorama per observed region (in order of first observation) and the
frontiers, and answers "Region k" or "Frontier i". Step 2: the Phase 2b decision prompt with only region k's frames.
No region observed -> nearest frontier without a VLM call (empty-context rule); an invalid step-1 reply ends the
subtask. QA sees all kept frames (as F1). Ground truth is used for RG's regions and for logged diagnostics only.
Each decision record gets a "p3" block (regions, step-1 reply / choice / target, tokens, seconds, render time,
segmentation scores for RP). --viz also writes maps, panoramas and the step-1 prompt per decision to <run>/viz/ for
the pilot report (analysis/p3_qual.py).

Usage (repo root, lmee env): CUDA_VISIBLE_DEVICES=0 python analysis/p3_nav.py --exp RP --vlm Qwen/Qwen3-VL-8B-Instruct --run_name x
       or as a queue worker: python analysis/p3_nav.py --worker analysis/runs/p3_jobs.txt
"""
import collections
import json
import os
import re
import sys
import time

if __name__ == "__main__" and os.environ.get("PYTHONUTF8") != "1":
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONUTF8": "1"})

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import p1_experiments as P1  # noqa: E402

E0_QUERY = P1.ir.run_lmee.query_mllm_for_response  # MemoryExplorer's query (instrumented), before Phase 2b replaces it
import p3_0_nav as N  # noqa: E402  (installs the Phase 2b and P3-0 hooks)
from rcmem import instruction as INS  # noqa: E402
from rcmem import regions as RG  # noqa: E402
from rcmem.geometry import habitat_pose_to_onecanvas, intrinsics, to_onecanvas  # noqa: E402
from rcmem.panorama import RegionPanorama  # noqa: E402

F, ir, eu, C = N.F, N.ir, N.eu, N.C
RES, LAB = N.PANO_RES, 224  # frames for panoramas at 448 px; region labels and GT on a 224 grid (upsampled x2)
OPT = {"viz": False}
EP = {}  # per episode: builder, kept-frame data, subtask goals, committed targets


def new_episode():
    EP.clear()
    EP.update(rb=None, frames={}, goals={}, targets={}, path=[], gt_code={"": 0}, decision=0)


_prev_ie = ir.Logger.init_episode


def init_episode(self, episode_id):
    new_episode()
    return _prev_ie(self, episode_id)


ir.Logger.init_episode = init_episode
new_episode()


# ---------------------------------------------------------------- frames, as they are captured
_prev_usg = ir.Scene.update_scene_graph


def update_scene_graph(self, *a, **k):
    n_before = len(F.CTX["kept"])
    out = _prev_usg(self, *a, **k)
    if F.EXP not in ("RG", "RP") or k.get("semantic_obs") is None:
        return out
    N.scene_regions(self)  # GT instance -> region (RG labels, diagnostics) and GT centres (RG)
    depth = np.asarray(k["depth"], np.float32)
    if EP["rb"] is None:
        EP["rb"] = RG.RegionBuilder(np.asarray(k["pts"]), float(np.asarray(k["pts"])[1]))
    pts, _ = C.backproject(depth, k["cam_pos"], stride=C.DEPTH_STRIDE)
    EP["rb"].add(pts)  # the map takes every captured frame
    EP["path"].append(np.asarray(k["pts"], float)[[0, 2]].tolist())
    if len(F.CTX["kept"]) == n_before:
        return out
    name = F.CTX["kept"][-1]
    d448 = cv2.resize(depth, (RES, RES), interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(np.asarray(k["semantic_obs"]).astype(np.int32), (LAB, LAB), interpolation=cv2.INTER_NEAREST)
    gt = N.REG["region_of"][np.clip(sem, 0, len(N.REG["region_of"]) - 1)]
    code = np.array([EP["gt_code"].setdefault(r, len(EP["gt_code"])) for r in gt.ravel()], np.int16).reshape(LAB, LAB)
    d224 = cv2.resize(depth, (LAB, LAB), interpolation=cv2.INTER_NEAREST)
    p224, ok = backproject_full(d224, k["cam_pos"])
    EP["frames"][name] = {
        "rgb": cv2.resize(np.asarray(k["image_rgb"])[..., :3], (RES, RES), interpolation=cv2.INTER_AREA),
        "depth_mm": d448 * 1000, "pose": habitat_pose_to_onecanvas(k["cam_pos"]),
        "cells": np.where(ok, EP["rb"].cells(p224), -1).reshape(LAB, LAB), "gt": code, "sem": sem,
        "valid": ((d448 > 0.1) & (d448 < 10.0)), "frac": F.CTX["frac"].get(name)}
    return out


ir.Scene.update_scene_graph = update_scene_graph


def backproject_full(depth, cam_pose, hfov=C.HFOV):
    """Every pixel -> Habitat world point (H*W, 3) and validity (C.backproject keeps only the valid samples)."""
    H, W = depth.shape
    f = (W / 2.0) / np.tan(np.radians(hfov) / 2.0)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    d = depth.reshape(-1)
    p = np.stack([(u.reshape(-1) + 0.5 - W / 2) / f * d, -(v.reshape(-1) + 0.5 - H / 2) / f * d, -d], 1)
    R, t = np.asarray(cam_pose)[:3, :3], np.asarray(cam_pose)[:3, 3]
    return p @ R.T + t, (d > 0.1) & (d < 10.0)


# ---------------------------------------------------------------- regions at a decision
def key(code):
    """Integer label -> region key: the GT region id (RG) or "p<k>" (RP; predicted ids change between decisions)."""
    if F.EXP == "RG":
        return {v: k for k, v in EP["gt_code"].items()}[code]
    return f"p{code}"


def frame_labels():
    """name -> (LAB, LAB) int label per pixel (0 = none): GT region codes (RG) or predicted regions (RP)."""
    if F.EXP == "RG":
        return {n: fr["gt"].astype(np.int32) for n, fr in EP["frames"].items()}
    EP["rb"].segment()
    return {n: EP["rb"].assign_cells(fr["cells"].ravel()).reshape(LAB, LAB) for n, fr in EP["frames"].items()}


def observed(labels):
    """Region codes with >= MIN_SHARE of some kept frame, in order of first such frame, and each region's frames."""
    order, frames = [], collections.defaultdict(list)
    for n in F.CTX["kept"]:
        if n not in labels:
            continue
        cnt = np.bincount(labels[n].ravel())
        for r in np.nonzero(cnt[1:] >= N.MIN_SHARE * labels[n].size)[0] + 1:
            frames[int(r)].append(n)
            if r not in order:
                order.append(int(r))
    return order, frames


def centre(r):
    if F.EXP == "RG":
        return N.REG["centres"].get(key(r))
    return EP["rb"].centre(r)


def panoramas(regions, labels):
    """One A1-band panorama per region from every kept frame's pixels in it -> (images, hole fractions, centres)."""
    up = lambda a: np.repeat(np.repeat(a, RES // LAB, 0), RES // LAB, 1)
    K = intrinsics(RES, RES)
    panos, cs = {}, {}
    for r in regions:
        c = centre(r)
        if c is not None:
            cs[r] = c
            panos[r] = RegionPanorama(to_onecanvas(c), N.PANO_W, "cuda:0")
    for n, fr in EP["frames"].items():
        lab = up(labels[n])
        for r in set(np.flatnonzero(np.bincount(lab.ravel()))) & set(panos):
            panos[r].add(fr["rgb"], fr["depth_mm"], fr["valid"] & (lab == r), fr["pose"], K)
    return ({r: p.band_image() for r, p in panos.items()}, {r: p.hole_fraction() for r, p in panos.items()}, cs)


# ---------------------------------------------------------------- subtask markers
def subtask_index():
    return int(str(ir.CUR["subtask"]).rsplit("_", 1)[-1])


def goal_name(q):
    m = re.search(r"subtask of finding the (?:exact )?(.+?), which", q)
    return m.group(1) if m else ""


def instruction_of(q):
    m = re.search(r"The complete task is: (.*?)\nNow you need", q, re.S)
    return m.group(1).strip() if m else q


def region_of_point(t, regions):
    """Observed region code holding an earlier subtask's committed target (None otherwise): the region step 1 chose
    for it (RG: ids are stable), or the predicted region of its object's centre now (RP: ids change)."""
    if F.EXP == "RG":
        r = EP["gt_code"].get(t["region"])
    else:
        r = int(EP["rb"].assign(np.asarray(t["point"], float)[None])[0])
    return r if r in regions else None


def markers(md, regions):
    """Subtask marker lines for step 1 and the parse they come from."""
    i = subtask_index()
    EP["goals"][i] = goal_name(md["question"])
    idx = sorted(k for k in EP["goals"] if k <= i)
    parse = INS.parse(instruction_of(md["question"]), [EP["goals"][k] for k in idx])
    P = dict(zip(idx, parse))
    lines, where = ["Subtasks of the instruction, in order:\n"], {}
    for k in idx:
        room = P[k]["phrase"] and " / ".join(t.lower() for t in P[k]["types"]) or "room not stated"
        if k < i:
            t = EP["targets"].get(k)
            r = region_of_point(t, regions) if t else None
            where[k] = r
            tail = (f"done; its target is in Region {regions.index(r)}" if r is not None else
                    "done; its target is not in any Region shown" if t else "done; no target was chosen")
        else:
            tail = "current"
            same = [j for j in idx if j < i and P[j]["instance"] is not None and P[j]["instance"] == P[i]["instance"]]
            other = [j for j in idx if j < i and P[j]["types"] and P[j]["types"] == P[i]["types"]
                     and P[j]["instance"] != P[i]["instance"]]
            if same:
                j = same[-1]
                tail += f"; the same room as subtask {j + 1}" + (f" (Region {regions.index(where[j])})" if where.get(j) is not None else "")
            elif other:
                rs = sorted({where[j] for j in other if where.get(j) is not None}, key=regions.index)
                tail += (f"; a different {room} from subtask(s) {', '.join(str(j + 1) for j in other)}"
                         + (f" (not Region {', '.join(str(regions.index(r)) for r in rs)})" if rs else ""))
        lines.append(f"<subtask {k + 1}: {EP['goals'][k]}, {room}> {tail}\n")
    return "".join(lines), P[i]


# ---------------------------------------------------------------- step 1
SYS1 = N.SYS1
INTRO = N.INTRO["PA"]


def step1(md, tsdf_planner, regions, bands, model, processor):
    shown = [r for r in regions if r in bands]
    mk, parsed = markers(md, shown)
    content = []
    q = f"Task Instruction: {md['question']}"
    if md["task_type"] == "image":
        content += [(f"{q}\nImage goal:", N.base64.b64encode(open(md["image"], "rb").read()).decode("utf-8")), ("\n",)]
    else:
        content.append((q + "\n",))
    content += [(mk,), (INTRO,)]
    for k, r in enumerate(shown):
        content += [(f"Region {k}: ", eu.encode_tensor2base64(bands[r])), ("\n",)]
    content.append(("The followings are all the Frontiers that you can explore:\n",))
    if not tsdf_planner.frontiers:
        content.append(("No Frontier is available.\n",))
    for i, f in enumerate(tsdf_planner.frontiers):
        content += [(f"Frontier {i}: ", eu.encode_tensor2base64(f.feature)), ("\n",)]
    content.append(("You MUST output the THOUGHT section followed by ONE valid ANSWER line:\n"
                    "   ANSWER: Region i  (the room that holds the target)\n"
                    "   ANSWER: Frontier i  (when no room shown holds it)\n",))
    reply, n_tok, dt = F.call_vlm(SYS1, content, model, processor)
    p = N.parse_step1(reply, len(shown), len(tsdf_planner.frontiers))
    block = {"step1_reply": reply[:2000], "step1_tokens": n_tok, "step1_s": dt, "markers": mk, "parse": parsed,
             "shown": [key(r) for r in shown], "step1_choice": None if p is None else (key(shown[p[1]]) if p[0] == "region" else "explore"),
             "step1_frontier": p[1] if p and p[0] == "frontier" else None}
    return p, block, content


# ---------------------------------------------------------------- diagnostics (ground truth, never shown to the model)
def goal_info(scene, md, regions, labels, frames_of):
    """Goal regions (GT ids, or predicted regions holding a goal object's centre), the step-1 target, and the regions
    whose frames show >= 100 goal pixels."""
    sids = set(int(x) for x in md["goal_obj_ids"])
    gts = sorted({P1.gt_scene(scene)["region"].get(x) for x in sids} - {None, "_-1"})
    if F.EXP == "RG":
        goal = {EP["gt_code"][g] for g in gts if g in EP["gt_code"]}
    else:
        cs = [np.asarray(o.aabb.center) for o in scene.simulator.semantic_scene.objects
              if o is not None and int(o.semantic_id) in sids]
        goal = {int(k) for k in (EP["rb"].assign(np.stack(cs)) if cs else []) if k}
    tgt = next((key(r) for r in regions if r in goal), "explore")
    vis = [key(r) for r in regions
           if sum(int((np.isin(EP["frames"][n]["sem"], list(sids)) & (labels[n] == r)).sum()) for n in frames_of[r]) >= 100]
    return {"goal_regions": sorted(key(r) for r in goal), "goal_regions_gt": gts, "target": tgt, "goal_visible_regions": vis}


def gt_of_regions(labels, regions):
    """Predicted region -> majority GT region of its kept-frame pixels (diagnostics), and the segmentation scores."""
    inv = {v: k for k, v in EP["gt_code"].items()}
    pred = np.concatenate([labels[n].ravel() for n in EP["frames"]])
    gt = np.concatenate([EP["frames"][n]["gt"].ravel() for n in EP["frames"]]).astype(np.int64)
    maj = {}
    for r in regions:
        c = np.bincount(gt[(pred == r) & (gt > 0)])
        if c.any():
            maj[key(r)] = inv[int(c.argmax())]
    sub = slice(None, None, 4)  # every 4th pixel is plenty for the scores
    return maj, RG.score(pred[sub], np.array([inv[i] for i in range(len(inv))], dtype=object)[gt[sub]])


# ---------------------------------------------------------------- viz for the pilot report
def save_viz(run_dir, regions, bands, holes, cs, labels, content, info):
    d = os.path.join(run_dir, "viz", f"{ir.CUR['subtask']}_{EP['decision']:03d}")
    os.makedirs(d, exist_ok=True)
    for k, r in enumerate(regions):
        if r in bands:
            cv2.imwrite(os.path.join(d, f"pano_{k}.jpg"), bands[r][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    texts = [x[0] if len(x) == 1 else (x[0] + " <image>") for x in content]
    rb = EP["rb"]
    S = rb.size
    pred = rb.segment()
    gt = np.zeros(S * S, np.int32)  # GT region per cell from the kept frames' pixels
    for n, fr in EP["frames"].items():
        c, g = fr["cells"].ravel(), fr["gt"].ravel()
        ok = (c >= 0) & (g > 0)
        gt[c[ok]] = g[ok]
    gt = gt.reshape(S, S)
    wall = (rb.wall_n >= RG.WALL_MIN).reshape(S, S)
    obs = (rb.obs_n > 0).reshape(S, S)
    rows, cols = np.nonzero(obs | wall)
    a, b, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    pal = np.random.default_rng(3).integers(60, 255, (512, 3)).astype(np.uint8)

    def draw(lab, centres):
        im = pal[lab[a:b, c0:c1] % 512].copy()
        im[lab[a:b, c0:c1] == 0] = 0
        im[obs[a:b, c0:c1] & (lab[a:b, c0:c1] == 0)] = 70
        im[wall[a:b, c0:c1]] = 255
        im = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        cell = lambda x, z: (int(((x - rb.x0) / rb.cell - c0) * 4), int(((z - rb.z0) / rb.cell - a) * 4))
        pts = [cell(x, z) for x, z in EP["path"]]
        for p, q in zip(pts, pts[1:]):
            cv2.line(im, p, q, (0, 0, 255), 1)
        for k, (x, z) in centres:
            cv2.circle(im, cell(x, z), 6, (0, 0, 0), -1)
            cv2.putText(im, str(k), (cell(x, z)[0] - 4, cell(x, z)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for x, z in info.get("goal_xz", []):
            cv2.drawMarker(im, cell(x, z), (255, 0, 255), cv2.MARKER_STAR, 14, 2)
        return im

    shown = [(k, (cs[r][0], cs[r][2])) for k, r in enumerate(regions) if r in cs]
    if F.EXP == "RP":
        left = draw(pred, shown)
    else:
        left = draw(np.where(np.isin(gt, regions), gt, 0), shown)
    right = draw(gt, [])
    cv2.imwrite(os.path.join(d, "map.png"), np.hstack([left, np.full((left.shape[0], 8, 3), 255, np.uint8), right])[..., ::-1])
    json.dump({"prompt": texts, "regions": [key(r) for r in regions], "holes": [holes.get(r) for r in regions],
               "centres": [list(map(float, cs[r])) if r in cs else None for r in regions],
               "kept": list(F.CTX["kept"]), "frac": {n: F.CTX["frac"].get(n) for n in F.CTX["frac"]}, **info},
              open(os.path.join(d, "info.json"), "w"), default=str)


# ---------------------------------------------------------------- the decision
P3 = N.P3
LAST = {}
_emit = ir.emit


def emit(rec):
    if rec.get("type") == "decision":
        LAST["rec"] = rec
    return _emit(rec)


ir.emit = emit


def save_step2(d, scene, md, names):
    """The step-2 frames as the model saw them (numbered boxes), the GT goal box in green, the choice in the record."""
    sids = set(int(x) for x in md["goal_obj_ids"])
    out = []
    for n in names[-16:]:
        objs = F.frame_objects(scene, n)
        img = F.draw_frame(scene.all_observations[n], objs).copy()
        m = np.isin(EP["frames"][n]["sem"], list(sids)) if n in EP["frames"] else np.zeros((1, 1), bool)
        if m.sum() >= 20:
            ys, xs = np.nonzero(m)
            s_ = img.shape[0] / m.shape[0]
            cv2.rectangle(img, (int(xs.min() * s_), int(ys.min() * s_)), (int(xs.max() * s_), int(ys.max() * s_)), (0, 200, 0), 2)
        cv2.imwrite(os.path.join(d, f"s2_{F.CTX['kept'].index(n)}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append({"frame": F.CTX["kept"].index(n), "name": n, "objects": [c for _, c, _ in objs], "goal_px": int(m.sum())})
    return out


def query(subtask_metadata, scene, tsdf_planner, rgb_egocentric_views, cfg, model, processor, verbose=False):
    import inspect
    caller = inspect.currentframe().f_back.f_locals  # run_lmee.main
    args = (subtask_metadata, scene, tsdf_planner, rgb_egocentric_views, cfg, model, processor, verbose)
    md = subtask_metadata
    if F.EXP == "E0":
        P3.clear()
        return E0_QUERY(*args)
    if F.EXP not in ("RG", "RP"):
        return N.query(*args)
    P3.clear()
    if md["task_type"] == "QA":  # QA: all kept frames, as F1
        F.context_frames = lambda md_, sc: (list(F.CTX["kept"]), "qa_all_frames")
        try:
            return F.f_query(*args, caller=caller)
        finally:
            F.context_frames = N._ctx_f
    EP["decision"] += 1
    t0 = time.time()
    labels = frame_labels()
    regions, frames_of = observed(labels)
    bands, holes, cs = panoramas(regions, labels) if regions else ({}, {}, {})
    render_s = time.time() - t0
    info = goal_info(scene, md, regions, labels, frames_of)
    P3.update(exp=F.EXP, regions=[key(r) for r in regions], **info, render_s=render_s, n_regions=len(regions),
              holes=[holes.get(r) for r in regions], step1_choice=None, step2=False, step1_tokens=0, step1_s=0.0)
    if F.EXP == "RP":
        maj, sc = gt_of_regions(labels, regions)
        P3.update(region_gt=maj, seg=sc, n_map_regions=int(EP["rb"].segment().max()))
    content, p, shown_codes = [], None, []
    LAST.clear()
    if not regions:  # empty-context rule
        F.context_frames = lambda md_, sc: ([], "no_region_seen")
        try:
            ret = F.f_query(*args, caller=caller)
        finally:
            F.context_frames = N._ctx_f
    else:
        p, block, content = step1(md, tsdf_planner, regions, bands, model, processor)
        shown_codes = [r for r in regions if r in bands]
        P3.update(block)
        if p is not None and p[0] == "region":
            r = shown_codes[p[1]]
            P3["step2"] = True
            F.context_frames = lambda md_, sc: (frames_of[r], f"region {r}")
            try:
                ret = F.f_query(*args, caller=caller)
            finally:
                F.context_frames = N._ctx_f
            if ret is not None and hasattr(ret[0], "cluster") and ret[0].cluster:
                oid = ret[0].cluster[0]
                EP["targets"][subtask_index()] = {"point": np.asarray(scene.objects[oid]["bbox"].center).tolist(), "region": key(r)}
        else:
            canned = f"ANSWER: Frontier {p[1]}" if p is not None else ""
            F.context_frames, F.EMPTY_EXPLORE = (lambda md_, sc: ([], "step1_" + ("frontier" if p else "invalid"))), False
            F.call_vlm = lambda *a_, **k_: (canned, 0, 0.0)
            try:
                ret = F.f_query(*args, caller=caller)
            finally:
                F.context_frames, F.EMPTY_EXPLORE, F.call_vlm = N._ctx_f, True, N._call_vlm
    if OPT["viz"]:
        d = os.path.join(cfg.output_dir, "viz", f"{ir.CUR['subtask']}_{EP['decision']:03d}")
        os.makedirs(d, exist_ok=True)
        rec = LAST.get("rec", {})
        s2 = save_step2(d, scene, md, frames_of[shown_codes[p[1]]]) if P3.get("step2") else []
        # LMEE deletes the episode folder at its end: keep thumbnails of the frames captured since the last decision
        for n in [n for n in F.CTX["frac"] if n not in EP.setdefault("viz_seen", set())]:
            if n in scene.all_observations:
                th = cv2.resize(np.asarray(scene.all_observations[n])[..., :3], (128, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(d, f"f_{n[:-4]}.jpg"), th[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 60])
            EP["viz_seen"].add(n)
        if md["task_type"] == "image" and not os.path.exists(os.path.join(d, "..", f"{ir.CUR['subtask']}_goal.jpg")):
            cv2.imwrite(os.path.join(d, "..", f"{ir.CUR['subtask']}_goal.jpg"), cv2.imread(md["image"]), [cv2.IMWRITE_JPEG_QUALITY, 70])
        sids = set(int(x) for x in md["goal_obj_ids"])
        gxz = [np.asarray(o.aabb.center)[[0, 2]].tolist() for o in scene.simulator.semantic_scene.objects
               if o is not None and int(o.semantic_id) in sids]
        save_viz(cfg.output_dir, regions, bands, holes, cs, labels, content,
                 {"goal_xz": gxz, "p3": {k: v for k, v in P3.items()}, "frames_of": {key(r): v for r, v in frames_of.items()},
                  "frontier_imgs": [f.image for f in tsdf_planner.frontiers], "step2_frames": s2,
                  "choice": rec.get("choice"), "step2_reply": rec.get("p2b", {}).get("reply") if P3.get("step2") else None,
                  "step2_tokens": rec.get("p2b", {}).get("prompt_tokens") if P3.get("step2") else None})
    return ret


ir.run_lmee.query_mllm_for_response = query

_run_job = F.run_job


def run_job(a, model=None, processor=None):
    OPT["viz"] = bool(getattr(a, "viz", False))
    return _run_job(a, model, processor)


F.run_job = run_job

if __name__ == "__main__":
    a = F.parse(sys.argv[1:])
    if a.worker:
        F.worker(a.worker if os.path.isabs(a.worker) else os.path.join(REPO, a.worker))
    else:
        run_job(a)

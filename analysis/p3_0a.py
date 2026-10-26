"""P3-0a stage 2: which region unit does a zero-shot VLM read best? (offline, over the logged decisions of a run)

Input: the frame cache of analysis/p3_0_cache.py (kept views of every episode, frame-level voxel pruning at 0.05 m).
For each navigation decision (description and image subtasks) the views kept up to that step are split by
ground-truth HM3D region. A region is "observed" once a kept view has >= MIN_SHARE of its pixels in it.
- A1   (--model qwen25): one RGB panorama per observed region (rcmem/panorama.py, OneCanvas projection, rendered at
       the region's centre), all panoramas in one prompt; MemoryExplorer's Qwen2.5-VL-7B names a panorama or
       "not seen yet" and says whether the goal is visible.
- B    (--model qwen25): the same VLM labels each region from its own views, then sees `<region k: label>` + those
       views per region and names a region or "not seen yet". B_FRAMES views per region, about A1's token count.
- C    (--model qwen3): Qwen3-VL-8B-Instruct (base) on A1's RGB panoramas: the A1 prompt (C_multi) and the
       per-region protocol below (C_pr). Backbone control for A2.
- C_B  (--model qwen3_B): option B with Qwen3-VL-8B base, so A and B are also compared on the control's backbone.
- A2   (--model onecanvas): OneCanvas-Qwen3-VL-8B reads one feature canvas per region (OneCanvas's reprojection with
       center_override = region centre). The released checkpoint takes one canvas per sequence (multi-canvas blocks
       need temporal_raw_frame_index, which it was not trained with), so A2 uses the per-region protocol:
       P(yes) to "is this the subtask's room?" per region; the best region if P(yes) > 0.5, else "not seen yet";
       then "is the target visible?" on the chosen region. OneCanvas takes no second image, so image-goal subtasks
       give A2 only the text (the goal's class); compare A2 on description subtasks.
Answers are cached on identical inputs (greedy decoding, so exact). Scoring: analysis/p3_0_report.py.

Run (onecanvas env, repo root): CUDA_VISIBLE_DEVICES=0 python analysis/p3_0a.py --run p2_E0_fix_s77 --model qwen25
Output: analysis/out_p3_0/<run>__<model>.jsonl (one row per decision; resumable per subtask);
        analysis/out_p3_0/examples/ (panoramas, B views and decision rows for 5 decisions; --model qwen25, seed 77)
"""
import argparse
import ast
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import analyze as A  # noqa: E402
from rcmem.geometry import c2w_onecanvas, intrinsics, to_onecanvas  # noqa: E402
from rcmem.panorama import RegionPanorama  # noqa: E402

CACHE = os.environ.get("P3_CACHE", "/ssd1/quang/embodied/outputs/p3_0_cache")
OUT = os.path.join(REPO, "analysis/out_p3_0")
CKPT = {"qwen25": os.path.join(REPO, "data/MemoryExplorer"),
        "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3_B": "Qwen/Qwen3-VL-8B-Instruct",
        "onecanvas": "/ssd1/quang/embodied/models/onecanvas/OneCanvas-Qwen3-VL-8B"}
MIN_SHARE = 0.05   # region observed once a kept view has this share of its pixels in it
PANO_W = 1024      # panorama 1024 x 512 (full sphere)
B_FRAMES, B_RES = 4, 336
GOAL_PX = 100      # goal "visible" in a region panorama: at least this many of its pixels
N_EXAMPLES = 5

SYS = "You are an embodied agent exploring a house to find objects."
INTRO_A = ("Each panorama below shows one room of the house as seen so far, rendered over 360 degrees from the room's "
           "centre; grey areas have not been seen yet.")
ASK_A = ("Which panorama shows the room where the target of the current subtask is? If none of them is that room, "
         "answer 'not seen yet'. Reply in exactly two lines:\nRegion: <panorama number or not seen yet>\n"
         "Goal visible: <yes or no>")
INTRO_B = "The views below are grouped by room; each group starts with a room marker."
ASK_B = ("Which room is the one where the target of the current subtask is? If none of them is that room, answer "
         "'not seen yet'. Reply in exactly one line:\nRegion: <room number or not seen yet>")
ASK_LABEL = "These views are from one room of a house. What type of room is it? Answer with the room type only, e.g. kitchen."
PR_RGB = ("The image is a 360-degree panorama of one room of the house, as seen so far from the room's centre; grey "
          "areas have not been seen yet.")
PR_CANVAS = "The scene is one room of the house, as seen so far."
ASK_ROOM = "Is this the room where the target of the current subtask is? Answer yes or no."
ASK_VIS = "Is the target of the current subtask visible in this room? Answer yes or no."


# ------------------------------------------------------------------------------------------------ prompts / parsing
def text_only(question):
    """Subtask text without the reference to a target image (for a model that cannot take one)."""
    return re.sub(r",? which is captured at the center of the following image\?.*", ".", question, flags=re.S)


def parse_region(reply, n):
    m = re.search(r"region\s*:\s*(.*)", reply, re.I)
    s = (m.group(1) if m else reply).strip().lower()
    if "not seen" in s:
        return "none"
    k = re.search(r"\d+", s)
    return int(k.group()) - 1 if k and 1 <= int(k.group()) <= n else "invalid"


def parse_visible(reply):
    m = re.search(r"goal visible\s*:\s*(yes|no)", reply, re.I)
    return None if m is None else m.group(1).lower() == "yes"


# ------------------------------------------------------------------------------------------------ models
class HFVLM:
    """Qwen2.5-VL / Qwen3-VL through transformers; images are numpy arrays, resized by the processor."""

    def __init__(self, path):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                                 device_map="cuda:0").eval()
        self.proc = AutoProcessor.from_pretrained(path)
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})

    def _inputs(self, sys_prompt, parts):
        content = [{"type": "text", "text": p} if isinstance(p, str) else {"type": "image"} for p in parts]
        msgs = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [Image.fromarray(p) for p in parts if not isinstance(p, str)]
        return self.proc(text=[text], images=imgs or None, return_tensors="pt").to(self.model.device)

    def _n_img(self, inp):
        return int((inp.input_ids == self.model.config.image_token_id).sum())

    @torch.no_grad()
    def generate(self, sys_prompt, parts, max_new_tokens=24):
        inp = self._inputs(sys_prompt, parts)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return reply.strip(), int(inp.input_ids.shape[1]), self._n_img(inp)

    @torch.no_grad()
    def p_yes(self, sys_prompt, parts):
        inp = self._inputs(sys_prompt, parts)
        p = self.model(**inp).logits[0, -1].float().softmax(-1)
        y, n = float(p[self.yes].sum()), float(p[self.no].sum())
        return y / max(y + n, 1e-9), int(inp.input_ids.shape[1]), self._n_img(inp)


class OneCanvas:
    """OneCanvas-Qwen3-VL-8B: per-frame Qwen3-VL features, reprojected onto one canvas per region."""

    def __init__(self, path):
        sys.path.insert(0, os.path.join(REPO, "third_party/onecanvas"))
        sys.path.insert(0, os.path.join(REPO, "third_party/onecanvas/training"))  # its `onecanvas` package alias
        from inference.pipeline import _build_adapter_config, _image_pad_token_id, load_onecanvas_model
        from model_adapters.qwen3_vl.adapter import prepare_batch
        from model_adapters.qwen3_vl.feature_extraction import extract_features
        from reprojection import reproject_scene
        self.model, self.proc = load_onecanvas_model(path, device="cuda:0", attn_implementation="sdpa")
        self.cfg = _build_adapter_config(self.model, self.proc)
        self.pad_id = _image_pad_token_id(self.proc)
        self._prepare, self._extract, self._reproject = prepare_batch, extract_features, reproject_scene
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")})

    @torch.no_grad()
    def features(self, rgbs, chunk=16):
        out = []
        for i in range(0, len(rgbs), chunk):
            ip = self.proc.image_processor(images=[Image.fromarray(x) for x in rgbs[i:i + chunk]], return_tensors="pt")
            out.append(self._extract(self.model, ip["pixel_values"].to(self.model.device, torch.bfloat16),
                                     ip["image_grid_thw"].to(self.model.device)))
        return torch.cat(out)

    @torch.no_grad()
    def p_yes(self, question, feats, depths_mm, poses, K, dims, centre):
        dev = self.model.device
        scene = self._reproject(features=feats, depths=depths_mm.to(dev), poses=poses, intrinsics=K,
                                image_dims=dims.to(dev), device=str(dev), center_override=centre)
        if scene.n_valid == 0:
            return 0.0, 0
        dummy = Image.new("RGB", (64, 64))
        msg = [{"role": "user", "content": [{"type": "image", "image": dummy}, {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[dummy], return_tensors="pt").to(dev)
        pos = (inp.input_ids[0] == self.pad_id).nonzero(as_tuple=True)[0]
        proj = self._prepare(scene=scene, input_ids=inp.input_ids, attention_mask=inp.attention_mask,
                             labels=torch.full_like(inp.input_ids, -100), first_idx=int(pos[0]), last_idx=int(pos[-1]),
                             config=self.cfg)
        out = self.model.generate(
            input_ids=proj["input_ids"].unsqueeze(0).to(dev), attention_mask=proj["attention_mask"].unsqueeze(0).to(dev),
            projection_done=True, projected_input_ids=[proj["input_ids"].to(dev)],
            projected_position_ids=[proj["position_ids"].to(dev)],
            projected_attention_mask=[proj["attention_mask"].to(dev)], projected_labels=None,
            projected_embeds=[proj["embeds"].to(dev)], projected_aux_layers=[proj["aux_layers"]],
            projected_depth_bins=[proj["depth_bins"].to(dev)] if proj["depth_bins"] is not None else None,
            projected_ray_dirs=[proj["ray_dirs"].to(dev)] if proj.get("ray_dirs") is not None else None,
            rope_deltas=proj["rope_deltas"].unsqueeze(0).to(dev), max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True)
        p = out.scores[0][0].float().softmax(-1)
        y, n = float(p[self.yes].sum()), float(p[self.no].sum())
        return y / max(y + n, 1e-9), int(scene.n_valid)


# ------------------------------------------------------------------------------------------------ regions
def region_centres(objs):
    """Region -> centre of the bounding box of its GT objects, at eye height (lowest object bottom + 1.5 m)."""
    lo, hi = {}, {}
    for o in objs.values():
        r = o["region"]
        if r in (None, "_-1"):
            continue
        c, s = np.asarray(o["center"]), np.asarray(o["size"]) / 2
        lo[r] = np.minimum(lo.get(r, c - s), c - s)
        hi[r] = np.maximum(hi.get(r, c + s), c + s)
    return {r: np.array([(lo[r][0] + hi[r][0]) / 2, lo[r][1] + 1.5, (lo[r][2] + hi[r][2]) / 2]) for r in lo}


def load_room_types():
    return {sc: {"_" + k: v["types"] for k, v in m.items()}
            for sc, m in json.load(open(os.path.join(REPO, "analysis/out_p2/room_types.json")))["scenes"].items()}


def choice_type(c):
    c = ast.literal_eval(c) if isinstance(c, str) else c
    return (c or {}).get("type")


def pr_choice(ps):
    """Per-region protocol: best region if its P(yes) > 0.5, else 'not seen yet'."""
    if not ps or max(ps) <= 0.5:
        return "none"
    return int(np.argmax(ps))


# ------------------------------------------------------------------------------------------------ main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True, choices=list(CKPT))
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (smoke test)")
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "examples"), exist_ok=True)
    out_f = os.path.join(OUT, f"{args.run}__{args.model}.jsonl")
    done = {json.loads(l)["subtask"] for l in open(out_f)} if os.path.exists(out_f) else set()
    scenes, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", args.run))
    rtypes = load_room_types()
    eps = collections.OrderedDict()
    for sid, st in subtasks.items():
        eps.setdefault(st["episode"], []).append(sid)
    vlm = OneCanvas(CKPT[args.model]) if args.model == "onecanvas" else HFVLM(CKPT[args.model])
    dev = "cuda:0"
    make_examples = args.model == "qwen25" and args.run.endswith("s77")
    n_ex = len(os.listdir(os.path.join(OUT, "examples")))
    f = open(out_f, "a")
    for n_ep, (ep, sids) in enumerate(eps.items()):
        if args.limit and n_ep >= args.limit:
            break
        nav = [s for s in sids if subtasks[s]["goal_type"] != "QA" and s not in done and decisions.get(s)]
        if not nav:
            continue
        t_ep = time.time()
        scene = A.scene_of(ep)
        objs, types = scenes[scene], rtypes.get(scene, {})
        inst_region = np.full(max(objs) + 2, "", dtype=object)
        for k, o in objs.items():
            inst_region[k] = o["region"] if o["region"] not in (None, "_-1") else ""
        centres = region_centres(objs)
        z = np.load(os.path.join(CACHE, args.run, ep + ".npz"))
        rgb, depth_mm, sem = z["rgb"], z["depth_mm"], z["sem"].astype(np.int64)
        H, W = depth_mm.shape[1:]
        K = intrinsics(W, H)
        poses = [c2w_onecanvas(c, y) for c, y in zip(z["cam"], z["yaw"])]
        regmap = inst_region[np.clip(sem, 0, len(inst_region) - 1)]  # (N, H, W) region id per pixel, "" = none
        valid = (depth_mm > 100) & (depth_mm < 10000)
        feats = vlm.features(list(rgb)) if args.model == "onecanvas" else None
        small = [np.asarray(Image.fromarray(x).resize((B_RES, B_RES))) for x in rgb] if args.model in ("qwen25", "qwen3_B") else None
        panos, share, first_seen, frames_of = {}, collections.defaultdict(dict), {}, collections.defaultdict(list)
        fi, cache, labels_cache = 0, {}, {}
        for sid in nav:
            st = subtasks[sid]
            q = st["question"]
            gimg = z["goal_" + sid] if st["goal_type"] == "image" and ("goal_" + sid) in z else None
            qparts = [f"Subtask: {q}"] + (["Target image:", gimg] if gimg is not None else [])
            g_regions = sorted({objs[x]["region"] for x in st["goal_obj_ids"] if x in objs} - {None, "_-1"})
            rows = []
            for di, d in enumerate(decisions[sid]):
                t0 = time.time()
                while fi < len(z["step"]) and z["step"][fi] <= d["global_step"]:
                    m_all = valid[fi]
                    for r in set(regmap[fi][m_all].tolist()) - {""}:
                        m = m_all & (regmap[fi] == r)
                        if r not in centres:
                            continue
                        if r not in panos:
                            panos[r] = RegionPanorama(to_onecanvas(centres[r]), PANO_W, dev)
                        panos[r].add(rgb[fi], depth_mm[fi], m, poses[fi], K, inst=sem[fi])
                        share[r][fi] = m.sum() / m.size
                        frames_of[r].append(fi)
                        if share[r][fi] >= MIN_SHARE and r not in first_seen:
                            first_seen[r] = len(first_seen)
                    fi += 1
                regions = sorted(first_seen, key=first_seen.get)
                imgs = [panos[r].image() for r in regions] if args.model in ("qwen25", "qwen3") else None
                render_s = time.time() - t0
                state = tuple((r, len(frames_of[r])) for r in regions)
                row = {"run": args.run, "model": args.model, "subtask": sid, "decision": di,
                       "global_step": d["global_step"], "goal_type": st["goal_type"], "goal_regions": g_regions,
                       "goal_types": sorted({t for r in g_regions for t in types.get(r, [])}),
                       "regions": regions, "region_types": [types.get(r, []) for r in regions],
                       "goal_px": [panos[r].instance_pixels(st["goal_obj_ids"]) for r in regions],
                       "hole_fraction": [panos[r].hole_fraction() for r in regions],
                       "region_frames": [len(frames_of[r]) for r in regions], "kept_frames": fi,
                       "seen_frames": int((z["all_steps"] <= d["global_step"]).sum()), "render_s": render_s,
                       "baseline_choice": choice_type(d["choice"])}
                tq = time.time()
                if args.model in ("qwen25", "qwen3"):
                    key = ("multi", sid, state)
                    if key not in cache:
                        parts = list(qparts) + [INTRO_A]
                        for k, p in enumerate(imgs):
                            parts += [f"Panorama {k + 1}:", p]
                        cache[key] = vlm.generate(SYS, parts + [ASK_A]) if regions else ("Region: not seen yet", 0, 0)
                    rep, tok, ntok = cache[key]
                    pre = "A1" if args.model == "qwen25" else "C_multi"
                    row.update({f"{pre}_reply": rep, f"{pre}_choice": parse_region(rep, len(regions)),
                                f"{pre}_visible": parse_visible(rep), f"{pre}_tokens": tok, f"{pre}_image_tokens": ntok})
                if args.model in ("qwen25", "qwen3_B"):
                    pre = "B" if args.model == "qwen25" else "C_B"
                    labels, parts, bfr = [], list(qparts) + [INTRO_B], {}
                    for k, r in enumerate(regions):
                        fr = sorted(share[r], key=lambda n: (-share[r][n], n))[:B_FRAMES]
                        bfr[r] = fr
                        lk = (r, tuple(sorted(fr)))
                        if lk not in labels_cache:
                            labels_cache[lk] = vlm.generate("You describe rooms.", [small[n] for n in fr] + [ASK_LABEL], 12)[0]
                        labels.append(labels_cache[lk].strip().rstrip(".").lower()[:40])
                        parts += [f"<region {k + 1}: {labels[-1]}>"] + [small[n] for n in fr]
                    key = ("B", sid, tuple((r, tuple(bfr[r]), lab) for r, lab in zip(regions, labels)))
                    if key not in cache:
                        cache[key] = vlm.generate(SYS, parts + [ASK_B]) if regions else ("Region: not seen yet", 0, 0)
                    rep, tok, ntok = cache[key]
                    row.update({f"{pre}_labels": labels, f"{pre}_reply": rep, f"{pre}_choice": parse_region(rep, len(regions)),
                                f"{pre}_tokens": tok, f"{pre}_image_tokens": ntok})
                if args.model in ("qwen3", "onecanvas"):
                    pre = "C_pr" if args.model == "qwen3" else "A2"
                    ps, toks = [], []
                    for k, r in enumerate(regions):
                        key = ("room", sid, r, len(frames_of[r]))
                        if key not in cache:
                            cache[key] = (vlm.p_yes(SYS, list(qparts) + [PR_RGB, imgs[k], ASK_ROOM])[:2] if args.model == "qwen3"
                                          else canvas_p(vlm, f"{text_only(q)}\n{PR_CANVAS}\n{ASK_ROOM}", r, frames_of,
                                                        feats, depth_mm, regmap, poses, K, centres))
                        ps.append(cache[key][0])
                        toks.append(cache[key][1])
                    ch = pr_choice(ps)
                    vis = None
                    if ch != "none":
                        r = regions[ch]
                        key = ("vis", sid, r, len(frames_of[r]))
                        if key not in cache:
                            cache[key] = (vlm.p_yes(SYS, list(qparts) + [PR_RGB, imgs[ch], ASK_VIS])[:2] if args.model == "qwen3"
                                          else canvas_p(vlm, f"{text_only(q)}\n{PR_CANVAS}\n{ASK_VIS}", r, frames_of,
                                                        feats, depth_mm, regmap, poses, K, centres))
                        vis = cache[key][0] > 0.5
                    row.update({f"{pre}_p_room": ps, f"{pre}_choice": ch, f"{pre}_visible": vis, f"{pre}_tokens": toks})
                row["query_s"] = time.time() - tq
                rows.append(row)
                if make_examples and n_ex < N_EXAMPLES and len(regions) >= 3 and not any(
                        e.startswith(ep.split("_ep_")[0]) for e in os.listdir(os.path.join(OUT, "examples"))):
                    ex = os.path.join(OUT, "examples", f"{sid}_d{di}")
                    os.makedirs(ex, exist_ok=True)
                    for k, p in enumerate(imgs):
                        Image.fromarray(p).save(os.path.join(ex, f"A1_pano_{k + 1}_region{regions[k]}.png"))
                    for k, r in enumerate(regions):
                        Image.fromarray(np.concatenate([small[n] for n in bfr[r]], 1)).save(
                            os.path.join(ex, f"B_views_{k + 1}_region{r}.png"))
                    if gimg is not None:
                        Image.fromarray(gimg).save(os.path.join(ex, "goal_image.png"))
                    json.dump({**row, "question": q}, open(os.path.join(ex, "decision.json"), "w"), indent=1)
                    n_ex += 1
            for row in rows:  # written per subtask, so a resumed run never holds half a subtask
                f.write(json.dumps(row) + "\n")
            f.flush()
        print(f"{ep}: {len(first_seen)} regions, kept {fi}/{len(z['step'])} frames, {time.time() - t_ep:.0f}s", flush=True)
    f.close()


def canvas_p(vlm, question, r, frames_of, feats, depth_mm, regmap, poses, K, centres):
    """A2: P(yes) on region r's feature canvas, built from the kept views with pixels in r (others masked out)."""
    sel = frames_of[r]
    d = torch.as_tensor(np.where(regmap[sel] == r, depth_mm[sel], 0).astype(np.float32))
    n = len(sel)
    return vlm.p_yes(question, feats[sel], d, torch.as_tensor(np.stack([poses[i] for i in sel]), dtype=torch.float32),
                     torch.as_tensor(np.tile(K, (n, 1)), dtype=torch.float32),
                     torch.tensor([[d.shape[2], d.shape[1]]] * n), torch.as_tensor(to_onecanvas(centres[r]), dtype=torch.float32))


if __name__ == "__main__":
    main()

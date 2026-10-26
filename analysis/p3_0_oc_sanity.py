"""Does the A2 pipeline read a region canvas at all? Short questions to OneCanvas-Qwen3-VL-8B on two GT regions.

Same canvases as A2 in analysis/p3_0a.py (kept views of one episode, depth masked to the region, centre_override at the
region centre), but through OneCanvas's own `process_vision_and_generate` and short questions of the kind it was
trained on. If these come out right while A2's per-region P(yes) is flat, the weakness is the instruction format,
not the canvas pipeline.
Run (onecanvas env, repo root): CUDA_VISIBLE_DEVICES=0 python analysis/p3_0_oc_sanity.py > analysis/out_p3_0/onecanvas_sanity.txt
"""
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import analyze as A  # noqa: E402
import p3_0a as P  # noqa: E402
from rcmem.geometry import c2w_onecanvas, intrinsics, to_onecanvas  # noqa: E402

RUN, EP, REGIONS = "p2_E0_fix_s77", "00829-QaLdnwvtxbs_ep_0", {"_4": "living room", "_7": "bathroom"}
QUESTIONS = ["What type of room is this? Answer with one or two words.", "Is this a bathroom? Answer yes or no.",
             "Is this a living room? Answer yes or no."]


def main():
    vlm = P.OneCanvas(P.CKPT["onecanvas"])
    from inference.pipeline import process_vision_and_generate
    objs = A.load_records(os.path.join(REPO, "analysis/runs", RUN))[0][A.scene_of(EP)]
    z = np.load(os.path.join(P.CACHE, RUN, EP + ".npz"))
    ir = np.full(max(objs) + 2, "", dtype=object)
    for k, o in objs.items():
        ir[k] = o["region"] or ""
    reg, cen = ir[z["sem"].astype(np.int64)], P.region_centres(objs)
    K = intrinsics(*z["depth_mm"].shape[1:][::-1])
    ok = 0
    with tempfile.TemporaryDirectory() as tmp:
        for r, gt in REGIONS.items():
            sel = [i for i in range(len(reg)) if (reg[i] == r).any()]
            paths = []
            for i in sel:
                paths.append(os.path.join(tmp, f"{i}.png"))
                Image.fromarray(z["rgb"][i]).save(paths[-1])
            for q in QUESTIONS:
                a = process_vision_and_generate(
                    vlm.model, vlm.proc, q, image_paths=paths,
                    poses=[c2w_onecanvas(z["cam"][i], z["yaw"][i]) for i in sel],
                    depths=[torch.as_tensor(np.where(reg[i] == r, z["depth_mm"][i], 0).astype(np.float32)) for i in sel],
                    intrinsics=torch.as_tensor(np.tile(K, (len(sel), 1)), dtype=torch.float32),
                    image_dims=[torch.tensor(z["depth_mm"].shape[1:][::-1])] * len(sel), max_tokens=20,
                    center_override=to_onecanvas(cen[r])).strip().lower()
                want = gt if "type" in q else ("yes" if gt in q else "no")
                ok += want in a
                print(f"region {r} (GT {gt}, {len(sel)} views) | {q} -> {a} | expected {want}")
    print(f"{ok}/{len(REGIONS) * len(QUESTIONS)} correct")


if __name__ == "__main__":
    main()

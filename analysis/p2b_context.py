"""Shared pieces of the Phase 2b experiments (DEV_PLAN.md P2b): 3D voxel pruning of the episode's frames, the
context-as-memory decision prompt ("Frame f, Object o" commit format) and exact Qwen2.5-VL token counting.

Used by analysis/p2b_f0.py (offline token budget from existing run records) and analysis/p2b_experiments.py
(F1 / F2 / F2b live runs), so the pruning and the token count are the same code in both.
"""
import math

import numpy as np

# Qwen2.5-VL image tokenisation (qwen_vl_utils.smart_resize defaults): 14 px patches merged 2x2 -> 28 px per token
TOKEN_PX = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
PROMPT_PX = 360            # frames and frontiers enter the prompt at the evaluation resolution (cfg.prompt_h/w)
HFOV = 120.0               # cfg.hfov
MAX_DEPTH = 6.0            # m; farther depth is not counted as covered space
DEPTH_STRIDE = 5           # 1280 px depth -> 256 x 256 samples for the voxel sets
NOVELTY = 0.25             # frame-level pruning: keep a frame if >= 25% of its voxels are not covered yet


def smart_resize(h, w, factor=None, min_pixels=None, max_pixels=None):
    # defaults read TOKEN_PX at call time: p2b_experiments sets it to 32 for Qwen3-VL (16 px patches merged 2x2)
    factor = factor or TOKEN_PX
    min_pixels, max_pixels = min_pixels or 4 * factor * factor, max_pixels or 16384 * factor * factor
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def image_tokens(h, w):
    """LM tokens of one image, without the <|vision_start|>/<|vision_end|> pair."""
    hb, wb = smart_resize(h, w)
    return (hb // TOKEN_PX) * (wb // TOKEN_PX)


FRAME_GRID = smart_resize(PROMPT_PX, PROMPT_PX)[0] // TOKEN_PX   # 13 -> 169 tokens per frame
FRAME_TOKENS = FRAME_GRID * FRAME_GRID


# ---------------------------------------------------------------- 3D back-projection and voxel keys
def backproject(depth, cam_pose, hfov=HFOV, stride=DEPTH_STRIDE, max_depth=MAX_DEPTH):
    """Depth image (z-depth, metres) + 4x4 habitat camera-to-world pose -> (N, 3) world points and the (h, w) grid
    of sampled depths. Habitat cameras look along -z with y up."""
    H, W = depth.shape[:2]
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    off = stride // 2
    d = depth[off::stride, off::stride].astype(np.float32)
    v, u = np.mgrid[off:H:stride, off:W:stride].astype(np.float32)
    x = (u + 0.5 - W / 2.0) / f * d
    y = -(v + 0.5 - H / 2.0) / f * d
    p = np.stack([x, y, -d], -1)
    ok = (d > 0.05) & (d < max_depth)
    R, t = np.asarray(cam_pose)[:3, :3], np.asarray(cam_pose)[:3, 3]
    return p[ok] @ R.T + t, d


def voxel_keys(points, voxel):
    """Unique int64 keys of the voxels containing the points."""
    if len(points) == 0:
        return np.zeros(0, np.int64)
    q = np.floor(points / voxel).astype(np.int64) + (1 << 20)
    return np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2])


def token_points(depth, cam_pose, hfov=HFOV, grid=FRAME_GRID, max_depth=MAX_DEPTH):
    """One 3D point per LM visual token: the median depth of the token's image cell, on the cell-centre ray.
    Returns (grid*grid, 3) world points and a validity mask."""
    H, W = depth.shape[:2]
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    ys, xs = np.linspace(0, H, grid + 1).astype(int), np.linspace(0, W, grid + 1).astype(int)
    pts, ok = np.zeros((grid * grid, 3), np.float32), np.zeros(grid * grid, bool)
    R, t = np.asarray(cam_pose)[:3, :3], np.asarray(cam_pose)[:3, 3]
    for i in range(grid):
        for j in range(grid):
            cell = depth[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            cell = cell[(cell > 0.05) & (cell < max_depth)]
            if cell.size == 0:
                continue
            d = float(np.median(cell))
            u, v = (xs[j] + xs[j + 1]) / 2.0, (ys[i] + ys[i + 1]) / 2.0
            p = np.array([(u - W / 2.0) / f * d, -(v - H / 2.0) / f * d, -d])
            pts[i * grid + j], ok[i * grid + j] = p @ R.T + t, True
    return pts, ok


class FramePruner:
    """Frame-level voxel pruning: a frame is kept if at least `novelty` of its voxels are not covered by the frames
    kept so far. Online and order-dependent, so a frame's fate never depends on later frames."""

    def __init__(self, voxel, novelty=NOVELTY):
        self.voxel, self.novelty, self.covered = voxel, novelty, set()

    def add(self, points):
        keys = voxel_keys(points, self.voxel)
        if keys.size == 0:
            return False, 0.0
        new = [k for k in keys.tolist() if k not in self.covered]
        frac = len(new) / keys.size
        if frac >= self.novelty:
            self.covered.update(new)
            return True, frac
        return False, frac


class TokenPruner:
    """Token-level voxel pruning (StreamVLN-style): a visual token is kept if its 3D point falls in a voxel that no
    kept token covers yet. Returns the number of tokens kept for the frame."""

    def __init__(self, voxel):
        self.voxel, self.covered = voxel, set()

    def add(self, pts, ok):
        if not ok.any():
            return 0
        keys = voxel_keys(pts[ok], self.voxel)  # unique voxels of this frame's tokens
        new = [k for k in keys.tolist() if k not in self.covered]
        self.covered.update(new)
        # tokens whose voxel was new (several tokens of one frame can share a voxel: keep one per voxel)
        return len(new)


# ---------------------------------------------------------------- the context-as-memory decision prompt
NAV_SYS = (
    "You are an embodied agent exploring an indoor environment.\n"
    "Your goal is to analyze the frames you have observed so far in this episode (Frames) and unexplored regions (Frontiers) "
    "to determine the best next step to locate or find the target object described in the task instruction.\n"
    "RULES:\n"
    "1) Your whole reply MUST contain exactly two sections in this order:\n"
    "  A THOUGHT section that begins with 'THOUGHT:' (uppercase) followed by your reasoning.\n"
    "  A single ANSWER line that begins with 'ANSWER:' (uppercase) and provides exactly ONE decision.\n"
    "2) THOUGHT content rules (keep it short, at most 10 lines):\n"
    "  Mention only the Frames that show an object that could be the target, with the object number.\n"
    "  For each Frontier i, describe its spatial direction or likelihood in a few words.\n"
    "3) The ANSWER line MUST be one of these exact formats:\n"
    "  ANSWER: Frame i, Object j\n"
    "  ANSWER: Frontier i\n"
    "  where i and j are plain integers.\n"
    "  IMPORTANT: When choosing a Frame, you MUST specify both the frame index and the object index.\n"
    "4) Do NOT invent or hallucinate new frame or object indices.\n"
    "5) After analyzing all options, select the Frame object that is the target, or the Frontier that maximizes discovery.\n"
)

NAV_DEF = (
    "Definitions:\n"
    "Frame: An image you observed earlier in this episode. Frames are listed in the order you saw them. "
    "The detected objects in a frame are marked in the image with numbered boxes and listed below the frame as 'Object j: name'. "
    "Choosing a frame means that the object asked in the task is visible in that frame, and you will choose that object as the result of task instruction. "
    "Therefore, if you choose a frame, you should also choose the object in the frame that you think is the result of task completion.\n"
    "Frontier: An unexplored region that could provide new information or a closer view of the target object. Selecting a frontier means that you will further explore that direction.\n"
    "Indices start at 0. Always reference indices. Do NOT write names instead of indices.\n"
)

NAV_EX1 = (
    "Example 1 — choose a Frontier:\n"
    "Task Instruction: Find the red kettle and bring it to the kitchen counter.\n"
    "Frames available:\n"
    "Frame 0: [living room with sofa, table, lamp]\n"
    "  Object 0: sofa\n"
    "  Object 1: table\n"
    "Frame 1: [partial kitchen with counter and a blue kettle]\n"
    "  Object 0: counter\n"
    "  Object 1: kettle\n"
    "Frontiers available:\n"
    "Frontier 0: doorway leading deeper into kitchen\n"
    "Frontier 1: corridor to another room\n"
    "THOUGHT:\n"
    "- Frame 1, Object 1: a kettle, but it is blue, not the target.\n"
    "- Frontier 0: leads deeper into the kitchen, likely toward locations where a red kettle could be found.\n"
    "- Frontier 1: leads to another room, less relevant.\n"
    "Decision: Frontier 0 maximizes the chance to find the target object.\n"
    "ANSWER: Frontier 0\n"
)

NAV_EX2 = (
    "Example 2 — choose a Frame object (correct strict format):\n"
    "Task Instruction: Locate the refrigerator in the environment.\n"
    "Frames available:\n"
    "Frame 0: [kitchen with large white refrigerator beside a counter]\n"
    "  Object 0: counter\n"
    "  Object 1: refrigerator\n"
    "Frame 1: [living room with sofa and TV, no refrigerator visible]\n"
    "  Object 0: sofa\n"
    "  Object 1: television\n"
    "Frontiers available:\n"
    "Frontier 0: hallway leading to the living room\n"
    "Frontier 1: door leading outside\n"
    "THOUGHT:\n"
    "- Frame 0, Object 1: the refrigerator — strong match.\n"
    "- Frontier 0: leads to living room already explored.\n"
    "- Frontier 1: leads outside, irrelevant.\n"
    "Decision: Frame 0, Object 1 is the correct match.\n"
    "ANSWER: Frame 0, Object 1\n"
)


def nav_prompt(question, image_goal, frames, frontier_imgs):
    """frames: list of (frame_index, image, [object names]) in viewing order; images are base64 strings (or any
    placeholder when only counting tokens). Same structure as LMEE's format_explore_prompt_object, Memory -> Frame,
    no per-object crops (the objects are the numbered boxes drawn in the frame)."""
    content = [(NAV_DEF,), (NAV_EX1,), (NAV_EX2,)]
    task_text = f"Task Instruction: {question}"
    if image_goal is not None:
        content.append((f"{task_text}\nImage goal:", image_goal))
        content.append(("\n",))
    else:
        content.append((task_text + "\n",))
    content.append(("Select the Frame object or Frontier that would help complete the task.\n",))
    content.append(("The followings are all the Frames you have observed (in viewing order) that you can choose:\n",))
    if not frames:
        content.append(("No Frame is available.\n",))
    for fi, img, names in frames:
        content.append((f"Frame {fi}: ", img))
        content.append(("".join(f"Object {j}: {n}\n" for j, n in enumerate(names)) if names else "No detected object.\n",))
    content.append(("The followings are all the Frontiers that you can explore:\n",))
    if not frontier_imgs:
        content.append(("No Frontier is available.\n",))
    for i, img in enumerate(frontier_imgs):
        content.append((f"Frontier {i}: ", img))
        content.append(("\n",))
    content.append(("You MUST output the THOUGHT section followed by ONE valid ANSWER line:\n"
                    "   ANSWER: Frame i, Object j  (when selecting a specific detected object)\n"
                    "   ANSWER: Frontier i  (when deciding to explore a new region)\n",))
    return NAV_SYS, content


QA_SYS = (
    "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and recalling the frames you observed.\n"
    "Definitions:\n"
    "Frame: An image you observed earlier in this episode; frames are listed in the order you saw them. Choosing a Frame means that this frame contains enough information for you to answer the question. "
    "If you choose a Frame, you need to directly give an answer to the question. "
    "If you don't have enough information to give an answer, then state that the information is insufficient.\n"
)


def qa_prompt(question, image_goal, frames, ego_img=None):
    """Same structure as LMEE's format_explore_prompt_qa (choice answers), Memory -> Frame."""
    content = []
    if image_goal is not None:
        content += [(f"Question: {question}", image_goal), ("\n",)]
    else:
        content.append((f"Question: {question}\n",))
    content.append(("Select the Frame that would help you answer the question.\n",))
    if ego_img is not None:
        content += [("The following is the egocentric view of the agent in forward direction: ", ego_img), ("\n",)]
    content.append(("The followings are all the Frames you have observed (in viewing order) that you can choose (followed with contained object classes):\n"
                    "Please note that the contained classes may not be accurate (wrong or missing classes) due to the limitation of the object detection model. "
                    "So you still need to utilize the images to make decisions.\n",))
    if not frames:
        content.append(("No Frame is available.\n",))
    for fi, img, names in frames:
        content.append((f"Frame {fi}: ", img))
        if names:
            content.append((f"Objects: {', '.join(names)}\n",))
    content.append(("Please provide your answer in the following format: "
                    "'Frame i\n[Answer]', where i is the index of the frame you choose. "
                    "[Answer] must be one of the options provided in the question (use the letter corresponding to your choice, e.g., A, B, C, D).\n",))
    return QA_SYS, content


def to_messages(sys_prompt, content, image_url=lambda x: f"data:image;base64,{x}"):
    """Same message layout as LMEE's call_qwen25 (one user message per content item)."""
    msgs = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    for c in content:
        part = [{"type": "text", "text": c[0]}]
        if len(c) == 2:
            part.append({"type": "image", "image": image_url(c[1])})
        msgs.append({"role": "user", "content": part})
    return msgs


class TokenCounter:
    """Exact prompt length without encoding images: tokenised chat text + smart_resize token counts per image."""

    def __init__(self, processor):
        self.p = processor
        self.pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def count(self, sys_prompt, content, image_hw):
        """image_hw: dict id(image object) or image -> (h, w), looked up by the content's image entries in order."""
        msgs = to_messages(sys_prompt, content, image_url=lambda x: "x")
        text = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n = len(self.p.tokenizer(text)["input_ids"])  # each image is one <|image_pad|> here
        for c in content:
            if len(c) == 2:
                h, w = image_hw(c[1])
                n += image_tokens(h, w) - 1
        return n

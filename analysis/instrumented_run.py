"""Re-run the MemoryExplorer baseline on the 58-task subset with extra ground-truth logging.

LMEE evaluation code is NOT modified: this script imports it and wraps a few functions at
runtime to record, per subtask, (1) which HM3D instances are visible in every stored frame
(semantic sensor), (2) the full CLIP ranking of memories at every retrieval, (3) the memory
list, its detected objects and the model's choice, (4) the subtask outcome.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=0 python analysis/instrumented_run.py --start_ratio 0 --end_ratio 0.25 --split 1
Output: analysis/runs/instr_choice_sub/records_<start>_<end>.jsonl (+ the usual LMEE result files).
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO, "evaluation")
sys.path.insert(0, EVAL_DIR)
os.chdir(EVAL_DIR)  # LMEE configs use paths relative to evaluation/

import run_lmee  # noqa: E402
from src import eval_utils_qwen_lmee as eu  # noqa: E402
from src.scene import Scene  # noqa: E402
from src.logger_lmeebench import Logger  # noqa: E402
from src.tsdf_planner import SnapShot, Frontier  # noqa: E402

MIN_PIX = 200  # a GT instance counts as visible in a 1280x1280 frame if it covers >= 200 px (~0.012%)
CUR = {"episode": None, "subtask": None, "frames": {}, "queries": [], "scene_dumped": set()}
OUT = None


def emit(rec):
    OUT.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    OUT.flush()


# ---- scene: dump GT semantic objects once per scene ----
_orig_scene_init = Scene.__init__


def scene_init(self, scene_id, *a, **k):
    _orig_scene_init(self, scene_id, *a, **k)
    self._gap_scene_id = scene_id
    if scene_id in CUR["scene_dumped"]:
        return
    CUR["scene_dumped"].add(scene_id)
    objs = []
    for o in self.simulator.semantic_scene.objects:
        if o is None or o.category is None:
            continue
        objs.append({
            "sid": int(o.semantic_id),
            "id": o.id,
            "cat": o.category.name(),
            "center": np.array(o.aabb.center).tolist(),
            "size": np.array(o.aabb.sizes).tolist(),
            "region": o.region.id if o.region is not None else None,
        })
    emit({"type": "scene", "scene_id": scene_id, "objects": objs})


Scene.__init__ = scene_init

# ---- per frame: GT instance visibility ----
_orig_usg = Scene.update_scene_graph


def update_scene_graph(self, *a, **k):
    sem = k.get("semantic_obs")
    if sem is not None:
        ids, cnt = np.unique(sem, return_counts=True)
        vis = {int(i): int(c) for i, c in zip(ids, cnt) if c >= MIN_PIX and i != 0}
        CUR["frames"][k["img_path"]] = {
            "vis": vis,
            "pts": np.asarray(k["pts"]).tolist(),
            "cam": np.asarray(k["cam_pos"])[:3, 3].tolist(),
            "fwd": (-np.asarray(k["cam_pos"])[:3, 2]).tolist(),
        }
    return _orig_usg(self, *a, **k)


Scene.update_scene_graph = update_scene_graph

# ---- retrieval: record full rankings (text index and image index separately) ----
_kb = eu.KnowledgeBase


def _full_rank(index, qv):
    n = index.ntotal
    if n == 0:
        return [], []
    d, i = index.search(qv, n)
    return i[0].tolist(), d[0].tolist()


def _text_vec(self, text):
    t = self.preprocess(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
    return eu.l2norm(self.text_embedder.get_text_features(**t).cpu().detach().numpy())


_orig_t2p = _kb.search_text_to_pair


def search_text_to_pair(self, query_text, top_k=5):
    res = _orig_t2p(self, query_text, top_k)
    qv = _text_vec(self, query_text)
    ti, td = _full_rank(self.text_index, qv)
    ii, idd = _full_rank(self.image_index, qv)
    names = [x["rgb_id"] for x in self.data]
    CUR["queries"].append({"query": query_text, "mode": "text", "top_k": top_k,
                           "text_rank": [names[i] for i in ti], "text_dist": td,
                           "image_rank": [names[i] for i in ii], "image_dist": idd,
                           "index_aligned": self.text_index.ntotal == self.image_index.ntotal == len(names),
                           "returned": [r["rgb_id"] for r in res[0]]})
    return res


_orig_i2p = _kb.search_image_to_pair


def search_image_to_pair(self, query_text, query_image, top_k=5):
    res = _orig_i2p(self, query_text, query_image, top_k)
    t = self.preprocess(images=query_image, return_tensors="pt").to(self.device)
    qi = eu.l2norm(self.image_model.get_image_features(**t).cpu().detach().numpy())
    qt = _text_vec(self, query_text)
    ti, td = _full_rank(self.text_index, qt)
    ii, idd = _full_rank(self.image_index, qi)
    names = [x["rgb_id"] for x in self.data]
    CUR["queries"].append({"query": query_text, "mode": "image", "top_k": top_k,
                           "text_rank": [names[i] for i in ti], "text_dist": td,
                           "image_rank": [names[i] for i in ii], "image_dist": idd,
                           "index_aligned": self.text_index.ntotal == self.image_index.ntotal == len(names),
                           "returned": [r["rgb_id"] for r in res[0]]})
    return res


_kb.search_text_to_pair = search_text_to_pair
_kb.search_image_to_pair = search_image_to_pair

# ---- each MLLM decision: memory list, detected objects, choice ----
_orig_q = run_lmee.query_mllm_for_response


def query_mllm_for_response(subtask_metadata, scene, tsdf_planner, *a, **k):
    nq = len(CUR["queries"])
    out = _orig_q(subtask_metadata, scene, tsdf_planner, *a, **k)
    caller = inspect.currentframe().f_back.f_locals
    choice = None
    if out is not None:
        c = out[0]
        if isinstance(c, SnapShot):
            choice = {"type": "memory", "image": c.image, "cluster": [int(x) for x in c.cluster]}
        elif isinstance(c, Frontier):
            choice = {"type": "frontier", "image": c.image, "position": np.asarray(c.position).tolist()}
    emit({
        "type": "decision",
        "episode": CUR["episode"], "subtask": CUR["subtask"],
        "cnt_step": caller.get("cnt_step"), "global_step": caller.get("global_step"),
        "pts": np.asarray(caller.get("pts")).tolist(),
        "goal_obj_ids_mapping": {str(g): [int(x) for x in v] for g, v in caller.get("goal_obj_ids_mapping", {}).items()},
        "snapshots": {rid: [int(x) for x in s.cluster] for rid, s in scene.snapshots.items()},
        "objects": {int(oid): {"cls": o["class_name"], "center": np.asarray(o["bbox"].center).tolist(),
                               "image": o.get("image")} for oid, o in scene.objects.items()},
        "n_frontiers": len(tsdf_planner.frontiers),
        "retrieval": CUR["queries"][nq:],
        "choice": choice,
        "pred_answer": out[1] if (out is not None and subtask_metadata["task_type"] == "QA") else None,
    })
    return out


run_lmee.query_mllm_for_response = query_mllm_for_response

# ---- episode / subtask bookkeeping ----
_orig_ie = Logger.init_episode


def init_episode(self, episode_id):
    CUR["episode"] = episode_id
    CUR["frames"] = {}
    return _orig_ie(self, episode_id)


Logger.init_episode = init_episode

_orig_is = Logger.init_subtask


def init_subtask(self, subtask_id, *a, **k):
    CUR["subtask"] = subtask_id
    CUR["queries"] = []
    return _orig_is(self, subtask_id, *a, **k)


Logger.init_subtask = init_subtask

_orig_lsr = Logger.log_subtask_result


def log_subtask_result(self, **k):
    md = k["subtask_metadata"]
    caller = inspect.currentframe().f_back.f_locals
    emit({
        "type": "subtask",
        "episode": CUR["episode"], "subtask": k["subtask_id"], "goal_type": k["goal_type"],
        "difficulty": md["difficulty"], "question": md["question"], "class": md["class"],
        "goal_obj_ids": md["goal_obj_ids"], "viewpoints": md["viewpoints"],
        "QA": md.get("QA"), "pred_answer": k["pred_answer"],
        "success_by_snapshot": k["success_by_snapshot"], "success_by_distance": k["success_by_distance"],
        "task_step": k["task_step"], "num_step": caller.get("num_step"),
        "final_pts": np.asarray(caller.get("pts")).tolist(),
        "explore_dist": self.subtask_explore_dist, "gt_dist": k["gt_subtask_explore_dist"],
        "goal_obj_ids_mapping": {str(g): [int(x) for x in v] for g, v in caller.get("goal_obj_ids_mapping", {}).items()},
        "frames": CUR["frames"],  # cumulative within the episode (memory persists across subtasks)
    })
    return _orig_lsr(self, **k)


Logger.log_subtask_result = log_subtask_result


if __name__ == "__main__":
    import logging
    import time
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", default=os.path.join(REPO, "analysis/cfg/instr_choice_sub.yaml"))
    p.add_argument("--start_ratio", type=float, default=0.0)
    p.add_argument("--end_ratio", type=float, default=1.0)
    p.add_argument("--split", type=int, default=1)
    args = p.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    OUT = open(os.path.join(cfg.output_dir, f"records_{args.start_ratio}_{args.end_ratio}.jsonl"), "a")
    t0 = time.time()

    class F(logging.Formatter):
        def formatTime(self, r, datefmt=None):
            s = int(r.created - t0)
            return f"{s // 3600:02}:{s % 3600 // 60:02}:{s % 60:02}"

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[
        logging.FileHandler(os.path.join(cfg.output_dir, f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log"), mode="a"),
        logging.StreamHandler()])
    for h in logging.getLogger().handlers:
        h.setFormatter(F(fmt="%(asctime)s - %(message)s"))
    model, processor = run_lmee.load_model(cfg)
    run_lmee.main(cfg, model, processor, start_ratio=args.start_ratio, end_ratio=args.end_ratio,
                  split=args.split, answer_type="choice")

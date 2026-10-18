"""Run the MemoryExplorer baseline on a GOAT-Bench val_unseen subset with the same ground-truth logging as
instrumented_run.py (LMEE code is wrapped at runtime, not modified; no GOAT evaluator ships with this repo).

GOAT protocol as followed here: each episode's tasks are given one at a time, in order, memory persists across
the episode; 'object' goals succeed at any instance of the category, 'description'/'image' goals at one instance.
Deviations from the official GOAT evaluator (UNVERIFIED equivalence): question templates are LMEE's (object goals:
"Can you find the {category}?", 3D-Mem style), there is no compound task instruction, image goals are rendered by
LMEE from the goal's first viewpoint instead of GOAT's image_goals camera, success = within 1 m of a goal viewpoint.

Usage (repo root):  CUDA_VISIBLE_DEVICES=1 python analysis/goat_runner.py --episodes 0 --start_ratio 0 --end_ratio 0.5
Output: analysis/runs/goat_ep0/records_<start>_<end>.jsonl + LMEE result files.
"""
import argparse
import builtins
import copy
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import instrumented_run as ir  # noqa: E402  (applies the logging wrappers on import)

GOAT = os.path.join(REPO, "data/goat_bench/val_unseen/content")


def convert(episode_ids, out_dir):
    """Write one LMEE-format file per scene holding the chosen GOAT episodes (tasks kept verbatim)."""
    os.makedirs(out_dir, exist_ok=True)
    hm3d = os.listdir(os.path.join(REPO, "data/hm3d-0.2/hm3d/val")) + os.listdir(os.path.join(REPO, "data/hm3d-0.2/hm3d/train"))
    for f in sorted(os.listdir(GOAT)):
        sid = f.split(".")[0]
        full = next(s for s in hm3d if s.endswith(sid))
        eps = json.load(open(os.path.join(GOAT, f)))["episodes"]
        out = [{"Task instruction": "", "Difficulty": "goat", "Object_id": [], "QAs": [],
                "Start_position": [eps[i]["start_position"]], "Start_rotation": [eps[i]["start_rotation"]],
                "goat_episode_id": eps[i]["episode_id"], "goat_tasks": eps[i]["tasks"]} for i in episode_ids if i < len(eps)]
        json.dump(out, open(os.path.join(out_dir, full + ".json"), "w"))


def prepare_goat_goals(scene_name, episode, all_navigation_goals):
    """GOAT replacement for prepare_lmeebench_navigation_goals: one subtask per GOAT task, in order."""
    by_id = {x["object_id"]: x for v in all_navigation_goals.values() for x in v}
    types, goals = [], []
    for task in episode["goat_tasks"]:
        cat, mode, oid = task[0], task[1], task[2]
        if mode == "object":
            insts = all_navigation_goals.get(f"{scene_name}.basis.glb_{cat}", [])
            if not insts:
                continue
            goals.append(copy.deepcopy(insts))
        else:
            if oid not in by_id:
                continue
            goals.append([copy.deepcopy(by_id[oid])])
        types.append(mode)
    return types, goals


_orig_init_subtask = ir.Logger.init_subtask


def init_subtask(self, subtask_id, goal_type, *a, **k):
    md = _orig_init_subtask(self, subtask_id, goal_type, *a, **k)
    if goal_type == "object":
        md["question"] = f"Can you find the {md['class']}?"
    return md


ir.Logger.init_subtask = init_subtask
ir.run_lmee.prepare_lmeebench_navigation_goals = prepare_goat_goals
# run_lmee skips an episode when (#finished subtasks) >= (#QA subtasks); GOAT has no QA, so make the threshold 1.
# sum() is used only on that line of run_lmee.py (line 111).
ir.run_lmee.sum = lambda it: builtins.sum(it) + 1

if __name__ == "__main__":
    import logging
    import time
    from omegaconf import OmegaConf

    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", default=os.path.join(REPO, "analysis/cfg/instr_choice_sub.yaml"))
    p.add_argument("--episodes", default="0", help="comma-separated GOAT episode indices per scene")
    p.add_argument("--start_ratio", type=float, default=0.0)
    p.add_argument("--end_ratio", type=float, default=1.0)
    args = p.parse_args()
    eps = [int(x) for x in args.episodes.split(",")]
    tag = "goat_ep" + "_".join(map(str, eps))
    data_dir = os.path.join(REPO, "analysis/runs", tag + "_data")
    convert(eps, data_dir)
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    cfg.test_data_dir = data_dir
    cfg.exp_name = tag
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
    model, processor = ir.run_lmee.load_model(cfg)
    ir.run_lmee.main(cfg, model, processor, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=1, answer_type="choice")

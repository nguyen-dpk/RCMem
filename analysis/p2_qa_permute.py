"""Phase 2: choice-permutation evaluation of LMEE memory QA (prior correction b).

Replays only the answering step of each QA subtask of a recorded run: the memories the run retrieved for that
question (recorded order), re-rendered at 1280 px from the recorded poses and resized like the run (360 px), the
agent's egocentric view, and LMEE's own QA prompt (format_explore_prompt_qa, choice mode) and parsing. The choices
are presented (0) in the original order, to measure replay fidelity against the recorded answers, then (1..K) in
K random orders (seeded); accuracy under permutation is averaged over the K orders.
Run (repo root): CUDA_VISIBLE_DEVICES=0 python analysis/p2_qa_permute.py --run p2_E0_fix_s77 --perms 3
Output: analysis/out_p2/qa_permute_<run>.json
"""
import argparse
import collections
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import analyze as A  # noqa: E402
import gt_question_views as G  # noqa: E402
import habitat_sim  # noqa: E402
from src import eval_utils_qwen_lmee as eu  # noqa: E402
from src.habitat import make_semantic_cfg  # noqa: E402
from src.utils import resize_image  # noqa: E402


def make_sim(scene_full):
    split = "train" if int(scene_full.split("-")[0]) < 800 else "val"
    sid = scene_full.split("-")[1]
    settings = {"scene": os.path.join(G.HM3D, split, scene_full, sid + ".basis.glb"), "default_agent": 0,
                "sensor_height": 1.5, "width": 1280, "height": 1280, "hfov": 120,
                "scene_dataset_config_file": os.path.join(REPO, "data/hm3d_annotated_basis.scene_dataset_config.json"),
                "camera_tilt": -30 * np.pi / 180}
    sim = habitat_sim.Simulator(make_semantic_cfg(settings))
    return sim, sim.initialize_agent(0)


def render_frame(sim, agent, fr):
    fx, _, fz = fr["fwd"]
    rgb, _ = G.render(agent, sim, fr["pts"], G.yaw_q(float(np.arctan2(-fx, -fz))))
    return resize_image(rgb, 360, 360)


def ask(model, processor, question, ego, snaps, classes):
    sys_prompt, content = eu.format_explore_prompt_qa(question, ego, snaps, classes, egocentric_view=True,
                                                      use_snapshot_class=True, image_goal=None, answer_type="choice")
    out = (eu.call_qwen25(sys_prompt, content, model, processor) or "").strip()
    if "\n" in out:  # same parsing as explore_step_qa: first line "memory i", last line the answer
        lines = out.split("\n")
        return lines[-1].strip(), lines[0].strip().lower()
    return "", out.lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--perms", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(os.path.join(REPO, "data/MemoryExplorer"), torch_dtype="bfloat16", device_map="cuda:0").eval()
    processor = AutoProcessor.from_pretrained(os.path.join(REPO, "data/MemoryExplorer"))
    scenes, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", args.run))
    rng = random.Random(args.seed)
    items = []
    by_scene = collections.defaultdict(list)
    for sid, st in subtasks.items():
        if st["goal_type"] == "QA" and decisions.get(sid):
            by_scene[A.scene_of(sid)].append(sid)
    for scene, sids in sorted(by_scene.items()):
        sim, agent = make_sim(scene)
        for sid in sids:
            st, d = subtasks[sid], decisions[sid][0]
            qa, frames = st["QA"], st["frames"]
            returned = []
            for q in d["retrieval"]:
                returned += [x for x in q["returned"] if x not in returned]
            snaps = [r for r in returned if r in frames and r in d["snapshots"]]
            if not snaps:
                continue
            imgs = {r: eu.encode_tensor2base64(render_frame(sim, agent, frames[r])) for r in snaps}
            classes = {r: [d["objects"][str(o)]["cls"] for o in d["snapshots"][r] if str(o) in d["objects"]] for r in snaps}
            gstep = int(d["global_step"])
            views = sorted((n for n in frames if n.startswith(f"{gstep}-view_")), key=lambda n: int(n.split("_")[-1].split(".")[0]))
            ego = [eu.encode_tensor2base64(render_frame(sim, agent, frames[views[-1]]))] if views else []
            letters = sorted(qa["choices"])
            texts = [qa["choices"][L] for L in letters]
            correct_text = qa["choices"][qa["answer"]]
            res = {"subtask": sid, "category": qa["category"], "n_choices": len(letters), "recorded_pred": st["pred_answer"],
                   "answer": qa["answer"], "runs": []}
            orders = [list(range(len(texts)))] + [rng.sample(range(len(texts)), len(texts)) for _ in range(args.perms)]
            for k, order in enumerate(orders):
                shown = {letters[i]: texts[j] for i, j in enumerate(order)}
                gold = next(L for L, t in shown.items() if t == correct_text)
                question = f"{qa['question']} " + ", ".join(f"{L}: {t}" for L, t in shown.items())
                pred, mem = ask(model, processor, question, ego, imgs, classes)
                res["runs"].append({"perm": k, "order": order, "gold": gold, "pred": pred, "correct": pred == gold,
                                    "pred_text": shown.get(pred), "memory": mem})
            items.append(res)
            print(sid, [(r["pred"], r["gold"]) for r in res["runs"]], flush=True)
        sim.close()
    n = len(items)
    ident = [it["runs"][0] for it in items]
    perm_acc = [np.mean([r["correct"] for r in it["runs"][1:]]) for it in items]
    pred_letter = collections.Counter(r["pred"] for it in items for r in it["runs"][1:])
    summary = {"run": args.run, "questions_replayed": n,
               "replay_fidelity_pred_agreement": A.pct(sum(r["pred"] == it["recorded_pred"] for r, it in zip(ident, items)), n),
               "recorded_accuracy_on_replayed": A.pct(sum(it["recorded_pred"] == it["answer"] for it in items), n),
               "replay_accuracy_original_order": A.pct(sum(r["correct"] for r in ident), n),
               "permuted_accuracy_mean_over_perms": round(100 * float(np.mean(perm_acc)), 2),
               "permuted_accuracy_per_perm": [round(100 * float(np.mean([it["runs"][k]["correct"] for it in items])), 2) for k in range(1, args.perms + 1)],
               "answer_consistent_across_all_orders": A.pct(sum(len({r["pred_text"] for r in it["runs"]}) == 1 for it in items), n),
               "pred_letter_dist_permuted": dict(pred_letter),
               "permuted_accuracy_by_category": {c: round(100 * float(np.mean([a for a, it in zip(perm_acc, items) if it["category"] == c])), 1)
                                                 for c in sorted({it["category"] for it in items})}}
    os.makedirs(os.path.join(REPO, "analysis/out_p2"), exist_ok=True)
    json.dump({"summary": summary, "items": items}, open(os.path.join(REPO, f"analysis/out_p2/qa_permute_{args.run}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

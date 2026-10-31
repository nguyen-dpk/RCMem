"""P3: choice-permutation QA for RCMem (the Phase 2 check, analysis/p2_qa_permute.py, applied to an RCMem run).

Replays only the answering step of each QA subtask of a recorded RCMem run (QA reads all kept frames, as F1): the frames
that were in the QA context (decision record's `snapshots`, in viewing order), re-rendered at 1280 px from the recorded
poses and resized to 360 px, the agent's egocentric view at that step, and the run's own QA prompt (p2b_context.qa_prompt)
and parsing (first line "Frame i", last line the answer letter; correct = exact letter match, as analysis/analyze.py).
Difference from the run: the detector's boxes are not drawn (their coordinates are not logged); the object class list
under each frame is kept. The choices are presented (0) in the original order, to measure replay fidelity against the
recorded answers, then (1..K) in K random orders drawn exactly as Phase 2 did (same seed and loop order), so the
permutations match the LMEE baseline's replay question for question.
Run (repo root, lmee env): CUDA_VISIBLE_DEVICES=2,3 python analysis/p3_qa_permute.py --run p3_RP_s77 --perms 3
Output: analysis/out_p3/qa_permute_<run>.json
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
import p2_qa_permute as Q  # noqa: E402  (make_sim, render_frame: Phase 2's re-rendering)
import p2b_context as C  # noqa: E402
import p2b_experiments as F  # noqa: E402  (call_vlm: Qwen3-aware)
from src import eval_utils_qwen_lmee as eu  # noqa: E402


def parse(reply):
    lines = [l.strip() for l in reply.strip().split("\n") if l.strip()]
    return (lines[-1] if len(lines) > 1 else ""), (lines[0].lower() if lines else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="p3_RP_s77")
    p.add_argument("--vlm", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--perms", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(args.vlm, torch_dtype="bfloat16", device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(args.vlm)
    C.TOKEN_PX = 32 if "qwen3" in model.config.model_type else 28
    scenes, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", args.run))
    rng = random.Random(args.seed)
    items = []
    by_scene = collections.defaultdict(list)
    for sid, st in subtasks.items():
        if st["goal_type"] == "QA" and decisions.get(sid):
            by_scene[A.scene_of(sid)].append(sid)
    for scene, sids in sorted(by_scene.items()):
        sim, agent = Q.make_sim(scene)
        for sid in sids:
            st, d = subtasks[sid], decisions[sid][0]
            qa, frames = st["QA"], st["frames"]
            ctx = [n for n in d["snapshots"] if n in frames]
            if not ctx:
                continue
            fr = [(i, eu.encode_tensor2base64(Q.render_frame(sim, agent, frames[n])),
                   [d["objects"][str(o)]["cls"] for o in d["snapshots"][n] if str(o) in d["objects"]]) for i, n in enumerate(ctx)]
            gstep = int(d["global_step"])
            views = sorted((n for n in frames if n.startswith(f"{gstep}-view_")), key=lambda n: int(n.split("_")[-1].split(".")[0]))
            ego = eu.encode_tensor2base64(Q.render_frame(sim, agent, frames[views[-1]])) if views else None
            letters = sorted(qa["choices"])
            texts = [qa["choices"][L] for L in letters]
            correct_text = qa["choices"][qa["answer"]]
            res = {"subtask": sid, "category": qa["category"], "n_choices": len(letters), "recorded_pred": st["pred_answer"],
                   "answer": qa["answer"], "frames": len(ctx), "runs": []}
            orders = [list(range(len(texts)))] + [rng.sample(range(len(texts)), len(texts)) for _ in range(args.perms)]
            for k, order in enumerate(orders):
                shown = {letters[i]: texts[j] for i, j in enumerate(order)}
                gold = next(L for L, t in shown.items() if t == correct_text)
                question = f"{qa['question']} " + ", ".join(f"{L}: {t}" for L, t in shown.items())
                sys_prompt, content = C.qa_prompt(question, None, fr, ego)
                reply, _, _ = F.call_vlm(sys_prompt, content, model, processor)
                pred, frame = parse(reply)
                res["runs"].append({"perm": k, "order": order, "gold": gold, "pred": pred, "correct": pred == gold,
                                    "pred_text": shown.get(pred), "frame": frame})
            items.append(res)
            print(sid, [(r["pred"], r["gold"]) for r in res["runs"]], flush=True)
        sim.close()
    n = len(items)
    ident = [it["runs"][0] for it in items]
    perm_acc = [np.mean([r["correct"] for r in it["runs"][1:]]) for it in items]
    summary = {"run": args.run, "vlm": args.vlm, "questions_replayed": n,
               "replay_fidelity_pred_agreement": A.pct(sum(r["pred"] == (it["recorded_pred"] or "").strip() for r, it in zip(ident, items)), n),
               "recorded_accuracy_on_replayed": A.pct(sum((it["recorded_pred"] or "").strip() == it["answer"] for it in items), n),
               "replay_accuracy_original_order": A.pct(sum(r["correct"] for r in ident), n),
               "permuted_accuracy_mean_over_perms": round(100 * float(np.mean(perm_acc)), 2),
               "permuted_accuracy_per_perm": [round(100 * float(np.mean([it["runs"][k]["correct"] for it in items])), 2) for k in range(1, args.perms + 1)],
               "answer_consistent_across_all_orders": A.pct(sum(len({r["pred_text"] for r in it["runs"]}) == 1 for it in items), n),
               "pred_letter_dist_original": dict(collections.Counter(r["pred"] for r in ident)),
               "pred_letter_dist_permuted": dict(collections.Counter(r["pred"] for it in items for r in it["runs"][1:])),
               "permuted_accuracy_by_category": {c: round(100 * float(np.mean([a for a, it in zip(perm_acc, items) if it["category"] == c])), 1)
                                                 for c in sorted({it["category"] for it in items})}}
    os.makedirs(os.path.join(REPO, "analysis/out_p3"), exist_ok=True)
    json.dump({"summary": summary, "items": items}, open(os.path.join(REPO, f"analysis/out_p3/qa_permute_{args.run}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

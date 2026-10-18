"""Failure taxonomy + ground-truth diagnostics (a-e) for the MemoryExplorer baseline on LMEE-Bench (58-task subset).

Inputs : analysis/runs/instr_choice_sub/records_*.jsonl  (instrumented_run.py)
         analysis/out/gt_question_views.json            (gt_question_views.py)
         evaluation/results/exp_eval_lmee_choice/        (user's baseline run, read-only, for comparison)
Outputs: analysis/out/nav_items.csv, qa_items.csv, summary.json, and a printed report.
Run    : python analysis/analyze.py
"""
import collections
import csv
import glob
import json
import os
import pickle
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "analysis/runs", os.environ.get("RUN_NAME", "instr_choice_sub"))
OUT = os.path.join(REPO, os.environ.get("OUT_DIR", "analysis/out"))
BASE = os.path.join(REPO, "evaluation/results/exp_eval_lmee_choice")
STRUCT = {"wall", "floor", "ceiling", "Unknown", "unknown", "door frame", "window frame", "stair", "stairs",
          "beam", "pillar", "column", "baseboard", "ceiling lamp", "light", "ceiling light", "unknown object"}
MIN_PX = int(os.environ.get("MIN_PX", 1638))  # "visible" = covers >= 0.1% of a 1280x1280 frame (env MIN_PX for sensitivity)
MATCH_DIST = 1.5  # m: a detected object is matched to the nearest GT instance centre within this distance
# question words -> HM3D raw category names they may refer to (only for words whose HM3D name differs)
SYN = {"picture": ["picture", "painting", "frame", "photo", "poster", "art"], "painting": ["painting", "picture", "art"],
       "artwork": ["picture", "painting", "art", "frame"], "photo": ["picture", "photo", "frame"],
       "sofa": ["sofa", "couch"], "couch": ["couch", "sofa"], "tv": ["tv", "television"], "television": ["tv", "television"],
       "fridge": ["refrigerator", "fridge"], "plant": ["plant", "flower", "flowerpot", "vase"], "flower": ["flower", "plant", "vase"],
       "rug": ["rug", "carpet", "mat"], "carpet": ["carpet", "rug"], "nightstand": ["nightstand", "night stand", "bedside table"],
       "basin": ["sink", "basin"], "washbasin": ["sink", "washbasin"], "bathtub": ["bathtub", "tub"], "tub": ["bathtub", "tub"],
       "cupboard": ["cabinet", "cupboard"], "drawer": ["drawer", "dresser", "chest of drawers"], "lamp": ["lamp", "light"],
       "towel": ["towel"], "mirror": ["mirror"], "shelf": ["shelf", "shelves", "bookshelf"], "bed": ["bed"], "chair": ["chair", "stool"]}


# HM3D raw categories that name the same kind of navigation target (used for "same-category distractor")
GROUPS = [{"picture", "painting", "wall hanging decoration", "poster", "frame", "photo", "artwork", "wall art", "picture frame", "art"},
          {"rug", "mat", "doormat", "carpet", "bath mat"}, {"plant", "potted plant", "flowerpot", "flower pot", "plant pot"},
          {"vase", "flower vase"}, {"hanging clothes", "clothes", "clothing", "coat", "jacket"}, {"tv", "television", "tv stand"},
          {"couch", "sofa"}, {"nightstand", "night stand", "bedside table"}, {"refrigerator", "fridge"},
          {"towel", "hand towel", "bath towel"}, {"cabinet", "cupboard", "kitchen cabinet"}, {"lamp", "table lamp", "floor lamp"},
          {"sink", "washbasin", "bathroom sink", "kitchen sink"}, {"bathtub", "tub"}, {"shelf", "shelves", "bookshelf"}]


def canon(cat):
    c = (cat or "").lower().strip()
    for g in GROUPS:
        if c in g:
            return min(g)
    return c


def load_records():
    recs = []
    for f in sorted(glob.glob(os.path.join(RUN, "records_*.jsonl"))):
        recs += [json.loads(l) for l in open(f)]
    scenes = {r["scene_id"]: {o["sid"]: o for o in r["objects"]} for r in recs if r["type"] == "scene"}
    subtasks = collections.OrderedDict((r["subtask"], r) for r in recs if r["type"] == "subtask")  # last write wins
    decisions = collections.defaultdict(list)
    for r in recs:
        if r["type"] == "decision":
            decisions[r["subtask"]].append(r)
    return scenes, subtasks, decisions


def scene_of(subtask_id):
    return subtask_id.split("_")[0]


def nearest_gt(objs, center, group=None):
    best, bd = None, np.inf
    for sid, o in objs.items():
        if o["cat"] in STRUCT or (group is not None and canon(o["cat"]) != group):
            continue
        d = np.linalg.norm(np.asarray(o["center"]) - np.asarray(center))
        if d < bd:
            best, bd = sid, d
    return (best, bd) if bd <= MATCH_DIST else (None, bd)


def vis_of(frames, name):
    return {int(k) for k, c in frames.get(name, {}).get("vis", {}).items() if c >= MIN_PX}


def dominant_region(frames, name, objs):
    """Pixel-weighted majority HM3D region of the non-structural instances visible in a stored view."""
    w = collections.Counter()
    for k, c in frames.get(name, {}).get("vis", {}).items():
        o = objs.get(int(k))
        if o is not None and o["cat"] not in STRUCT and o["region"] not in (None, "_-1"):
            w[o["region"]] += c
    return w.most_common(1)[0][0] if w else None


def returned_set(dec):
    s = set()
    for q in dec["retrieval"]:
        s |= set(q["returned"])
    return s


def topk_set(dec, key, k):
    s = set()
    for q in dec["retrieval"]:
        s |= set(q[key][:k])
    return s


def load_goal_regions():
    """(scene, instruction) -> {goal_sid: region label}, from task_test/*/*/config.json (region number == HM3D region id)."""
    out = {}
    for f in glob.glob(os.path.join(REPO, "data/LMEE-Bench/task_test/*/*/config.json")):
        c = json.load(open(f))
        out[(c["Scene"], c["Task instruction"].strip())] = {int(o.split("_")[-1]): reg for o, reg in c["Object"]}
    return out


ROOM_WORDS = ["bathroom", "bedroom", "kitchen", "living room", "dining room", "office", "hall", "laundry", "closet",
              "garage", "toilet", "lounge", "family room", "entryway", "porch", "balcony", "stair", "spare room", "gym", "library"]


# ---------------------------------------------------------------- navigation
def analyze_nav(scenes, subtasks, decisions):
    regions = load_goal_regions()
    rows = []
    goals_by_ep = collections.defaultdict(list)
    for sid_, st in subtasks.items():
        if st["goal_type"] == "QA":
            continue
        objs = scenes[scene_of(sid_)]
        goal = int(st["goal_obj_ids"][0])
        G = {int(x) for x in st["goal_obj_ids"]}  # any of these counts (GOAT 'object' goals have several)
        gcat = objs.get(goal, {}).get("cat", "?")
        gregions = {objs[g]["region"] for g in G if g in objs}
        prev_goals = set(goals_by_ep[st["episode"]])
        goals_by_ep[st["episode"]] += sorted(G)
        frames = st["frames"]
        seen_frames = [n for n in frames if G & vis_of(frames, n)]
        mapped = {int(x) for v in st["goal_obj_ids_mapping"].values() for x in v}
        decs = decisions.get(sid_, [])
        last = decs[-1] if decs else None
        distractors = [s for s, o in objs.items() if canon(o["cat"]) == canon(gcat) and s not in G]
        exact_distractors = [s for s, o in objs.items() if o["cat"] == gcat and s not in G]
        r = {"subtask": sid_, "difficulty": st["difficulty"], "goal_type": st["goal_type"], "goal_sid": goal,
             "goal_cat": gcat, "n_same_cat_distractors": len(distractors), "n_exact_cat_distractors": len(exact_distractors),
             "n_same_cat_distractors_same_region": sum(objs[s]["region"] in gregions for s in distractors), "n_goal_instances": len(G),
             "success": bool(st["success_by_distance"]), "success_by_memory": bool(st["success_by_snapshot"]),
             "task_step": st["task_step"], "num_step": st["num_step"], "step_cap": st["task_step"] >= st["num_step"],
             "goal_seen": bool(seen_frames), "goal_detected": bool(mapped), "n_decisions": len(decs)}
        # gold memory (GT): a stored snapshot whose frame shows the goal; (det): a snapshot whose cluster holds a mapped detection
        gold_vis_any, gold_det_any, ret_gold_any = False, False, False
        for d in decs:
            gv = {n for n in d["snapshots"] if G & vis_of(frames, n)}
            gd = {n for n, cl in d["snapshots"].items() if set(cl) & mapped}
            gold_vis_any |= bool(gv)
            gold_det_any |= bool(gd)
            if gd and (returned_set(d) & gd):
                ret_gold_any = True
        r.update(gold_vis_in_memory=gold_vis_any, gold_det_in_memory=gold_det_any, gold_retrieved_any=ret_gold_any)
        # recall at the last decision (if a gold memory existed then)
        if last is not None:
            gd = {n for n, cl in last["snapshots"].items() if set(cl) & mapped}
            gv = {n for n in last["snapshots"] if G & vis_of(frames, n)}
            r["last_gold_det"] = bool(gd)
            r["last_gold_vis"] = bool(gv)
            r["last_recall_det"] = bool(returned_set(last) & gd) if gd else None
            r["last_recall_vis"] = bool(returned_set(last) & gv) if gv else None
            r["last_recall_vis_text3"] = bool(topk_set(last, "text_rank", 3) & gv) if gv else None
            r["last_recall_vis_image3"] = bool(topk_set(last, "image_rank", 3) & gv) if gv else None
            r["n_memories_last"] = len(last["snapshots"])
        ch = last["choice"] if last else None
        r["last_choice"] = ch["type"] if ch else "none"
        r["n_frontiers_last"] = last["n_frontiers"] if last else None
        chosen_sid = None
        if ch and ch["type"] == "memory":
            oid = str(ch["cluster"][0])
            o = last["objects"].get(oid)
            if o is not None:
                chosen_sid, dist = nearest_gt(objs, o["center"])
                if canon(o["cls"]) == canon(gcat):
                    # detector believes it is the goal category: match among GT instances of that category only;
                    # none within MATCH_DIST -> false-positive detection
                    chosen_sid, dist = nearest_gt(objs, o["center"], group=canon(gcat))
                    r["chosen_false_positive"] = chosen_sid is None
                r["chosen_det_cls"] = o["cls"]
                r["chosen_gt_sid"] = chosen_sid
                r["chosen_gt_cat"] = objs[chosen_sid]["cat"] if chosen_sid is not None else None
                r["chosen_is_goal_det"] = int(oid) in mapped
        instr = st["question"].split("The complete task is: ")[-1].split("\nNow you need")[0].strip()
        reg_label = regions.get((scene_of(sid_), instr), {}).get(goal, "")
        rooms = [w for w in ROOM_WORDS if w in reg_label.lower()]
        r["goal_region_label"] = reg_label
        r["instruction_names_goal_room"] = bool(rooms) and any(w in instr.lower() for w in rooms)
        if chosen_sid is not None:
            r["chosen_same_region_as_goal"] = objs[chosen_sid]["region"] in gregions
        # a stored view whose dominant region is the goal's region (subtask frames, i.e. up to subtask end)
        r["goal_region_observed"] = any(dominant_region(frames, n, objs) in gregions for n in frames)
        r["chose_goal"] = bool(ch and ch["type"] == "memory" and (r.get("chosen_is_goal_det") or chosen_sid in G))
        r["chose_distractor"] = bool(chosen_sid is not None and chosen_sid not in G and canon(objs[chosen_sid]["cat"]) == canon(gcat))
        r["chose_prev_goal"] = bool(chosen_sid is not None and chosen_sid in prev_goals and chosen_sid not in G)
        r["category"] = "success" if r["success"] else nav_category(r)
        rows.append(r)
    return rows


def nav_category(r):
    if r["n_decisions"] == 0 or r["last_choice"] == "none":
        if r["n_frontiers_last"] == 0:  # no frontier left and no memory chosen -> run aborts the subtask
            return "exploration: aborted, no frontier available"
        return "other: aborted / invalid output"
    if r["last_choice"] == "memory":
        if r["chose_goal"]:
            return "goal chosen, viewpoint not reached"
        if r["chose_prev_goal"]:
            return "task-progress error"
        if r["chose_distractor"]:
            return "wrong instance (goal already observed)" if r["goal_seen"] else "wrong instance (goal not yet observed)"
        if r.get("chosen_false_positive"):
            return "perception: false-positive goal detection"
    # committed to a non-goal object of another category, or kept exploring until the step cap
    if not r["goal_seen"]:
        return "exploration failure / step cap" if r["last_choice"] == "frontier" else "exploration failure: premature commit"
    if not r["goal_detected"]:
        return "perception: goal seen, not detected"
    if not r["gold_det_in_memory"]:
        return "perception: detected, not written to memory"
    if not r["gold_retrieved_any"]:
        return "retrieval miss"
    return "selection error: gold memory shown, not chosen"


# ---------------------------------------------------------------- QA
def words(text):
    return set(re.findall(r"[a-z]+", text.lower())) | {w[:-1] for w in re.findall(r"[a-z]+s\b", text.lower())}


def relevant_instances(gtv, qa):
    """Instances in the GT question view whose HM3D category is named in the question / correct choice / open answer."""
    text = " ".join([qa["question"], qa["choices"].get(qa["answer"], ""), qa.get("open_answer", "")])
    ws = words(text)
    cand_names = set()
    for w in ws:
        cand_names |= set(SYN.get(w, [w]))
    R = set()
    for sid, v in gtv["vis"].items():
        cat = v["cat"].lower()
        if cat in {"unknown", "ceiling", "floor"}:
            continue
        toks = cat.split()
        if cat in cand_names or toks[-1] in cand_names or toks[-1] in ws or (toks[-1].endswith("s") and toks[-1][:-1] in ws):
            if v["px_frac"] >= 0.002:  # ignore slivers (<0.2% of the image)
                R.add(int(sid))
    return R


def analyze_qa(scenes, subtasks, decisions, gtviews):
    ok = json.load(open(os.path.join(REPO, "analysis/labels/gt_view_ok.json")))["ok"]  # manual check of each GT-view match
    gtv_by = {(g["scene"], g["episode"], g["object_id"], g["question"]): g for g in gtviews if ok.get(str(g["idx"]))}
    rows = []
    for sid_, st in subtasks.items():
        if st["goal_type"] != "QA":
            continue
        qa = st["QA"]
        scene = scene_of(sid_)
        ep = int(st["episode"].split("_ep_")[1])
        gtv = gtv_by.get((scene, ep, qa["object_id"], qa["question"]))
        frames = st["frames"]
        decs = decisions.get(sid_, [])
        d = decs[0] if decs else None  # the answer comes from the first QA decision
        pred = (st["pred_answer"] or "").strip()
        r = {"subtask": sid_, "difficulty": st["difficulty"], "qcat": qa["category"], "question": qa["question"],
             "answer": qa["answer"], "pred": pred, "correct": pred == qa["answer"], "n_choices": len(qa["choices"]),
             "gt_match_err": gtv["match_err"] if gtv else None}
        objs = scenes[scene]
        R = relevant_instances(gtv, qa) if gtv else set()
        r["R_source"] = "verified GT view + question nouns"
        if not R:  # GT view unverified, or no question noun matched an instance in it: use the QA's goal instance
            R = {int(qa["object_id"].split("_")[-1])}
            r["R_source"] = "goal instance (GT view unverified)" if not gtv else "goal instance (no noun match)"
            r["R_fallback"] = True
        r["R"] = sorted(R)
        r["R_cats"] = sorted({objs[s]["cat"] for s in R if s in objs})
        mem = list(d["snapshots"]) if d else []
        covers = lambda n: R <= vis_of(frames, n)
        gold = [n for n in mem if covers(n)]
        union_mem = set().union(*[vis_of(frames, n) for n in mem]) if mem else set()
        union_all = set().union(*[vis_of(frames, n) for n in frames]) if frames else set()
        any_frame_single = any(covers(n) for n in frames)
        r.update(n_memories=len(mem), n_gold=len(gold), single_view_gold=bool(gold),
                 union_mem_covers=R <= union_mem, observed_all=R <= union_all, any_frame_single=any_frame_single)
        if d:
            ret = returned_set(d)
            r["n_retrieved"] = len(ret)
            r["recall_gold"] = bool(ret & set(gold)) if gold else None
            r["recall_text3"] = bool(topk_set(d, "text_rank", 3) & set(gold)) if gold else None
            r["recall_image3"] = bool(topk_set(d, "image_rank", 3) & set(gold)) if gold else None
            ranks = []
            for q in d["retrieval"]:
                for key in ("text_rank", "image_rank"):
                    rk = [i for i, n in enumerate(q[key]) if n in gold]
                    if rk:
                        ranks.append(rk[0] + 1)
            r["best_gold_rank"] = min(ranks) if ranks else None
            ch = d["choice"]
            r["chosen"] = ch["image"] if ch else None
            r["chosen_gold"] = bool(ch and ch["image"] in gold)
            r["chosen_has_goal"] = bool(ch and int(qa["object_id"].split("_")[-1]) in vis_of(frames, ch["image"]))
            greg = objs.get(int(qa["object_id"].split("_")[-1]), {}).get("region")
            r["chosen_region_is_goal_region"] = (dominant_region(frames, ch["image"], objs) == greg) if ch else None
            r["any_memory_in_goal_region"] = any(dominant_region(frames, n, objs) == greg for n in mem)
        r["view_need"] = ("single" if gold else "multi-view" if r["union_mem_covers"] else
                          "not in memory (observed)" if r["observed_all"] else "never observed")
        r["category"] = "correct" if r["correct"] else qa_category(r)
        rows.append(r)
    return rows


def qa_category(r):
    if not r["pred"] or r.get("chosen") is None:
        return "other: invalid output"
    if r["view_need"] == "never observed":
        return "exploration: evidence never observed"
    if r["view_need"] == "not in memory (observed)":
        return "memory writing: observed, not kept in any memory view"
    if r["view_need"] == "multi-view":
        return "multi-view needed"
    if not r["recall_gold"]:
        return "retrieval miss"
    if not r["chosen_gold"]:
        return "selection error: gold retrieved, other memory chosen"
    return {"Counting": "counting error", "Relationship": "spatial-relation error", "Location": "spatial-relation error",
            "Attribute": "perception/attribute error", "State": "perception/attribute error"}.get(r["qcat"], "other")


# ---------------------------------------------------------------- report
def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else f"{a}/0"


def table(rows, key="category", examples=3):
    fails = [r for r in rows if r[key] not in ("success", "correct")]
    c = collections.Counter(r[key] for r in fails)
    out = []
    for k, n in c.most_common():
        ex = [r["subtask"] for r in fails if r[key] == k][:examples]
        out.append({"category": k, "count": n, "share_of_failures": round(100 * n / len(fails), 1),
                    "share_of_items": round(100 * n / len(rows), 1), "examples": ex})
    return out, len(fails)


def baseline_compare(nav, qa):
    b = {}
    sbd = pickle.load(open(os.path.join(BASE, "success_by_distance.pkl"), "rb"))
    ans = json.load(open(os.path.join(BASE, "lmee_answer.json")))
    b["baseline_SR"] = round(100 * np.mean(list(sbd.values())), 2)
    b["baseline_acc"] = round(100 * np.mean([a["pred_answer"] == a["answer"] for a in ans]), 2)
    b["rerun_SR"] = round(100 * np.mean([r["success"] for r in nav]), 2)
    b["rerun_acc"] = round(100 * np.mean([r["correct"] for r in qa]), 2)
    b["rerun_n_nav"], b["rerun_n_qa"] = len(nav), len(qa)
    common = [r for r in nav if r["subtask"] in sbd]
    b["nav_agreement"] = round(100 * np.mean([r["success"] == bool(sbd[r["subtask"]]) for r in common]), 1) if common else None
    return b


def main():
    os.makedirs(OUT, exist_ok=True)
    scenes, subtasks, decisions = load_records()
    gtv_path = os.path.join(OUT, "gt_question_views.json")
    gtviews = json.load(open(gtv_path)) if os.path.exists(gtv_path) else []
    nav = analyze_nav(scenes, subtasks, decisions)
    qa = analyze_qa(scenes, subtasks, decisions, gtviews)
    S = {"baseline_vs_rerun": baseline_compare(nav, qa)}

    S["nav_taxonomy"], S["nav_n_fail"] = table(nav)
    S["qa_taxonomy"], S["qa_n_fail"] = table(qa)

    # (a) multi-view need
    vc = collections.Counter(r["view_need"] for r in qa)
    S["diag_a"] = {"n_qa": len(qa), "view_need": dict(vc),
                   "view_need_by_qcat": {c: dict(collections.Counter(r["view_need"] for r in qa if r["qcat"] == c))
                                         for c in sorted({r["qcat"] for r in qa})},
                   "accuracy_by_view_need": {k: pct(sum(r["correct"] for r in qa if r["view_need"] == k), v) for k, v in vc.items()},
                   "R_source": dict(collections.Counter(r["R_source"] for r in qa)),
                   "view_need_verified_only": dict(collections.Counter(r["view_need"] for r in qa if r["R_source"].startswith("verified"))),
                   "single_view_exists_in_any_observed_frame": pct(sum(r["any_frame_single"] for r in qa), len(qa))}
    # (b) same-category distractors
    nd = [r for r in nav]
    mem_choice = [r for r in nav if r["last_choice"] == "memory"]
    S["diag_b"] = {"targets_with_same_cat_distractor_in_scene": pct(sum(r["n_same_cat_distractors"] > 0 for r in nd), len(nd)),
                   "targets_with_exact_hm3d_category_distractor": pct(sum(r["n_exact_cat_distractors"] > 0 for r in nd), len(nd)),
                   "targets_with_same_cat_distractor_same_region": pct(sum(r["n_same_cat_distractors_same_region"] > 0 for r in nd), len(nd)),
                   "median_distractors": float(np.median([r["n_same_cat_distractors"] for r in nd])) if nd else None,
                   "chose_distractor_of_memory_choices": pct(sum(r["chose_distractor"] for r in mem_choice), len(mem_choice)),
                   "chose_distractor_of_all_targets": pct(sum(r["chose_distractor"] for r in nd), len(nd)),
                   "wrong_instance_choices_in_other_region_than_goal": pct(
                       sum(r["chose_distractor"] and not r.get("chosen_same_region_as_goal") for r in nd), sum(r["chose_distractor"] for r in nd)),
                   "goals_whose_room_type_is_named_in_instruction": pct(sum(r["instruction_names_goal_room"] for r in nd), len(nd)),
                   "wrong_instance_with_goal_room_named": pct(sum(r["chose_distractor"] and r["instruction_names_goal_room"] and not r.get("chosen_same_region_as_goal") for r in nd),
                                                             sum(r["chose_distractor"] for r in nd)),
                   "wrong_instance_failures_goal_region_already_observed": pct(
                       sum(r["category"].startswith("wrong instance") and r["goal_region_observed"] for r in nd), sum(r["category"].startswith("wrong instance") for r in nd)),
                   "premature_commit_choice_in_other_region": pct(
                       sum(r["category"].startswith("exploration failure: premature") and r.get("chosen_same_region_as_goal") is False for r in nd),
                       sum(r["category"].startswith("exploration failure: premature") for r in nd)),
                   "SR_with_vs_without_distractor": {
                       "with": pct(sum(r["success"] for r in nd if r["n_same_cat_distractors"] > 0), sum(r["n_same_cat_distractors"] > 0 for r in nd)),
                       "without": pct(sum(r["success"] for r in nd if r["n_same_cat_distractors"] == 0), sum(r["n_same_cat_distractors"] == 0 for r in nd))}}
    # (c) retrieval recall@3 of the gold memory
    qg = [r for r in qa if r.get("recall_gold") is not None]
    ng = [r for r in nav if r.get("last_recall_det") is not None]
    nv = [r for r in nav if r.get("last_recall_vis") is not None]
    S["diag_c"] = {"qa_items_with_single_view_gold_in_memory": pct(len(qg), len(qa)),
                   "qa_recall_union(text@3 U image@3)": pct(sum(r["recall_gold"] for r in qg), len(qg)),
                   "qa_recall_text@3": pct(sum(r["recall_text3"] for r in qg), len(qg)),
                   "qa_recall_image@3": pct(sum(r["recall_image3"] for r in qg), len(qg)),
                   "qa_acc_gold_retrieved": pct(sum(r["correct"] for r in qg if r["recall_gold"]), sum(r["recall_gold"] for r in qg)),
                   "qa_acc_gold_not_retrieved": pct(sum(r["correct"] for r in qg if not r["recall_gold"]), sum(not r["recall_gold"] for r in qg)),
                   "qa_mean_retrieved": round(float(np.mean([r["n_retrieved"] for r in qa if "n_retrieved" in r])), 2) if qa else None,
                   "qa_mean_memories": round(float(np.mean([r["n_memories"] for r in qa])), 2) if qa else None,
                   "nav_goals_with_gold_det_at_last_decision": pct(len(ng), len(nav)),
                   "nav_recall_det_union": pct(sum(r["last_recall_det"] for r in ng), len(ng)),
                   "nav_recall_vis_union": pct(sum(r["last_recall_vis"] for r in nv), len(nv)),
                   "nav_recall_vis_text@3": pct(sum(r["last_recall_vis_text3"] for r in nv), len(nv)),
                   "nav_recall_vis_image@3": pct(sum(r["last_recall_vis_image3"] for r in nv), len(nv))}
    # (d) exploration / step cap by difficulty
    d = {}
    for diff in ("easy", "medium", "hard"):
        rs = [r for r in nav if r["difficulty"] == diff]
        fs = [r for r in rs if not r["success"]]
        expl = [r for r in fs if r["category"].startswith("exploration")]
        cap = [r for r in fs if r["step_cap"]]
        d[diff] = {"targets": len(rs), "failures": len(fs), "exploration_failures": pct(len(expl), len(fs)),
                   "step_cap_reached": pct(len(cap), len(fs)), "goal_never_seen": pct(sum(not r["goal_seen"] for r in fs), len(fs))}
    S["diag_d"] = d
    # (e) other patterns
    letters = collections.Counter(r["pred"] for r in qa)
    gt_letters = collections.Counter(r["answer"] for r in qa)
    nav_ok = {(r["subtask"].rsplit("_", 1)[0], r["goal_sid"]): r for r in nav}
    for r in qa:
        st = subtasks[r["subtask"]]
        n = nav_ok.get((r["subtask"].rsplit("_", 1)[0], int(st["QA"]["object_id"].split("_")[-1])))
        r["goal_nav_success"] = n["success"] if n else None
        r["goal_nav_seen"] = n["goal_seen"] if n else None
    S["diag_e_qa_vs_nav"] = {
        "qa_acc_goal_reached": pct(sum(r["correct"] for r in qa if r["goal_nav_success"]), sum(bool(r["goal_nav_success"]) for r in qa)),
        "qa_acc_goal_not_reached": pct(sum(r["correct"] for r in qa if r["goal_nav_success"] is False), sum(r["goal_nav_success"] is False for r in qa)),
        "evidence_never_observed_given_goal_not_reached": pct(sum(r["view_need"] == "never observed" for r in qa if r["goal_nav_success"] is False), sum(r["goal_nav_success"] is False for r in qa)),
        "evidence_never_observed_given_goal_reached": pct(sum(r["view_need"] == "never observed" for r in qa if r["goal_nav_success"]), sum(bool(r["goal_nav_success"]) for r in qa)),
        "qa_items_without_matching_nav_subtask": sum(r["goal_nav_success"] is None for r in qa)}
    S["diag_e_qa_room"] = {
        "answered_from_view_in_goal_region": pct(sum(bool(r.get("chosen_region_is_goal_region")) for r in qa), len(qa)),
        "acc_view_in_goal_region": pct(sum(r["correct"] for r in qa if r.get("chosen_region_is_goal_region")), sum(bool(r.get("chosen_region_is_goal_region")) for r in qa)),
        "acc_view_in_other_region": pct(sum(r["correct"] for r in qa if r.get("chosen_region_is_goal_region") is False), sum(r.get("chosen_region_is_goal_region") is False for r in qa)),
        "wrong_answers_other_region_while_goal_region_view_in_memory": pct(sum((not r["correct"]) and r.get("chosen_region_is_goal_region") is False and r["any_memory_in_goal_region"] for r in qa), len(qa)),
        "other_region_answer_while_goal_region_view_in_memory": pct(sum(r.get("chosen_region_is_goal_region") is False and r["any_memory_in_goal_region"] for r in qa), len(qa))}
    ab = [r for r in nav if r["category"].startswith("exploration: aborted")]
    S["diag_e_aborts"] = {"aborts": len(ab), "episodes": len({r["subtask"].rsplit("_", 1)[0] for r in ab}),
                          "at_task_step_1": sum(int(r["task_step"]) == 1 for r in ab)}
    S["diag_e"] = {"pred_letter_dist": dict(letters), "gt_letter_dist": dict(gt_letters),
                   "acc_by_n_choices": {n: pct(sum(r["correct"] for r in qa if r["n_choices"] == n), sum(r["n_choices"] == n for r in qa))
                                        for n in sorted({r["n_choices"] for r in qa})},
                   "qa_acc_chosen_memory_shows_goal": pct(sum(r["correct"] for r in qa if r.get("chosen_has_goal")), sum(bool(r.get("chosen_has_goal")) for r in qa)),
                   "qa_acc_chosen_memory_not_goal": pct(sum(r["correct"] for r in qa if not r.get("chosen_has_goal")), sum(not r.get("chosen_has_goal") for r in qa)),
                   "nav_success_by_memory_but_not_distance": pct(sum(r["success_by_memory"] and not r["success"] for r in nav), len(nav)),
                   "constant_letter_baselines": {L: pct(sum(r["answer"] == L for r in qa), len(qa)) for L in "ABCD"},
                   "acc_when_pred_B": pct(sum(r["correct"] for r in qa if r["pred"] == "B"), sum(r["pred"] == "B" for r in qa)),
                   "acc_when_pred_not_B": pct(sum(r["correct"] for r in qa if r["pred"] != "B"), sum(r["pred"] != "B" for r in qa))}
    # upper bounds per candidate gap: every failure in the candidate's categories turns into a success
    def ub(rows, cats):
        n = sum(r["category"] in cats or any(r["category"].startswith(c) for c in cats) for r in rows)
        fails = sum(r["category"] not in ("success", "correct") for r in rows)
        return {"failures": n, "share_of_failures": round(100 * n / fails, 1), "upper_bound_gain_pts": round(100 * n / len(rows), 1)}
    sv = [r for r in qa if r["view_need"] == "single"]
    nev = [r for r in qa if r["view_need"] == "never observed"]
    S["candidates"] = {
        "G1_room_scoped_commitment_nav": ub(nav, ["exploration failure: premature commit", "wrong instance"]),
        "G1_room_choice_qa": {"wrong_other_region_with_goal_region_view_in_memory": S["diag_e_qa_room"]["wrong_answers_other_region_while_goal_region_view_in_memory"]},
        "G2_evidence_sufficiency_qa": dict(ub(qa, ["exploration: evidence never observed"]),
                                           expected_gain_if_never_observed_items_reach_single_view_accuracy_pts=round(
                                               100 * len(nev) * (np.mean([r["correct"] for r in sv]) - np.mean([r["correct"] for r in nev])) / len(qa), 1)),
        "G3_multi_view_qa": ub(qa, ["multi-view needed"]),
        "G4_retrieval_nav": ub(nav, ["retrieval miss"]), "G4_retrieval_qa": ub(qa, ["retrieval miss"]),
        "G5_task_progress_nav": ub(nav, ["task-progress error"]),
        "perception_nav": ub(nav, ["perception"]), "exploration_abort_nav": ub(nav, ["exploration: aborted", "exploration failure / step cap"])}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    for name, rows in (("nav_items.csv", nav), ("qa_items.csv", qa)):
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()

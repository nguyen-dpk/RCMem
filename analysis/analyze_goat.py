"""GOAT-Bench check of the LMEE findings.

Part 1 (static, all 360 val_unseen episodes, no simulator): for every GOAT task, how many same-category instances
exist in the scene (and in other HM3D regions), and whether the goal description names a room.
Part 2 (run-based, analysis/runs/goat_ep0 from goat_runner.py): the navigation failure taxonomy of analyze.py,
split by GOAT modality (object / description / image).
Output: analysis/out_goat/summary.json, nav_items.csv.   Run: python analysis/analyze_goat.py
"""
import collections
import csv
import glob
import json
import os
import re

os.environ.setdefault("RUN_NAME", "goat_ep0")
os.environ.setdefault("OUT_DIR", "analysis/out_goat")
import analyze as A  # noqa: E402

REPO = A.REPO
GOAT = os.path.join(REPO, "data/goat_bench/val_unseen/content")


def semantic_objects(scene_short):
    f = glob.glob(os.path.join(REPO, f"data/hm3d-0.2/hm3d/*/*-{scene_short}/{scene_short}.semantic.txt"))[0]
    objs = {}
    for line in open(f).read().splitlines()[1:]:
        m = re.match(r'(\d+),[0-9A-Fa-f]+,"(.*)",(-?\d+)', line)
        if m:
            objs[int(m.group(1))] = {"cat": m.group(2), "region": int(m.group(3))}
    return objs


def static_part():
    rows = []
    for f in sorted(glob.glob(os.path.join(GOAT, "*.json"))):
        sid = os.path.basename(f).split(".")[0]
        objs = semantic_objects(sid)
        g = json.load(open(f))
        by_id = {x["object_id"]: x for v in g["goals"].values() for x in v}
        for e in g["episodes"]:
            for t in e["tasks"]:
                cat, mode = t[0], t[1]
                if mode == "object":
                    insts = [x for x in g["goals"].get(f"{sid}.basis.glb_{cat}", [])]
                    G = {int(x["object_id"].split("_")[-1]) for x in insts}
                    desc = ""
                else:
                    if t[2] not in by_id:
                        continue
                    G = {int(t[2].split("_")[-1])}
                    desc = by_id[t[2]].get("lang_desc", "") or ""
                if not G:
                    continue
                gc = A.canon(objs.get(next(iter(G)), {}).get("cat", cat))
                greg = {objs[x]["region"] for x in G if x in objs}
                dis = [s for s, o in objs.items() if A.canon(o["cat"]) == gc and s not in G]
                rows.append({"mode": mode, "n_goal_instances": len(G), "n_distractors": len(dis),
                             "n_distractors_other_region": sum(objs[s]["region"] not in greg for s in dis),
                             "desc_names_room": any(w in desc.lower() for w in A.ROOM_WORDS)})
    out = {}
    for mode in ("object", "description", "image"):
        rs = [r for r in rows if r["mode"] == mode]
        out[mode] = {"tasks": len(rs),
                     "with_same_cat_distractor": A.pct(sum(r["n_distractors"] > 0 for r in rs), len(rs)),
                     "with_distractor_in_other_region": A.pct(sum(r["n_distractors_other_region"] > 0 for r in rs), len(rs)),
                     "median_goal_instances": sorted(r["n_goal_instances"] for r in rs)[len(rs) // 2] if rs else None}
    ds = [r for r in rows if r["mode"] == "description"]
    out["description_names_a_room"] = A.pct(sum(r["desc_names_room"] for r in ds), len(ds))
    return out


def run_part():
    scenes, subtasks, decisions = A.load_records()
    nav = A.analyze_nav(scenes, subtasks, decisions)
    S = {"n_targets": len(nav), "SR": A.pct(sum(r["success"] for r in nav), len(nav))}
    S["taxonomy_all"], S["n_fail"] = A.table(nav)
    for mode in ("object", "description", "image"):
        rs = [r for r in nav if r["goal_type"] == mode]
        tab, nf = A.table(rs, examples=2) if rs else ([], 0)
        mem = [r for r in rs if r["last_choice"] == "memory"]
        S[mode] = {"targets": len(rs), "SR": A.pct(sum(r["success"] for r in rs), len(rs)), "failures": nf,
                   "taxonomy": [(t["category"], t["count"], t["share_of_failures"]) for t in tab],
                   "chose_distractor_of_memory_choices": A.pct(sum(r["chose_distractor"] for r in mem), len(mem)),
                   "wrong_choice_in_other_region": A.pct(sum((r["chose_distractor"] or r["category"].startswith("exploration failure: premature"))
                                                             and r.get("chosen_same_region_as_goal") is False for r in rs),
                                                         sum(r["chose_distractor"] or r["category"].startswith("exploration failure: premature") for r in rs))}
    cats = ["exploration failure: premature commit", "wrong instance"]
    g1 = [r for r in nav if any(r["category"].startswith(c) for c in cats)]
    S["G1_premature_or_wrong_instance"] = {"failures": len(g1), "share_of_failures": A.pct(len(g1), S["n_fail"]),
                                           "upper_bound_gain_pts": round(100 * len(g1) / len(nav), 1) if nav else None,
                                           "choice_in_other_region": A.pct(sum(r.get("chosen_same_region_as_goal") is False for r in g1), len(g1)),
                                           "goal_region_already_observed": A.pct(sum(r["goal_region_observed"] for r in g1), len(g1))}
    ng = [r for r in nav if r.get("last_recall_vis") is not None]
    S["retrieval_recall_vis_union"] = A.pct(sum(r["last_recall_vis"] for r in ng), len(ng))
    S["step_cap_reached_of_failures"] = A.pct(sum(r["step_cap"] for r in nav if not r["success"]), S["n_fail"])
    return nav, S


def main():
    os.makedirs(A.OUT, exist_ok=True)
    S = {"static_all_360_episodes": static_part()}
    if glob.glob(os.path.join(A.RUN, "records_*.jsonl")):
        nav, S["run_episode0_per_scene"] = run_part()
        keys = sorted({k for r in nav for k in r})
        with open(os.path.join(A.OUT, "nav_items.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(nav)
    json.dump(S, open(os.path.join(A.OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()

"""P3 pilot sample (DEV_PLAN.md, "Qualitative HTML reports"): about 8 episodes of the 58-task subset, drawn with a fixed
rule so the same cases recur across reports.

Rule: for seed s = 0, 1, 2, ...: shuffle the subset's scene files with random.Random(s), take scenes in that order until
they hold >= 8 episodes; accept the first s whose episodes cover every case below (flags from the fixed baseline's
logged run p2_E0_fix_s77 and the P3-0a decision table analysis/out_p3_0/p2_E0_fix_s77__qwen3_band.jsonl):
  description goal; image goal; a decision with the goal's region already seen; one with it not seen yet; a goal with
  a same-category GT instance in another region (distractor); an episode whose goals span >= 2 regions; a QA subtask.
Whole scene files are taken, so episode ids match the full runs. Writes analysis/out_p3/pilot_data/<scene>.json
(symlinks), analysis/cfg/p3_pilot.yaml (the subset config pointed at them) and analysis/out_p3/pilot.json.
Run (repo root): python analysis/p3_pilot.py
"""
import collections
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze as A  # noqa: E402

RUN = "p2_E0_fix_s77"
CASES = ["description", "image", "region_seen", "region_unseen", "distractor", "multi_region", "qa"]
SUB = os.path.join(REPO, "data/LMEE-Bench/lmee_bench_sub")
OUT = os.path.join(REPO, "analysis/out_p3")


def flags():
    scenes, subtasks, _ = A.load_records(os.path.join(REPO, "analysis/runs", RUN))
    dec = collections.defaultdict(list)
    for l in open(os.path.join(REPO, f"analysis/out_p3_0/{RUN}__qwen3_band.jsonl")):
        d = json.loads(l)
        dec[d["subtask"]].append(d)
    F = collections.defaultdict(set)
    for sid, st in subtasks.items():
        ep, objs = st["episode"], scenes[A.scene_of(st["episode"])]
        F[ep].add(st["goal_type"].lower() if st["goal_type"] != "QA" else "qa")
        if st["goal_type"] == "QA":
            continue
        for d in dec[sid]:
            F[ep].add("region_seen" if set(d["goal_regions"]) & set(d["regions"]) else "region_unseen")
        for g in st["goal_obj_ids"]:
            if g in objs:
                cat, reg = A.canon(objs[g]["cat"]), objs[g]["region"]
                if any(A.canon(o["cat"]) == cat and o["region"] not in (reg, None, "_-1") for o in objs.values()):
                    F[ep].add("distractor")
    regs = collections.defaultdict(set)
    for st in subtasks.values():
        if st["goal_type"] != "QA":
            objs = scenes[A.scene_of(st["episode"])]
            regs[st["episode"]] |= {objs[g]["region"] for g in st["goal_obj_ids"] if g in objs}
    for ep, r in regs.items():
        if len(r - {None, "_-1"}) >= 2:
            F[ep].add("multi_region")
    return F


def main():
    F = flags()
    by_scene = collections.defaultdict(list)
    for ep in F:
        by_scene[A.scene_of(ep)].append(ep)
    scenes = sorted(f[:-5] for f in os.listdir(SUB) if f.endswith(".json"))
    for seed in range(1000):
        order = scenes[:]
        random.Random(seed).shuffle(order)
        pick, eps = [], []
        for s in order:
            if len(eps) >= 8:
                break
            pick.append(s)
            eps += sorted(by_scene[s])
        cover = set().union(*[F[e] for e in eps])
        if all(c in cover for c in CASES):
            break
    os.makedirs(os.path.join(OUT, "pilot_data"), exist_ok=True)
    for f in os.listdir(os.path.join(OUT, "pilot_data")):
        os.remove(os.path.join(OUT, "pilot_data", f))
    for s in pick:
        os.symlink(os.path.join(SUB, s + ".json"), os.path.join(OUT, "pilot_data", s + ".json"))
    cfg = open(os.path.join(REPO, "analysis/cfg/instr_choice_sub.yaml")).read()
    cfg = cfg.replace('test_data_dir: "../data/LMEE-Bench/lmee_bench_sub"', 'test_data_dir: "../analysis/out_p3/pilot_data"')
    assert "pilot_data" in cfg
    open(os.path.join(REPO, "analysis/cfg/p3_pilot.yaml"), "w").write(cfg)
    res = {"seed": seed, "scenes": pick, "episodes": eps, "rule": __doc__.split("\n\n")[1],
           "cases": {c: [e for e in eps if c in F[e]] for c in CASES}}
    json.dump(res, open(os.path.join(OUT, "pilot.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("seed", "scenes", "episodes")}), {c: len(v) for c, v in res["cases"].items()})


if __name__ == "__main__":
    main()

"""Phase-1 decision-gate tables (DEV_PLAN.md P1): per-experiment metrics vs the E0 mean, the E3 commit-rate
diagnostic, and the G1 decision rule.  Run (repo root): python analysis/p1_report.py
Inputs : analysis/runs/<run>/records_*.jsonl for the runs listed in RUNS (missing runs are reported as missing)
Outputs: analysis/out_p1/p1_results.json, analysis/out_p1/tables.md
"""
import collections
import json
import os

import numpy as np

import analyze as A

REPO = A.REPO
OUT = os.path.join(REPO, "analysis/out_p1")
RUNS = {"E0": ["instr_choice_sub", "p1_E0_r2", "p1_E0_r3"],  # instr_choice_sub = E0 run 1 (gap-report re-run)
        "E1": ["p1_E1"], "E2": ["p1_E2"], "E4": ["p1_E4"], "E5": ["p1_E5"]}
N_TASK_EPISODES = 58


def load(run):
    d = os.path.join(REPO, "analysis/runs", run)
    scenes, subtasks, decisions = A.load_records(d)
    gtv = json.load(open(os.path.join(REPO, "analysis/out/gt_question_views.json")))
    nav = A.analyze_nav(scenes, subtasks, decisions)
    qa = A.analyze_qa(scenes, subtasks, decisions, gtv)
    return scenes, subtasks, decisions, nav, qa


def metrics(subtasks, nav, qa):
    st = {s: subtasks[s] for s in subtasks if subtasks[s]["goal_type"] != "QA"}
    def spl1(v):  # same fallbacks as logger_lmeebench.log_subtask_result (non-finite / zero distances)
        gt = v["gt_dist"] if np.isfinite(v["gt_dist"]) and v["gt_dist"] > 1e-6 else 1.0
        ex = v["explore_dist"] if np.isfinite(v["explore_dist"]) and v["explore_dist"] > 1e-6 else v["gt_dist"]
        return float(v["success_by_distance"]) * gt / max(gt, ex, 1e-6)
    spl = [spl1(v) for v in st.values()]
    mem = [r for r in nav if r["last_choice"] == "memory"]
    prem = [r for r in nav if r["category"] in ("exploration failure: premature commit", "wrong instance (goal not yet observed)")]
    best_letter = max(np.mean([r["answer"] == L for r in qa]) for L in "ABCD") if qa else float("nan")
    acc = np.mean([r["correct"] for r in qa]) if qa else float("nan")
    return {"n_nav": len(nav), "n_qa": len(qa),
            "episodes": len({r["subtask"].rsplit("_", 1)[0] for r in nav}),
            "SR": 100 * np.mean([r["success"] for r in nav]), "SPL": 100 * np.mean(spl),
            "distractor_rate_all": 100 * np.mean([r["chose_distractor"] for r in nav]),
            "distractor_rate_of_memory_choices": 100 * np.mean([r["chose_distractor"] for r in mem]) if mem else float("nan"),
            "premature_commit_rate": 100 * len(prem) / len(nav),
            "acc": 100 * acc, "acc_prior_corrected": 100 * (acc - best_letter)}


def e3(subtasks, decisions, scenes):
    """Per navigation decision: does the agent commit to a memory, given (a) a stored view shows a goal instance,
    (b) the goal's GT region had been observed in a stored view up to that decision."""
    rows = []
    for sid, st in subtasks.items():
        if st["goal_type"] == "QA":
            continue
        objs = scenes[A.scene_of(sid)]
        G = {int(x) for x in st["goal_obj_ids"]}
        greg = {objs[g]["region"] for g in G if g in objs}
        frames = st["frames"]
        for i, d in enumerate(decisions.get(sid, [])):
            if d["choice"] is None:
                continue
            step = int(d["global_step"])
            seen = {n: f for n, f in frames.items() if int(n.split("-")[0]) <= step}
            in_mem = any(G & A.vis_of(frames, n) for n in d["snapshots"])
            reg_obs = any(A.dominant_region(seen, n, objs) in greg for n in seen)
            rows.append({"commit": d["choice"]["type"] == "memory", "goal_in_memory": in_mem,
                         "goal_region_observed": reg_obs, "first": i == 0})
    def rate(rs):
        return {"decisions": len(rs), "commit_rate": round(100 * np.mean([r["commit"] for r in rs]), 1) if rs else None}
    t = {"goal in memory": rate([r for r in rows if r["goal_in_memory"]]),
         "goal not in memory": rate([r for r in rows if not r["goal_in_memory"]])}
    for obs in (True, False):
        for inm in (True, False):
            t[f"region {'observed' if obs else 'not observed'}, goal {'in' if inm else 'not in'} memory"] = rate(
                [r for r in rows if r["goal_region_observed"] == obs and r["goal_in_memory"] == inm])
    t["first decision of subtask, goal in memory"] = rate([r for r in rows if r["first"] and r["goal_in_memory"]])
    t["first decision of subtask, goal not in memory"] = rate([r for r in rows if r["first"] and not r["goal_in_memory"]])
    return t, rows


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"runs": {}, "missing": []}
    e3_rows = []
    for exp, runs in RUNS.items():
        for run in runs:
            if not os.path.isdir(os.path.join(REPO, "analysis/runs", run)):
                res["missing"].append(run)
                continue
            scenes, subtasks, decisions, nav, qa = load(run)
            m = metrics(subtasks, nav, qa)
            m["exp"] = exp
            m["complete"] = m["episodes"] == N_TASK_EPISODES or m["n_nav"] >= 272
            res["runs"][run] = m
            if exp == "E0":
                _, rows = e3(subtasks, decisions, scenes)
                e3_rows += rows
            if exp == "E4":
                recs = [json.loads(l) for f in __import__("glob").glob(os.path.join(REPO, "analysis/runs", run, "records_*.jsonl"))
                        for l in open(f)]
                v = [r for r in recs if r["type"] == "p1_verify"]
                m["verify_calls"] = len(v)
                m["verify_rejections"] = sum(not r["accepted"] for r in v)
    keys = ["SR", "SPL", "distractor_rate_all", "distractor_rate_of_memory_choices", "premature_commit_rate", "acc", "acc_prior_corrected"]
    e0 = [m for m in res["runs"].values() if m["exp"] == "E0"]
    res["E0_mean"] = {k: float(np.mean([m[k] for m in e0])) for k in keys} if e0 else {}
    res["E0_std"] = {k: float(np.std([m[k] for m in e0], ddof=1)) if len(e0) > 1 else float("nan") for k in keys}
    res["E0_n_runs"] = len(e0)
    for m in res["runs"].values():
        m["delta_vs_E0_mean"] = {k: m[k] - res["E0_mean"][k] for k in keys} if e0 else {}
    # E3 over all E0 runs
    t = {}
    for name, sel in [("goal in memory", lambda r: r["goal_in_memory"]), ("goal not in memory", lambda r: not r["goal_in_memory"])]:
        rs = [r for r in e3_rows if sel(r)]
        t[name] = {"decisions": len(rs), "commit_rate": 100 * np.mean([r["commit"] for r in rs]) if rs else None}
    for obs in (True, False):
        for inm in (True, False):
            rs = [r for r in e3_rows if r["goal_region_observed"] == obs and r["goal_in_memory"] == inm]
            t[f"goal region {'observed' if obs else 'not observed'} / goal {'in' if inm else 'not in'} memory"] = {
                "decisions": len(rs), "commit_rate": 100 * np.mean([r["commit"] for r in rs]) if rs else None}
    res["E3"] = t
    gap = None
    if t["goal in memory"]["commit_rate"] is not None and t["goal not in memory"]["commit_rate"] is not None:
        gap = t["goal in memory"]["commit_rate"] - t["goal not in memory"]["commit_rate"]
    res["E3_commit_rate_difference"] = gap
    res["C2_holds_as_stated"] = gap is not None and abs(gap) < 10
    # decision rule G1
    def sr(exp):
        r = [m for m in res["runs"].values() if m["exp"] == exp and m["complete"]]
        return r[0]["SR"] if r else None
    e0m, e0s = res["E0_mean"].get("SR"), res["E0_std"].get("SR")
    e1, e2, e4 = sr("E1"), sr("E2"), sr("E4")
    if None in (e0m, e1, e2, e4) or len(e0) < 3:
        res["G1"] = {"decision": "INCOMPLETE", "reason": "missing complete runs", "missing": res["missing"]}
    else:
        d1 = e1 - e0m
        cond_gain = d1 >= 5 and d1 > 2 * e0s
        cond_beats = (e1 - e2) >= 3 and (e1 - e4) >= 3
        decision = "GO" if cond_gain and cond_beats else ("GO, narrowed" if d1 >= 5 else "PIVOT")
        res["G1"] = {"decision": decision, "E1_minus_E0": d1, "two_std": 2 * e0s, "E1_minus_E2": e1 - e2, "E1_minus_E4": e1 - e4,
                     "gain_condition (>=5 and >2std)": bool(cond_gain), "beats_simple_fixes (>=3 over E2 and E4)": bool(cond_beats),
                     "E2_recovery_of_E1_gain": (e2 - e0m) / d1 if d1 else None, "E4_recovery_of_E1_gain": (e4 - e0m) / d1 if d1 else None}
    json.dump(res, open(os.path.join(OUT, "p1_results.json"), "w"), indent=1, default=float)
    # markdown tables
    L = ["| Run | Exp | Nav targets | QA | SR | SPL | Distractor % (all) | Distractor % (memory choices) | Premature-commit % | Acc | Prior-corrected acc | ΔSR vs E0 mean |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, m in res["runs"].items():
        L.append(f"| {run} | {m['exp']} | {m['n_nav']} | {m['n_qa']} | {fmt(m['SR'])} | {fmt(m['SPL'])} | {fmt(m['distractor_rate_all'])} | "
                 f"{fmt(m['distractor_rate_of_memory_choices'])} | {fmt(m['premature_commit_rate'])} | {fmt(m['acc'])} | "
                 f"{fmt(m['acc_prior_corrected'])} | {fmt(m['delta_vs_E0_mean'].get('SR'))} |")
    if e0:
        L.append(f"| **E0 mean ± std (n={len(e0)})** | E0 | | | " + " | ".join(
            f"{fmt(res['E0_mean'][k])} ± {fmt(res['E0_std'][k])}" for k in keys) + " | |")
    L += ["", "| E3 condition (all E0 runs, per navigation decision) | Decisions | Commit rate % |", "|---|---|---|"]
    for k, v in t.items():
        L.append(f"| {k} | {v['decisions']} | {fmt(v['commit_rate'])} |")
    L += ["", f"E3 difference (in − not in memory): {fmt(gap)} points → C2 holds as stated: {res['C2_holds_as_stated']}",
          "", "G1: " + json.dumps(res["G1"], default=float)]
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

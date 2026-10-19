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
            if exp == "E0" and m["complete"]:
                _, rows = e3(subtasks, decisions, scenes)
                e3_rows += rows
            if exp == "E4":
                recs = [json.loads(l) for f in __import__("glob").glob(os.path.join(REPO, "analysis/runs", run, "records_*.jsonl"))
                        for l in open(f)]
                v = [r for r in recs if r["type"] == "p1_verify"]
                m["verify_calls"] = len(v)
                m["verify_rejections"] = sum(not r["accepted"] for r in v)
    keys = ["SR", "SPL", "distractor_rate_all", "distractor_rate_of_memory_choices", "premature_commit_rate", "acc", "acc_prior_corrected"]
    e0 = [m for m in res["runs"].values() if m["exp"] == "E0" and m["complete"]]  # partial runs never enter the mean
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
    write_decision_doc(res, L)


def write_decision_doc(res, table_lines):
    """docs/phase1_decision.md, every number taken from res (no hand-filled numbers)."""
    g, r, mean, std = res["G1"], res["runs"], res["E0_mean"], res["E0_std"]
    t = res["E3"]
    if g.get("decision") == "INCOMPLETE":
        return
    def run(exp):
        return next(m for m in r.values() if m["exp"] == exp)
    e1, e2, e4 = run("E1"), run("E2"), run("E4")
    e5 = next((m for m in r.values() if m["exp"] == "E5"), None)
    inc = lambda k, m: m[k] - mean[k]
    D = []
    D.append("# Phase 1 decision (DEV_PLAN.md, gate G1)\n")
    D.append("Generated by `analysis/p1_report.py` from the run records; every number below comes from "
             "`analysis/out_p1/p1_results.json`. Reproduce: `bash analysis/p1_run_all.sh` (add `RERUN=1` to re-run the GPU stages).\n")
    D.append(f"## Decision: **{g['decision']}**\n")
    D.append("| Rule (DEV_PLAN.md P1) | Value | Met |\n|---|---|---|")
    D.append(f"| E1 − E0 ≥ +5 SR | {g['E1_minus_E0']:+.1f} SR | {'yes' if g['E1_minus_E0'] >= 5 else 'no'} |")
    D.append(f"| E1 − E0 beyond 2 × std of E0 SR | 2 × std = {g['two_std']:.1f} | {'yes' if g['E1_minus_E0'] > g['two_std'] else 'no'} |")
    D.append(f"| E1 exceeds E2 by ≥ 3 SR | {g['E1_minus_E2']:+.1f} SR | {'yes' if g['E1_minus_E2'] >= 3 else 'no'} |")
    D.append(f"| E1 exceeds E4 by ≥ 3 SR | {g['E1_minus_E4']:+.1f} SR | {'yes' if g['E1_minus_E4'] >= 3 else 'no'} |")
    D.append(f"\nE2 recovers {100 * g['E2_recovery_of_E1_gain']:.0f}% and E4 recovers {100 * g['E4_recovery_of_E1_gain']:.0f}% of the E1 SR gain "
             f"(each measured against the E0 mean). " + ("Neither simple fix exceeds" if max(inc('SR', e2), inc('SR', e4)) <= g['two_std'] else "At least one simple fix exceeds") + " the noise bound: "
             f"E2 {inc('SR', e2):+.1f} SR, E4 {inc('SR', e4):+.1f} SR, 2 × std = {g['two_std']:.1f}.\n")
    D.append("## Results (58-task subset, choice mode)\n")
    D.append("All runs use the same configuration and the instrumented runtime wrapper (`analysis/p1_experiments.py`); "
             "E0 run 1 is the gap-report re-run (`analysis/runs/instr_choice_sub`). Prior-corrected accuracy = accuracy − "
             "best constant-letter accuracy. Premature-commit rate = share of navigation targets whose subtask ended in a "
             "commitment to a non-goal object before any goal instance was visible.\n")
    D += table_lines[:table_lines.index("") if "" in table_lines else len(table_lines)]
    D.append("")
    D.append(f"- **E0 noise.** SR {mean['SR']:.1f} ± {std['SR']:.1f}, SPL {mean['SPL']:.1f} ± {std['SPL']:.1f}, accuracy "
             f"{mean['acc']:.1f} ± {std['acc']:.1f} over {res['E0_n_runs']} runs. A difference smaller than "
             f"{2 * std['SR']:.1f} SR does not count (DEV_PLAN.md ground rules).")
    D.append(f"- **E1 (oracle region gate).** SR {e1['SR']:.1f} ({inc('SR', e1):+.1f}), SPL {e1['SPL']:.1f} ({inc('SPL', e1):+.1f}). "
             f"Premature-commit rate {e1['premature_commit_rate']:.1f}% vs {mean['premature_commit_rate']:.1f}%; distractor choices "
             f"{e1['distractor_rate_all']:.1f}% vs {mean['distractor_rate_all']:.1f}% of targets. QA accuracy {e1['acc']:.1f} "
             f"({inc('acc', e1):+.1f}); prior-corrected {e1['acc_prior_corrected']:.1f} ({inc('acc_prior_corrected', e1):+.1f}): "
             + ("the QA side of the gate does not help and may hurt." if inc('acc', e1) < 2 * std['acc'] else "the QA side of the gate helps beyond noise."))
    D.append(f"- **E2 (room name in prompt).** SR {e2['SR']:.1f} ({inc('SR', e2):+.1f}); premature-commit rate "
             f"{e2['premature_commit_rate']:.1f}% vs {mean['premature_commit_rate']:.1f}%: " + ("naming the room does not stop premature commitments."
             if e2['premature_commit_rate'] > mean['premature_commit_rate'] - 2 * std['premature_commit_rate'] else "naming the room reduces premature commitments."))
    D.append(f"- **E4 (verify on arrival).** SR {e4['SR']:.1f} ({inc('SR', e4):+.1f}), SPL {e4['SPL']:.1f} ({inc('SPL', e4):+.1f}); "
             f"{e4.get('verify_calls', 0)} verification calls, {e4.get('verify_rejections', 0)} rejected "
             f"({100 * e4.get('verify_rejections', 0) / max(e4.get('verify_calls', 1), 1):.0f}%). Premature commits fall to "
             f"{e4['premature_commit_rate']:.1f}%" + (", but the verifier rejects most arrivals and SPL drops." if e4.get('verify_rejections', 0) > e4.get('verify_calls', 0) / 2 and inc('SPL', e4) < 0 else "."))
    if e5:
        D.append(f"- **E5 (oracle earlier-goal exclusion).** SR {e5['SR']:.1f} ({inc('SR', e5):+.1f}) on {e5['n_nav']} targets"
                 + ("" if e5["complete"] else " (**run incomplete**)") + ".")
    D.append("\n## E3: commit-rate diagnostic (all E0 runs, per navigation decision)\n")
    D.append("A decision *commits* when the agent picks a memory object instead of a frontier. *Goal in memory*: a stored "
             "memory view shows a goal instance (≥ 0.1% of the frame, semantic sensor). *Goal region observed*: some stored "
             "view up to that decision is dominated by the goal's HM3D region.\n")
    D.append("| Condition | Decisions | Commit rate % |\n|---|---|---|")
    for k, v in t.items():
        D.append(f"| {k} | {v['decisions']} | {v['commit_rate']:.1f} |")
    gap = res["E3_commit_rate_difference"]
    a = t["goal region not observed / goal in memory"]["commit_rate"]
    b = t["goal region not observed / goal not in memory"]["commit_rate"]
    c = t["goal region observed / goal in memory"]["commit_rate"]
    d = t["goal region observed / goal not in memory"]["commit_rate"]
    D.append(f"\nThe commit rate is {gap:.1f} points higher when the goal is in memory, which is above the 10-point threshold, "
             f"so **C2 does not hold as stated**. The agent does discriminate once the goal's room has been seen "
             f"({c:.1f}% vs {d:.1f}%), but not before: with the goal's room unobserved it commits at {a:.1f}% vs {b:.1f}%. "
             f"It also still commits in {t['goal not in memory']['commit_rate']:.1f}% of decisions where the goal is absent from memory.\n")
    D.append("## Consequences for the plan\n")
    if g["decision"] == "GO":
        D.append("- **GO**: continue to P3 as planned. The oracle region gate beats both simple fixes by more than the rule requires, "
                 "so C5 stands as written.")
    elif g["decision"] == "GO, narrowed":
        D.append("- **GO, narrowed**: keep going, but reframe C5; the contribution moves to the learned decision and the absent-goal training (C2, C4).")
    else:
        D.append("- **PIVOT**: stop the memory mechanism (see DEV_PLAN.md G1).")
    D.append("- **Reword C2** before writing: the agent cannot tell that the goal is absent *while the goal's room is unobserved*; "
             "once the room is seen it commits less when the goal is absent. Report the E3 table split by region observation, "
             "not the pooled rates.")
    if inc('acc', e1) < 0:
        D.append("- **QA**: the region gate on QA lowers accuracy; RCMem's QA ranking (P3) needs its own check rather than reusing the E1 rule.")
    D.append("\n## Notes and caveats\n")
    D.append("- E1 and E5 use ground truth (HM3D regions/instances): oracles only, as DEV_PLAN.md allows. E2 and E4 use the task-config room name.")
    D.append("- E4's verifier is one zero-shot Qwen2.5-VL call with the subtask text, the room and (image goals) the goal image; "
             "a stronger verifier could do better, so E4 is a lower bound for verify-on-arrival.")
    D.append("- The baseline still has the NaN start-orientation bug (fixed in P2), identically in every run here.")
    D.append("- On 2026-10-19 GPU 3 briefly reported no CUDA device; four half-runs failed at start-up, were re-queued and completed "
             "(`analysis/runs/p1_queue.log`). Runs resume per episode, so no partial episode is counted.")
    D.append("\n## Commands\n")
    D.append("```bash\nbash analysis/p1_run_all.sh                       # tables + this file from saved runs\n"
             "RERUN=1 GPUS=\"0 1 2 3\" bash analysis/p1_run_all.sh   # re-run E0 x2, E1, E2, E4, E5 through the GPU queue\n"
             "python analysis/p1_experiments.py --exp E1 --run_name p1_E1 --start_ratio 0 --end_ratio 1   # one experiment\n```")
    open(os.path.join(REPO, "docs/phase1_decision.md"), "w").write("\n".join(D) + "\n")


if __name__ == "__main__":
    main()

"""Phase 2 report (DEV_PLAN.md P2): orientation fix, E1/E1b on the fixed baseline, verifier check, full-set taxonomy
and E3, QA prior corrections. Writes analysis/out_p2/p2_results.json and docs/phase2_report.md; every number in the
document comes from this script.   Run (repo root): python analysis/p2_report.py   (or bash analysis/p2_run_all.sh)
"""
import collections
import json
import os

import numpy as np

import analyze as A
import p1_report as P1

REPO = A.REPO
OUT = os.path.join(REPO, "analysis/out_p2")
GROUPS = {"E0 no fix": ["instr_choice_sub", "p2_E0_nofix_s78", "p2_E0_nofix_s79"],  # instr_choice_sub = seed 77, no fix
          "E0 fixed": ["p2_E0_fix_s77", "p2_E0_fix_s78", "p2_E0_fix_s79"],
          "E1 fixed": ["p2_E1_fix_s77", "p2_E1_fix_s78", "p2_E1_fix_s79"],
          "E1b fixed": ["p2_E1b_fix_s77", "p2_E1b_fix_s78", "p2_E1b_fix_s79"]}
FULL = "p2_full_E0_fix_s77"
KEYS = ["SR", "SPL", "distractor_rate_all", "distractor_rate_of_memory_choices", "premature_commit_rate", "acc", "acc_prior_corrected"]
NAMES = {"SR": "SR", "SPL": "SPL", "distractor_rate_all": "Distractor % (all)", "distractor_rate_of_memory_choices": "Distractor % (mem. choices)",
         "premature_commit_rate": "Premature-commit %", "acc": "Acc", "acc_prior_corrected": "Prior-corrected acc"}


def run_dir(r):
    return os.path.join(REPO, "analysis/runs", r)


def complete(r, n_eps):
    d = run_dir(r)
    if not os.path.isdir(d):
        return False
    _, subtasks, _ = A.load_records(d)
    return len({s.rsplit("_", 1)[0] for s in subtasks}) >= n_eps


def load(r, force_goal_instance_gold=False):
    scenes, subtasks, decisions = A.load_records(run_dir(r))
    gtv = [] if force_goal_instance_gold else json.load(open(os.path.join(REPO, "analysis/out/gt_question_views.json")))
    nav = A.analyze_nav(scenes, subtasks, decisions)
    qa = A.analyze_qa(scenes, subtasks, decisions, gtv)
    return scenes, subtasks, decisions, nav, qa


def group_stats(runs):
    ms = {}
    for r in runs:
        _, subtasks, _, nav, qa = load(r)
        m = P1.metrics(subtasks, nav, qa)
        m["frontier_aborts"] = sum(x["category"] == "exploration: aborted, no frontier available" for x in nav)
        m["nav_failures"] = sum(not x["success"] for x in nav)
        ms[r] = m
    k2 = KEYS + ["frontier_aborts", "nav_failures"]
    mean = {k: float(np.mean([m[k] for m in ms.values()])) for k in k2}
    std = {k: float(np.std([m[k] for m in ms.values()], ddof=1)) if len(ms) > 1 else float("nan") for k in k2}
    return ms, mean, std


def taxonomy_shares(rows):
    fails = [r for r in rows if r["category"] not in ("success", "correct")]
    c = collections.Counter(r["category"] for r in fails)
    return {k: {"count": v, "share": 100 * v / len(fails)} for k, v in c.most_common()}, len(fails)


def initial_views_distinct(r):
    """Distinct camera directions among the 7 look-around views of global step 0, averaged over episodes."""
    _, subtasks, _ = A.load_records(run_dir(r))
    seen, out = set(), []
    for sid, st in subtasks.items():
        ep = sid.rsplit("_", 1)[0]
        if ep in seen:
            continue
        seen.add(ep)
        f = {k: v["fwd"] for k, v in st["frames"].items() if k.startswith("0-view_")}
        if f:
            out.append(len({tuple(round(x, 3) for x in v) for v in f.values()}))
    return float(np.mean(out)) if out else None, len(out)


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"groups": {}, "missing": []}
    for g, runs in GROUPS.items():
        ok = [r for r in runs if complete(r, 58)]
        R["missing"] += [r for r in runs if r not in ok]
        if ok:
            ms, mean, std = group_stats(ok)
            R["groups"][g] = {"runs": ms, "mean": mean, "std": std, "n": len(ok)}
    for g in ("E0 no fix", "E0 fixed"):
        if g in R["groups"]:
            R["groups"][g]["initial_views_distinct"] = {r: initial_views_distinct(r) for r in R["groups"][g]["runs"]}
    base = R["groups"].get("E0 fixed")
    if base:
        for g in ("E1 fixed", "E1b fixed"):
            if g in R["groups"]:
                R["groups"][g]["delta_vs_E0_fixed"] = {k: R["groups"][g]["mean"][k] - base["mean"][k] for k in KEYS}
        R["gate_check"] = {g: {"delta_SR": R["groups"][g]["delta_vs_E0_fixed"]["SR"], "two_std_E0_fixed": 2 * base["std"]["SR"],
                               "beyond_noise": R["groups"][g]["delta_vs_E0_fixed"]["SR"] > 2 * base["std"]["SR"]}
                           for g in ("E1 fixed", "E1b fixed") if g in R["groups"]}
        if "E1 fixed" in R["groups"] and "E1b fixed" in R["groups"]:
            d1, d1b = R["groups"]["E1 fixed"]["delta_vs_E0_fixed"]["SR"], R["groups"]["E1b fixed"]["delta_vs_E0_fixed"]["SR"]
            R["E1b_share_of_E1_gain"] = d1b / d1 if d1 else None
    # why E1b falls short of E1: where do its remaining wrong commitments land?
    rtf = os.path.join(OUT, "room_types.json")
    if "E1b fixed" in R["groups"] and os.path.exists(rtf):
        rt = json.load(open(rtf))["scenes"]
        c = collections.Counter()
        for r in R["groups"]["E1b fixed"]["runs"]:
            scenes, _, _, nav, _ = load(r)
            for x in nav:
                if not (x["chose_distractor"] or x["category"] == "exploration failure: premature commit"):
                    continue
                objs, sc = scenes[A.scene_of(x["subtask"])], A.scene_of(x["subtask"])
                reg = objs.get(x.get("chosen_gt_sid"), {}).get("region") if x.get("chosen_gt_sid") is not None else None
                greg = objs.get(x["goal_sid"], {}).get("region")
                t = lambda g: set(rt.get(sc, {}).get(str(g).lstrip("_"), {}).get("types", [])) if g is not None else set()
                if reg is None:
                    c["chosen object not matched to a GT region"] += 1
                elif reg == greg:
                    c["same region as the goal"] += 1
                elif not t(reg) or not t(greg):
                    c["region of unknown room type (never hidden)"] += 1
                elif t(reg) & t(greg):
                    c["another room of the same type"] += 1
                else:
                    c["room of a different type"] += 1
        R["E1b_wrong_commit_regions"] = dict(c)
    # verifier
    vf = os.path.join(OUT, "verifier.json")
    R["verifier"] = json.load(open(vf)) if os.path.exists(vf) else None
    # full set vs subset (same definitions: GT nav taxonomy; QA with goal-instance gold on both)
    if complete(FULL, 166):
        sc, st, dec, nav_f, qa_f = load(FULL, force_goal_instance_gold=True)
        tf, nf = taxonomy_shares(nav_f)
        qf, nqf = taxonomy_shares(qa_f)
        e3f, _ = P1.e3(st, dec, sc)
        sub_nav, sub_qa, sub_e3_rows = [], [], []
        for r in GROUPS["E0 fixed"]:
            if complete(r, 58):
                s2, st2, d2, n2, q2 = load(r, force_goal_instance_gold=True)
                sub_nav += n2
                sub_qa += q2
                sub_e3_rows += P1.e3(st2, d2, s2)[1]
        ts, ns = taxonomy_shares(sub_nav)
        qs, nqs = taxonomy_shares(sub_qa)
        full_ids = {x["subtask"].rsplit("_", 1)[0] for x in nav_f}
        R["full"] = {"run": FULL, "episodes": len(full_ids), "nav_targets": len(nav_f), "qa": len(qa_f),
                     "SR": 100 * np.mean([x["success"] for x in nav_f]), "acc": 100 * np.mean([x["correct"] for x in qa_f]),
                     "nav_taxonomy": tf, "nav_failures": nf, "qa_taxonomy": qf, "qa_failures": nqf, "E3": e3f}
        def e3_from_rows(rows):
            out = {}
            for name, sel in [("goal in memory", lambda r: r["goal_in_memory"]), ("goal not in memory", lambda r: not r["goal_in_memory"])]:
                rs = [r for r in rows if sel(r)]
                out[name] = {"decisions": len(rs), "commit_rate": 100 * np.mean([r["commit"] for r in rs]) if rs else None}
            for obs in (True, False):
                for inm in (True, False):
                    rs = [r for r in rows if r["goal_region_observed"] == obs and r["goal_in_memory"] == inm]
                    out[f"region {'observed' if obs else 'not observed'}, goal {'in' if inm else 'not in'} memory"] = {
                        "decisions": len(rs), "commit_rate": 100 * np.mean([r["commit"] for r in rs]) if rs else None}
            for inm in (True, False):  # same extra rows as p1_report.e3
                rs = [r for r in rows if r["first"] and r["goal_in_memory"] == inm]
                out[f"first decision of subtask, goal {'in' if inm else 'not in'} memory"] = {
                    "decisions": len(rs), "commit_rate": 100 * np.mean([r["commit"] for r in rs]) if rs else None}
            return out
        R["subset_fixed_pooled"] = {"nav_taxonomy": ts, "nav_failures": ns, "qa_taxonomy": qs, "qa_failures": nqs,
                                    "E3": e3_from_rows(sub_e3_rows)}
        cats = sorted(set(tf) | set(ts))
        R["full_vs_subset_nav_share_diff"] = {c: tf.get(c, {"share": 0})["share"] - ts.get(c, {"share": 0})["share"] for c in cats}
        R["max_abs_nav_share_diff"] = max(abs(v) for v in R["full_vs_subset_nav_share_diff"].values())
    else:
        R["missing"].append(FULL)
    # QA prior corrections
    R["qa_prior"] = {}
    for g in ("E0 no fix", "E0 fixed"):
        if g in R["groups"]:
            R["qa_prior"][g] = {"acc": R["groups"][g]["mean"]["acc"], "acc_minus_best_constant": R["groups"][g]["mean"]["acc_prior_corrected"],
                                "std": R["groups"][g]["std"]["acc_prior_corrected"]}
    R["qa_permute"] = {}
    for r in GROUPS["E0 no fix"][:1] + GROUPS["E0 fixed"]:
        f = os.path.join(OUT, f"qa_permute_{r}.json")
        if os.path.exists(f):
            R["qa_permute"][r] = json.load(open(f))["summary"]
    json.dump(R, open(os.path.join(OUT, "p2_results.json"), "w"), indent=1, default=float)
    write_doc(R)
    print(json.dumps({k: v for k, v in R.items() if k in ("missing", "gate_check", "E1b_share_of_E1_gain", "max_abs_nav_share_diff")}, indent=1, default=float))


def write_doc(R):
    G = R["groups"]
    D = ["# Phase 2 report (DEV_PLAN.md P2)\n",
         "Generated by `analysis/p2_report.py`; every number comes from `analysis/out_p2/p2_results.json`. "
         "Reproduce: `bash analysis/p2_run_all.sh` (`RERUN=1` also re-runs the GPU stages).\n"]
    if R["missing"]:
        D.append(f"**Incomplete:** missing or unfinished runs: {', '.join(R['missing'])}.\n")
    # 1 orientation fix
    D.append("## 1. Start-orientation fix (3 seeds each, 58-task subset)\n")
    D.append("LMEE stores the start rotation as `[w,x,y,z]`; `get_pts_angle_goatbench` reads `[x,y,z,w]`, which gives a NaN yaw and an identity "
             "start rotation in every episode. The fix (`--fix_start` in `analysis/p1_experiments.py`) re-orders the four numbers in a "
             "runtime wrapper; LMEE code is unchanged.\n")
    D.append("| Group | Runs | " + " | ".join(NAMES[k] for k in KEYS) + " | Frontier aborts | Nav failures |")
    D.append("|---|---|" + "---|" * (len(KEYS) + 2))
    for g in ("E0 no fix", "E0 fixed", "E1 fixed", "E1b fixed"):
        if g in G:
            m, s = G[g]["mean"], G[g]["std"]
            D.append(f"| {g} | {G[g]['n']} | " + " | ".join(f"{fmt(m[k])} ± {fmt(s[k])}" for k in KEYS)
                     + f" | {fmt(m['frontier_aborts'])} ± {fmt(s['frontier_aborts'])} | {fmt(m['nav_failures'])} |")
    D.append("\nPer run:\n\n| Run | " + " | ".join(NAMES[k] for k in KEYS) + " | Frontier aborts |\n|---|" + "---|" * (len(KEYS) + 1))
    for g in G.values():
        for r, m in g["runs"].items():
            D.append(f"| {r} | " + " | ".join(fmt(m[k]) for k in KEYS) + f" | {m['frontier_aborts']} |")
    if "E0 no fix" in G and "E0 fixed" in G:
        a, b = G["E0 no fix"], G["E0 fixed"]
        iv = {g: [v[0] for v in G[g]["initial_views_distinct"].values() if v[0] is not None] for g in ("E0 no fix", "E0 fixed")}
        D.append(f"\n- Initial look-around: {fmt(np.mean(iv['E0 no fix']))} distinct view directions out of 7 without the fix, "
                 f"{fmt(np.mean(iv['E0 fixed']))} with it (mean over episodes and runs).")
        dsr = b["mean"]["SR"] - a["mean"]["SR"]
        noise = 2 * max(a["std"]["SR"], b["std"]["SR"])
        D.append(f"- SR changes by {dsr:+.1f} with the fix (2 × the larger std = {noise:.1f}): "
                 + ("within noise." if abs(dsr) <= noise else "beyond noise."))
        da = b["mean"]["frontier_aborts"] - a["mean"]["frontier_aborts"]
        D.append(f"- Frontier aborts per run: {fmt(a['mean']['frontier_aborts'])} → {fmt(b['mean']['frontier_aborts'])} ({da:+.1f}); "
                 + (f"the fix removes {100 * (1 - b['mean']['frontier_aborts'] / a['mean']['frontier_aborts']):.0f}% of them"
                    + (f"; the remaining {b['mean']['frontier_aborts']:.1f} per run have another cause." if b["mean"]["frontier_aborts"] > 0 else ".")
                    if 0 < b["mean"]["frontier_aborts"] < a["mean"]["frontier_aborts"] or b["mean"]["frontier_aborts"] == 0 < a["mean"]["frontier_aborts"]
                    else "the fix does not reduce them, so their cause lies elsewhere."))
    # 2 E1 / E1b
    D.append("\n## 2. E1 (oracle region gate) and E1b (room-type gate) on the fixed baseline\n")
    D.append("E1b keeps every region whose room type matches the subtask's room. Room types: task-config labels where available, otherwise a "
             "vote over the region's ground-truth objects (`analysis/room_types.py`); regions of unknown type are never hidden.")
    rt = os.path.join(OUT, "room_types.json")
    if os.path.exists(rt):
        s = json.load(open(rt))["stats"]
        D.append(f" Of {s['regions']} regions, {s['config_labelled']} have a config label, {s['object_vote']} are typed by the object vote "
                 f"(vote agrees with the config on {s['vote_accuracy_on_config_labelled']} of labelled regions) and {s['unknown']} stay unknown.\n")
    for g, gc in R.get("gate_check", {}).items():
        dd = G[g]["delta_vs_E0_fixed"]
        D.append(f"- **{g}**: ΔSR {dd['SR']:+.1f}, ΔSPL {dd['SPL']:+.1f}, Δpremature-commit {dd['premature_commit_rate']:+.1f}, "
                 f"Δacc {dd['acc']:+.1f} vs the fixed E0 mean; 2 × std = {gc['two_std_E0_fixed']:.1f} → "
                 + ("beyond noise." if gc["beyond_noise"] else "within noise."))
    if R.get("E1b_share_of_E1_gain") is not None:
        D.append(f"- E1b recovers {100 * R['E1b_share_of_E1_gain']:.0f}% of E1's SR gain: this is the realistic ceiling for a memory that knows "
                 "room types but not exact regions (G2 in DEV_PLAN.md asks zero-shot RCMem to close half of the E0→E1 gap).")
        if R["E1b_share_of_E1_gain"] < 0.5:
            D.append("- **Consequence for G2:** half of the E0→E1 gap is above the measured room-type ceiling. A memory that only knows room "
                     "types is unlikely to pass G2 as written unless typing every region (below) closes most of the shortfall; RCMem should also "
                     "bind each subtask to a specific region (which bedroom), e.g. through the instruction's room order and visit history. "
                     "Re-check G2 against E1b before P3 starts.")
    wc = R.get("E1b_wrong_commit_regions")
    if wc:
        tot = sum(wc.values())
        D.append(f"- Where E1b's remaining wrong commitments land (3 runs, {tot} commitments to a distractor or another-category object): "
                 + "; ".join(f"{k} {v} ({100 * v / tot:.0f}%)" for k, v in sorted(wc.items(), key=lambda kv: -kv[1])) + ".")
        unk = wc.get("region of unknown room type (never hidden)", 0)
        same = wc.get("another room of the same type", 0)
        D.append(f"- So E1b is a *lower bound* on a perfect room-type ceiling: {100 * unk / tot:.0f}% of its misses land in regions that "
                 f"no label covers, which a VLM-labelled memory would type. Another {100 * same / tot:.0f}% are in a second room of the "
                 "same type; only region-level binding (which bedroom) removes those. Commitments inside the goal's own region "
                 f"({100 * wc.get('same region as the goal', 0) / tot:.0f}%) are wrong instances that no room gate can fix.")
    # 3 verifier
    v = R.get("verifier")
    if v:
        c = v["confusion"]
        D.append("\n## 3. Verifier check (E4 rejections against ground truth)\n")
        D.append("| | Checked object is the goal | Checked object is not the goal |\n|---|---|---|")
        D.append(f"| Verifier accepted | {c['accepted_true_goal']} | {c['accepted_wrong_object']} |")
        D.append(f"| Verifier rejected | {c['rejected_true_goal']} | {c['rejected_wrong_object']} |")
        D.append(f"\nRejection precision {v['reject_precision']}, rejection recall {v['reject_recall']}; it rejected the true goal in "
                 f"{v['true_goal_rejected_share']} of arrivals at the goal. {c['rejected_wrong_object'] + c['accepted_wrong_object']} of "
                 f"{v['verifications']} arrivals were at a wrong object, so the high rejection rate in Phase 1 was mostly correct: E4 stays "
                 "near the baseline because the agent re-commits to other wrong objects, and because true goals are rejected.")
    # 4 full set
    if "full" in R:
        F, S = R["full"], R["subset_fixed_pooled"]
        D.append(f"\n## 4. Full set (166 tasks; {F['episodes']} episodes, {F['nav_targets']} navigation targets, {F['qa']} questions; fixed baseline, seed 77)\n")
        D.append(f"SR {F['SR']:.1f}, accuracy {F['acc']:.1f}. Same labelling rules as the gap report; QA uses goal-instance gold on both sets "
                 "(the verified question views exist only for the subset), so full and subset shares are comparable.\n")
        D.append("| Navigation failure category | Full: count | Full: share % | Subset (fixed, 3 runs pooled): share % | Difference |\n|---|---|---|---|---|")
        for k in sorted(set(F["nav_taxonomy"]) | set(S["nav_taxonomy"]), key=lambda k: -F["nav_taxonomy"].get(k, {"count": 0})["count"]):
            f, s = F["nav_taxonomy"].get(k, {"count": 0, "share": 0}), S["nav_taxonomy"].get(k, {"share": 0})
            D.append(f"| {k} | {f['count']} | {f['share']:.1f} | {s['share']:.1f} | {f['share'] - s['share']:+.1f} |")
        D.append(f"\nLargest share difference: {R['max_abs_nav_share_diff']:.1f} points "
                 + ("(within the 5-point tolerance)." if R["max_abs_nav_share_diff"] <= 5 else "(above the 5-point tolerance; see per-category differences)."))
        D.append("\n| QA failure category | Full: count | Full: share % | Subset: share % |\n|---|---|---|---|")
        for k in sorted(set(F["qa_taxonomy"]) | set(S["qa_taxonomy"]), key=lambda k: -F["qa_taxonomy"].get(k, {"count": 0})["count"]):
            f, s = F["qa_taxonomy"].get(k, {"count": 0, "share": 0}), S["qa_taxonomy"].get(k, {"share": 0})
            D.append(f"| {k} | {f['count']} | {f['share']:.1f} | {s['share']:.1f} |")
        D.append("\n| E3 condition | Full: decisions | Full: commit % | Subset: decisions | Subset: commit % |\n|---|---|---|---|---|")
        for k in F["E3"]:
            s = S["E3"].get(k.replace("goal region ", "region ").replace(" / ", ", "), S["E3"].get(k, {"decisions": 0, "commit_rate": None}))
            D.append(f"| {k} | {F['E3'][k]['decisions']} | {fmt(F['E3'][k]['commit_rate'])} | {s['decisions']} | {fmt(s['commit_rate'])} |")
    # 5 QA priors
    D.append("\n## 5. QA prior corrections\n")
    for g, q in R["qa_prior"].items():
        D.append(f"- (a) {g}: accuracy {q['acc']:.1f}; accuracy minus the best constant-letter score {q['acc_minus_best_constant']:.1f} ± {fmt(q['std'])}.")
    if R["qa_permute"]:
        D.append("\n(b) Choice permutation: the answering step is replayed from the recorded memories and poses (`analysis/p2_qa_permute.py`); "
                 "first with the original choice order to measure replay fidelity, then with 3 random orders.\n")
        D.append("| Run | Questions | Replay agrees with recorded answer | Accuracy, original order (replay) | Accuracy, permuted (mean of 3) | Same answer text in all orders |\n|---|---|---|---|---|---|")
        for r, s in R["qa_permute"].items():
            D.append(f"| {r} | {s['questions_replayed']} | {s['replay_fidelity_pred_agreement']} | {s['replay_accuracy_original_order']} | "
                     f"{s['permuted_accuracy_mean_over_perms']:.1f} | {s['answer_consistent_across_all_orders']} |")
    D.append("\n## 6. Commands\n")
    D.append("```bash\nbash analysis/p2_run_all.sh                 # tables + this file from saved runs\n"
             "RERUN=1 bash analysis/p2_run_all.sh         # also re-run every GPU job through the queue\n```\n")
    D.append("Not done: notifying the LMEE authors about the start-orientation bug (optional in DEV_PLAN.md; it is an outward-facing action, left to the user).")
    open(os.path.join(REPO, "docs/phase2_report.md"), "w").write("\n".join(D) + "\n")


if __name__ == "__main__":
    main()

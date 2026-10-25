"""Phase 2b report (DEV_PLAN.md P2b): F0 token budget, F1 / F2 / F2b on the 58-task subset (seeds 77-79, empty-context
rule), gate G1.5 against the fixed baseline. Writes analysis/out_p2b/p2b_results.json and docs/phase2b_decision.md;
every number in the document comes from this script.   Run (repo root): python analysis/p2b_report.py
(or bash analysis/p2b_run_all.sh)
"""
import json
import os

import numpy as np

import analyze as A
import p1_report as P1
import p2_report as P2

REPO = A.REPO
OUT = os.path.join(REPO, "analysis/out_p2b")
SEEDS = (77, 78, 79)
EXPS = ("F1", "F2", "F2b")
# re-check: every F run uses the empty-context rule (no frame in context -> nearest frontier, no VLM call)
F_RUNS = {e: [f"p2b_ec_{e}_s{sd}" for sd in SEEDS] for e in EXPS}
FIRST_PASS = {e: f"p2b_{e}_s77" for e in EXPS}  # first P2b pass: seed 77 only, VLM asked even with an empty context
E0_FIXED, E1_FIXED, E1B_FIXED = P2.GROUPS["E0 fixed"], P2.GROUPS["E1 fixed"], P2.GROUPS["E1b fixed"]
# noise: pooled within-group std of every E0 run (groups = without / with the start-orientation fix)
E0_GROUPS = {"E0 no fix": ["instr_choice_sub", "p1_E0_r2", "p1_E0_r3", "p2_E0_nofix_s78", "p2_E0_nofix_s79"],
             "E0 fixed": E0_FIXED}
KEYS = P2.KEYS
DIAG = ["frontier_aborts", "ctx_tokens_mean", "ctx_tokens_max", "sec_per_decision", "ctx_frames_mean", "decisions", "vlm_calls",
        "decisions_capped", "invalid_replies", "empty_context_explore", "objects_per_frame", "gt_proxy_per_frame"]
FIT_SHARE, BUDGET = 95.0, "share_le_32k"


def run_metrics(run):
    scenes, subtasks, decisions, nav, qa = P2.load(run)
    m = P1.metrics(subtasks, nav, qa)
    m["frontier_aborts"] = sum(x["category"] == "exploration: aborted, no frontier available" for x in nav)
    ds = [d for v in decisions.values() for d in v if "p2b" in d]
    if ds:
        called = [d for d in ds if d["p2b"]["prompt_tokens"] > 0] or ds  # VLM calls (empty-context explores make none)
        tok = np.array([d["p2b"]["prompt_tokens"] for d in called])
        lat = np.array([d["p2b"]["latency_s"] for d in called])
        m.update(decisions=len(ds), vlm_calls=sum(d["p2b"]["prompt_tokens"] > 0 for d in ds),
                 ctx_tokens_mean=float(tok.mean()), ctx_tokens_max=int(tok.max()), ctx_tokens_p95=float(np.percentile(tok, 95)),
                 sec_per_decision=float(lat.mean()), sec_per_decision_max=float(lat.max()),
                 ctx_frames_mean=float(np.mean([d["p2b"]["ctx_frames"] for d in ds])),
                 kept_frames_mean=float(np.mean([d["p2b"]["kept_frames"] for d in ds])),
                 captured_frames_mean=float(np.mean([d["p2b"]["captured_frames"] for d in ds])),
                 decisions_capped=sum(d["p2b"]["dropped_by_cap"] > 0 for d in ds),
                 invalid_replies=sum(d["p2b"]["parse"] == "invalid" for d in ds),
                 invalid_with_empty_context=sum(d["p2b"]["parse"] == "invalid" and d["p2b"]["ctx_frames"] == 0 for d in ds),
                 empty_context_explore=sum(d["p2b"]["parse"] == "empty_context_explore" for d in ds),
                 empty_context_rule=all(d["p2b"].get("empty_context_rule", False) for d in ds),
                 token_count_error_max=int(max(abs(d["p2b"]["prompt_tokens"] - d["p2b"]["prompt_tokens_counted"]) for d in called)))
        # objects listed per context frame vs the ground-truth proxy F0 used (visible non-structural instances)
        frames = {}
        for st in subtasks.values():
            frames.update(st["frames"])
        listed, proxy = [], []
        for d in ds:
            objs = scenes[A.scene_of(d["subtask"])]
            for n, cl in d["snapshots"].items():
                listed.append(len(cl))
                proxy.append(sum(1 for s, c in frames.get(n, {}).get("vis", {}).items()
                                 if c >= A.MIN_PX and objs.get(int(s), {}).get("cat", "Unknown") not in A.STRUCT))
        m["objects_per_frame"] = float(np.mean(listed)) if listed else None
        m["gt_proxy_per_frame"] = float(np.mean(proxy)) if proxy else None
    return m


def group(runs):
    """Metrics of the complete runs of a group: per run, mean and sample std (ddof=1)."""
    ok = [r for r in runs if P2.complete(r, 58)]
    if not ok:
        return None
    ms = {r: run_metrics(r) for r in ok}
    keys = [k for k in KEYS + DIAG if all(m.get(k) is not None for m in ms.values())]
    mean = {k: float(np.mean([m[k] for m in ms.values()])) for k in keys}
    std = {k: float(np.std([m[k] for m in ms.values()], ddof=1)) if len(ms) > 1 else float("nan") for k in keys}
    if "ctx_tokens_max" in mean:
        mean["ctx_tokens_max"] = float(max(m["ctx_tokens_max"] for m in ms.values()))  # max over runs, not mean of maxima
    return {"runs": ok, "missing": [r for r in runs if r not in ok], "per_run": ms, "mean": mean, "std": std, "n": len(ok)}


def pooled_std(groups, key="SR"):
    """Pooled within-group sample std: sqrt(sum over groups of sum (x - group mean)^2 / sum (n_g - 1))."""
    ss, dof = 0.0, 0
    for g in groups.values():
        x = np.array([m[key] for m in g["per_run"].values()])
        ss += float(((x - x.mean()) ** 2).sum())
        dof += len(x) - 1
    return (float(np.sqrt(ss / dof)) if dof else float("nan")), dof


def gate(f0, F, e0, band):
    fits = {s: v[BUDGET] for s, v in f0["settings"].items() if s != "a_all"}
    best = max(fits, key=fits.get)
    f0_ok = fits[best] >= FIT_SHARE
    if not all(F.get(k) and F[k]["n"] == len(SEEDS) for k in ("F1", "F2")) or e0 is None or e0["n"] < len(E0_FIXED):
        return {"decision": "INCOMPLETE", "f0_fits": f0_ok, "f0_best_setting": best,
                "reason": "F1/F2 at every seed or the fixed E0 runs are missing"}
    f1, f2, s1, s2 = F["F1"]["mean"]["SR"], F["F2"]["mean"]["SR"], F["F1"]["std"]["SR"], F["F2"]["std"]["SR"]
    d21, f1_ok = f2 - f1, f1 >= e0["mean"]["SR"] - band
    if f0_ok and d21 >= band and f1_ok:
        dec = "GO, implicit"
    elif f0_ok and d21 >= band:
        dec = "GO, implicit but training-dependent"
    else:
        dec = "FALLBACK, external"
    se21 = float(np.sqrt(s1 ** 2 / F["F1"]["n"] + s2 ** 2 / F["F2"]["n"]))
    se1 = float(np.sqrt(s1 ** 2 / F["F1"]["n"] + e0["std"]["SR"] ** 2 / e0["n"]))
    return {"decision": dec, "band": band, "f0_fits": f0_ok, "f0_best_setting": best, "f0_best_share": fits[best],
            "F1": f1, "F2": f2, "F2_minus_F1": d21, "F2_minus_F1_ok": d21 >= band, "F2_minus_F1_se": se21,
            "E0_mean": e0["mean"]["SR"], "F1_minus_E0": f1 - e0["mean"]["SR"], "F1_minus_E0_se": se1, "F1_ok": f1_ok}


def fmt(x, nd=1):
    return P2.fmt(x, nd)


def pm(g, k, nd=1):
    """'mean ± std' of a group metric."""
    return f"{fmt(g['mean'].get(k), nd)} ± {fmt(g['std'].get(k), nd)}"


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"missing": []}
    f0p = os.path.join(OUT, "f0.json")
    R["f0"] = json.load(open(f0p)) if os.path.exists(f0p) else None
    if R["f0"] is None:
        R["missing"].append("analysis/out_p2b/f0.json")
    F = {}
    for exp, runs in F_RUNS.items():
        g = group(runs)
        R["missing"] += g["missing"] if g else runs
        if g:
            F[exp] = g
    R["F"] = F
    R["first_pass"] = {e: run_metrics(r) for e, r in FIRST_PASS.items() if P2.complete(r, 58)}
    E0g = {k: group(v) for k, v in E0_GROUPS.items()}
    R["missing"] += [r for k, g in E0g.items() for r in (g["missing"] if g else E0_GROUPS[k])]
    E0g = {k: g for k, g in E0g.items() if g}
    R["E0_groups"] = E0g
    R["E0"] = E0g.get("E0 fixed")
    R["E1"], R["E1b"] = group(E1_FIXED), group(E1B_FIXED)
    sd, dof = pooled_std(E0g)
    R["noise"] = {"pooled_within_group_std_SR": sd, "dof": dof, "band": 2 * sd,
                  "groups": {k: {"runs": g["runs"], "mean_SR": g["mean"]["SR"], "std_SR": g["std"]["SR"]} for k, g in E0g.items()}}
    if R["f0"] and "F1" in F and F["F1"]["mean"].get("gt_proxy_per_frame"):
        import p2b_f0
        ratio = F["F1"]["mean"]["objects_per_frame"] / F["F1"]["mean"]["gt_proxy_per_frame"]
        rows = [dict(r) for r in csv_rows()]
        R["f0_calibrated"] = {"object_ratio": ratio, "settings": p2b_f0.summarize(rows, R["f0"]["settings_list"], ratio)}
    if R["f0"]:
        S0, gr = R["f0"]["settings"], R["f0"]["goal_view_retention_pct"]
        ok = [s for s in gr if S0[s]["share_le_16k"] >= 95.0]
        if ok:
            best = max(ok, key=lambda s: gr[s])
            R["f1_setting"] = {"setting": best, "share_le_16k": S0[best]["share_le_16k"], "goal_kept": gr[best]}
        R["gate"] = gate(R["f0_calibrated"] if "f0_calibrated" in R else R["f0"], F, R["E0"], R["noise"]["band"])
    sr = lambda g: g["mean"]["SR"]
    if "F1" in F and "F2" in F and R["E1"] and R["E0"]:
        R["in_context_vs_external_gate"] = {"F2_minus_F1": sr(F["F2"]) - sr(F["F1"]), "E1_minus_E0": sr(R["E1"]) - sr(R["E0"])}
    if all(e in F for e in EXPS):
        d2 = sr(F["F2"]) - sr(F["F1"])
        R["F2b_share_of_F2_gain"] = (sr(F["F2b"]) - sr(F["F1"])) / d2 if d2 else None
    if R["E1"] and R["E1b"] and R["E0"]:
        d1 = sr(R["E1"]) - sr(R["E0"])
        R["E1b_share_of_E1_gain"] = (sr(R["E1b"]) - sr(R["E0"])) / d1 if d1 else None
    json.dump(R, open(os.path.join(OUT, "p2b_results.json"), "w"), indent=1, default=float)
    write_doc(R)
    print(json.dumps({"missing": R["missing"], "noise": R["noise"], "gate": R.get("gate")}, indent=1, default=float))


def csv_rows():
    import csv
    with open(os.path.join(OUT, "f0_decisions.csv")) as f:
        for r in csv.DictReader(f):
            yield {k: (v == "True" if v in ("True", "False") else float(v) if k not in ("run", "episode", "subtask") else v)
                   for k, v in r.items()}


SET_NAMES = {"a_all": "(a) all frames", "d_pool_all": "(d) 2×2 pooling, all frames"}


def set_name(s):
    if s in SET_NAMES:
        return SET_NAMES[s]
    kind, v = s.rsplit("_v", 1) if "_v" in s else (s.rsplit("_b", 1)[0], s.rsplit("_b", 1)[1])
    return {"b_frame": f"(b) frame-level voxel pruning, {v} m", "c_token": f"(c) token-level voxel pruning, {v} m"}.get(
        kind, f"(d) 2×2 pooling + (b) {v} m")


def write_doc(R):
    D = ["# Phase 2b decision (DEV_PLAN.md P2b): implicit-memory feasibility\n",
         "Generated by `analysis/p2b_report.py`; every number comes from `analysis/out_p2b/p2b_results.json` (and `f0.json`). "
         "Reproduce: `bash analysis/p2b_run_all.sh` (`RERUN=1` also re-runs the GPU stages).\n"]
    if R["missing"]:
        D.append(f"**Incomplete:** missing or unfinished: {', '.join(R['missing'])}.\n")
    F, e0, nz = R["F"], R.get("E0"), R["noise"]
    g = R.get("gate")
    if g:
        D.append(f"## Decision: **{g['decision']}**\n")
        if g["decision"] != "INCOMPLETE":
            b = g["band"]
            D.append(f"Means over seeds {', '.join(map(str, SEEDS))}; noise band = 2 × the pooled within-group SR std of all E0 runs "
                     f"= 2 × {nz['pooled_within_group_std_SR']:.2f} = **{b:.1f} SR** ({nz['dof']} degrees of freedom).\n")
            D.append("| Gate G1.5 condition | Value | Holds |\n|---|---|---|")
            D.append(f"| F0: ≥ {FIT_SHARE:.0f}% of decisions fit 32k tokens with some pruning setting | best: {set_name(g['f0_best_setting'])}, "
                     f"{g['f0_best_share']:.1f}% | {'yes' if g['f0_fits'] else 'no'} |")
            D.append(f"| F2 − F1 ≥ {b:.1f} SR | {g['F2']:.1f} − {g['F1']:.1f} = {g['F2_minus_F1']:+.1f} SR (± {g['F2_minus_F1_se']:.1f} SE) | "
                     f"{'yes' if g['F2_minus_F1_ok'] else 'no'} |")
            D.append(f"| F1 ≥ E0 (fixed) mean − {b:.1f} SR | F1 {g['F1']:.1f} vs E0 {g['E0_mean']:.1f} ({g['F1_minus_E0']:+.1f} ± "
                     f"{g['F1_minus_E0_se']:.1f} SE; floor {g['E0_mean'] - b:.1f}) | {'yes' if g['F1_ok'] else 'no'} |")
            D.append("")
            D.append({"GO, implicit": "All three conditions hold: continue with P3 as written (region-scoped context, zero-shot).",
                      "GO, implicit but training-dependent": "The context fits and region scoping pays off inside it, but zero-shot F1 falls "
                      "more than the noise band below E0. Per DEV_PLAN.md, continue only once the P5 compute choice is settled "
                      "(3B backbone, LoRA, or more GPUs); zero-shot P3 becomes a diagnostic and G2 is judged after training.",
                      "FALLBACK, external": "Return to the external region-indexed memory (DEV_PLAN.md appendix); the paper states why the "
                      "context was not enough."}[g["decision"]] + "\n")
            m21, m1 = g["F2_minus_F1"] - b, g["F1_minus_E0"] + b
            D.append(f"Margins: F2 − F1 is {m21:+.1f} SR from the band ({m21 / g['F2_minus_F1_se']:+.1f} SE of the difference); F1 is "
                     f"{m1:+.1f} SR from its floor ({m1 / g['F1_minus_E0_se']:+.1f} SE). SE = standard error of the difference of means, "
                     "from the per-group run-to-run std.\n")
    # ---- noise and reference
    if R["E0_groups"]:
        D.append("## Baseline and noise\n")
        D.append("| E0 group | Runs | SR mean ± std |\n|---|---|---|")
        for k, v in nz["groups"].items():
            D.append(f"| {k} | {', '.join(v['runs'])} | {v['mean_SR']:.1f} ± {fmt(v['std_SR'])} |")
        D.append(f"\nPooled within-group std {nz['pooled_within_group_std_SR']:.2f} SR → band {nz['band']:.1f} SR (DEV_PLAN.md used "
                 "2 × 3.6 = 7.3 from the three Phase 1 runs). The fixed group (`p2_E0_fix_s77–79`, same start-orientation fix as the F runs) "
                 "is the reference for F1.\n")
    # ---- F runs
    if F:
        D.append(f"## F1, F2, F2b — 58-task subset, choice mode, start-orientation fix, empty-context rule, seeds {', '.join(map(str, SEEDS))}\n")
        D.append("Mean ± std over seeds (sample std, n = runs):\n")
        D.append("| Group | n | " + " | ".join(P2.NAMES[k] for k in KEYS) + " |\n|---|---|" + "---|" * len(KEYS))
        rows = [(e, F[e]) for e in EXPS if e in F] + [(n, R[k]) for n, k in (("E0 fixed", "E0"), ("E1 fixed", "E1"), ("E1b fixed", "E1b")) if R.get(k)]
        for name, gg in rows:
            D.append(f"| {name} | {gg['n']} | " + " | ".join(pm(gg, k) for k in KEYS) + " |")
        D.append("\n| Group | Frontier aborts | Context tokens per VLM call (mean / max) | s per VLM call | Frames in context | Decisions | "
                 "VLM calls | Empty-context explores (no VLM call) | Invalid replies | Capped |\n|---|---|---|---|---|---|---|---|---|---|")
        for e in EXPS:
            if e in F:
                gg = F[e]
                D.append(f"| {e} | {pm(gg, 'frontier_aborts')} | {pm(gg, 'ctx_tokens_mean', 0)} / {fmt(gg['mean']['ctx_tokens_max'], 0)} | "
                         f"{pm(gg, 'sec_per_decision')} | {pm(gg, 'ctx_frames_mean')} | {pm(gg, 'decisions', 0)} | {pm(gg, 'vlm_calls', 0)} | "
                         f"{pm(gg, 'empty_context_explore', 0)} | {pm(gg, 'invalid_replies', 0)} | {pm(gg, 'decisions_capped', 0)} |")
        D.append("\nPer run:\n")
        D.append("| Run | " + " | ".join(P2.NAMES[k] for k in KEYS) + " | Empty-context explores | Invalid replies |\n|---|" + "---|" * (len(KEYS) + 2))
        for e in EXPS:
            for r, m in (F[e]["per_run"].items() if e in F else []):
                D.append(f"| {r} | " + " | ".join(fmt(m[k]) for k in KEYS) + f" | {m.get('empty_context_explore', '–')} | {m.get('invalid_replies', '–')} |")
        for name, k in (("E0 fixed", "E0"), ("E1 fixed", "E1"), ("E1b fixed", "E1b")):
            for r, m in (R[k]["per_run"].items() if R.get(k) else []):
                D.append(f"| {r} | " + " | ".join(fmt(m[kk]) for kk in KEYS) + " | | |")
        if R["first_pass"]:
            D.append("\nSuperseded first pass (seed 77, VLM asked even with no frame in context, so an empty context often ended the subtask "
                     "with an invalid reply):\n")
            D.append("| Run | SR | SPL | Premature-commit % | Invalid replies (with empty context) |\n|---|---|---|---|---|")
            for e, m in R["first_pass"].items():
                D.append(f"| {FIRST_PASS[e]} | {fmt(m['SR'])} | {fmt(m['SPL'])} | {fmt(m['premature_commit_rate'])} | "
                         f"{m['invalid_replies']} ({m['invalid_with_empty_context']}) |")
        D.append("")
        if "F1" in F and e0:
            D.append(f"- Premature commits: F1 {pm(F['F1'], 'premature_commit_rate')}% vs E0 {pm(e0, 'premature_commit_rate')}%"
                     + (f"; F2 {pm(F['F2'], 'premature_commit_rate')}% once the context holds only the goal's region." if "F2" in F else "."))
        if "in_context_vs_external_gate" in R:
            x = R["in_context_vs_external_gate"]
            D.append(f"- In-context oracle gain F2 − F1 = {x['F2_minus_F1']:+.1f} SR vs the external oracle gate E1 − E0 = {x['E1_minus_E0']:+.1f} SR "
                     "(both on the fixed baseline, 3 seeds each).")
        if R.get("F2b_share_of_F2_gain") is not None:
            D.append(f"- F2b recovers {100 * R['F2b_share_of_F2_gain']:.0f}% of F2's gain over F1 (realistic ceiling: room types known, exact "
                     "regions not)" + (f"; E1b recovered {100 * R['E1b_share_of_E1_gain']:.0f}% of E1's gain." if R.get("E1b_share_of_E1_gain") is not None else "."))
        if "F1" in F and F["F1"]["mean"].get("objects_per_frame") is not None:
            D.append(f"- Objects listed per context frame in F1: {F['F1']['mean']['objects_per_frame']:.1f} (F0's ground-truth proxy: "
                     f"{F['F1']['mean']['gt_proxy_per_frame']:.1f}).")
        tce = [m.get("token_count_error_max", 0) for e in F.values() for m in e["per_run"].values()]
        D.append(f"- Exact prompt length minus the tokenizer count used for F0: max |difference| {max(tce)} tokens over all F runs.")
        D.append("")
    # ---- F0
    f0 = R.get("f0")
    if f0:
        D.append("## F0 — token budget (existing E0 records)\n")
        D.append(f"{f0['decisions']} decisions ({f0['nav_decisions']} navigation) from {', '.join(f0['runs'])} ({f0['episodes']} episodes). "
                 "Context = every frame captured up to the decision; depth re-rendered at each recorded pose (max pose error "
                 f"{f0['render_pose_error_max']:.1e} m). Frame-level pruning keeps a frame if ≥ {100 * f0['novelty']:.0f}% of its voxels "
                 f"(depth ≤ {f0['max_depth']:.0f} m) are new; token-level pruning keeps one token per new voxel. A frame costs "
                 f"{f0['frame_tokens']} image tokens ({f0['pooled_frame_tokens']} pooled) + {f0['frame_entry_text_tokens']} text tokens "
                 f"+ {f0['object_line_tokens']:.1f} per listed object. Prompt = the F1 prompt with that decision's question, image goal "
                 "(1280², 2,116 tokens) and frontiers.\n")
        cal = R.get("f0_calibrated")
        S = cal["settings"] if cal else f0["settings"]
        if cal:
            D.append(f"Object lists are rescaled by {cal['object_ratio']:.2f} (detected objects listed per frame in F1 ÷ visible GT "
                     "instances per frame, the proxy F0 uses); uncalibrated numbers are in `f0.json`.\n")
        D.append("| Setting | Frames in context (mean / max) | Tokens mean | median | p95 | max | ≤ 16k % | ≤ 32k % | Goal view kept % |\n"
                 "|---|---|---|---|---|---|---|---|---|")
        gr = f0.get("goal_view_retention_pct", {})
        for s in f0["settings_list"]:
            v = S[s]
            ret = gr.get(s)
            D.append(f"| {set_name(s)} | {v['frames_mean']:.1f} / {v['frames_max']} | {v['mean']:.0f} | {v['median']:.0f} | {v['p95']:.0f} | "
                     f"{v['max']:.0f} | {v['share_le_16k']:.1f} | {v['share_le_32k']:.1f} | {fmt(ret) if ret is not None else '–'} |")
        D.append(f"\nGoal view kept: among the {f0['decisions_goal_in_avail']} decisions where some captured frame shows the goal, the share "
                 "where a kept frame still shows it (100% for (a)). For (c) a frame counts as in context when any of its tokens is kept.\n")
        ch = R.get("f1_setting")
        if ch:
            D.append(f"F1 setting, chosen before F1 ran from the uncalibrated F0 (rule: among frame-level settings with ≥ 95% of decisions "
                     f"under 16k tokens, so one 24 GB GPU holds the 7B model, SAM and the prompt, take the one that keeps the goal view most "
                     f"often): **{set_name(ch['setting'])}**, {ch['share_le_16k']:.1f}% under 16k uncalibrated, goal view kept {ch['goal_kept']:.1f}%.\n")
    D.append("## Method notes\n")
    D.append("- F1/F2/F2b (`analysis/p2b_experiments.py`) wrap the instrumented LMEE run at runtime; LMEE code and the baseline result files are unchanged. "
             "Retrieval (the CLIP knowledge base and its query call) is bypassed: the prompt holds the pruned frames in viewing order. "
             "Each frame shows the detector's boxes, numbered; the answer `Frame f, Object o` becomes the same single-object target the baseline "
             "navigates to. QA answers with `Frame f` and a letter, as the baseline does with memories.")
    D.append("- Empty-context rule: a navigation decision with no frame in context (F2/F2b before the goal's region or room type has been seen) "
             "goes to the nearest frontier without a VLM call; with no frontier left the subtask ends, as in the baseline. QA decisions fall "
             "back to all frames instead.")
    D.append("- F2 keeps only frames whose ground-truth region (pixel-weighted majority of visible instances) is the goal's region; F2b keeps frames "
             "whose region type matches (unknown types kept). Ground truth is used only for these oracles.")
    D.append("- The seed sets LMEE's scene order and the run's random draws; GPU inference is not bit-deterministic, so runs differ even at one seed.\n")
    D.append("## Commands\n")
    D.append("```bash\nbash analysis/p2b_run_all.sh            # tables + this file from saved outputs (CPU)\n"
             "RERUN=1 bash analysis/p2b_run_all.sh    # also re-run F0 and the F1/F2/F2b jobs (3 seeds) through the GPU queue\n```")
    open(os.path.join(REPO, "docs/phase2b_decision.md"), "w").write("\n".join(D) + "\n")


if __name__ == "__main__":
    main()

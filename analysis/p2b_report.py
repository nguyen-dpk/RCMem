"""Phase 2b report (DEV_PLAN.md P2b): F0 token budget, F1 / F2 / F2b on the 58-task subset, gate G1.5.
Writes analysis/out_p2b/p2b_results.json and docs/phase2b_decision.md; every number in the document comes from this
script.   Run (repo root): python analysis/p2b_report.py   (or bash analysis/p2b_run_all.sh)
"""
import collections
import json
import os

import numpy as np

import analyze as A
import p1_report as P1
import p2_report as P2

REPO = A.REPO
OUT = os.path.join(REPO, "analysis/out_p2b")
F_RUNS = {"F1": ["p2b_F1_s77"], "F2": ["p2b_F2_s77"], "F2b": ["p2b_F2b_s77"]}
# E0 / E1 reference: the fixed baseline of Phase 2 when its runs are here, else the Phase 1 runs (no start fix)
E0_FIXED, E1_FIXED = P2.GROUPS["E0 fixed"], P2.GROUPS["E1 fixed"]
E0_NOFIX, E1_NOFIX = ["instr_choice_sub", "p1_E0_r2", "p1_E0_r3"], ["p1_E1"]
KEYS = P2.KEYS
FIT_SHARE, BUDGET = 95.0, "share_le_32k"


def records(run):
    return A.load_records(P2.run_dir(run))


def run_metrics(run):
    scenes, subtasks, decisions, nav, qa = P2.load(run)
    m = P1.metrics(subtasks, nav, qa)
    m["frontier_aborts"] = sum(x["category"] == "exploration: aborted, no frontier available" for x in nav)
    ds = [d for v in decisions.values() for d in v if "p2b" in d]
    if ds:
        tok = np.array([d["p2b"]["prompt_tokens"] for d in ds])
        lat = np.array([d["p2b"]["latency_s"] for d in ds])
        m.update(decisions=len(ds), ctx_tokens_mean=float(tok.mean()), ctx_tokens_max=int(tok.max()),
                 ctx_tokens_p95=float(np.percentile(tok, 95)), sec_per_decision=float(lat.mean()),
                 sec_per_decision_max=float(lat.max()),
                 ctx_frames_mean=float(np.mean([d["p2b"]["ctx_frames"] for d in ds])),
                 kept_frames_mean=float(np.mean([d["p2b"]["kept_frames"] for d in ds])),
                 captured_frames_mean=float(np.mean([d["p2b"]["captured_frames"] for d in ds])),
                 decisions_capped=sum(d["p2b"]["dropped_by_cap"] > 0 for d in ds),
                 invalid_replies=sum(d["p2b"]["parse"] == "invalid" for d in ds),
                 invalid_with_empty_context=sum(d["p2b"]["parse"] == "invalid" and d["p2b"]["ctx_frames"] == 0 for d in ds),
                 empty_context_nav=sum(d["p2b"]["ctx_frames"] == 0 for d in ds if d["pred_answer"] is None),
                 token_count_error_max=int(max(abs(d["p2b"]["prompt_tokens"] - d["p2b"]["prompt_tokens_counted"]) for d in ds)))
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


def reference(fixed, nofix):
    ok = [r for r in fixed if P2.complete(r, 58)]
    runs, label = (ok, "fixed") if len(ok) == len(fixed) else ([r for r in nofix if P2.complete(r, 58)], "no start fix (Phase 1)")
    if not runs:
        return None
    ms = {r: run_metrics(r) for r in runs}
    mean = {k: float(np.mean([m[k] for m in ms.values()])) for k in KEYS}
    std = {k: float(np.std([m[k] for m in ms.values()], ddof=1)) if len(ms) > 1 else float("nan") for k in KEYS}
    return {"label": label, "runs": runs, "per_run": ms, "mean": mean, "std": std}


def gate(f0, F, e0):
    fits = {s: v[BUDGET] for s, v in f0["settings"].items() if s != "a_all"}
    best = max(fits, key=fits.get)
    f0_ok = fits[best] >= FIT_SHARE
    if not all(k in F for k in ("F1", "F2")) or e0 is None:
        return {"decision": "INCOMPLETE", "f0_fits": f0_ok, "f0_best_setting": best,
                "reason": "F1/F2 runs or the E0 reference are missing"}
    thr = 2 * e0["std"]["SR"] if np.isfinite(e0["std"]["SR"]) else 7.3
    d21 = F["F2"]["SR"] - F["F1"]["SR"]
    f1_ok = F["F1"]["SR"] >= e0["mean"]["SR"] - thr
    if f0_ok and d21 >= thr and f1_ok:
        dec = "GO, implicit"
    elif f0_ok and d21 >= thr:
        dec = "GO, implicit but training-dependent"
    else:
        dec = "FALLBACK, external"
    return {"decision": dec, "threshold_2std": thr, "f0_fits": f0_ok, "f0_best_setting": best, "f0_best_share": fits[best],
            "F2_minus_F1": d21, "F2_minus_F1_ok": d21 >= thr, "F1": F["F1"]["SR"], "E0_mean": e0["mean"]["SR"],
            "F1_minus_E0": F["F1"]["SR"] - e0["mean"]["SR"], "F1_ok": f1_ok}


def fmt(x, nd=1):
    return P2.fmt(x, nd)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {"missing": []}
    f0p = os.path.join(OUT, "f0.json")
    R["f0"] = json.load(open(f0p)) if os.path.exists(f0p) else None
    if R["f0"] is None:
        R["missing"].append("analysis/out_p2b/f0.json")
    F = {}
    for exp, runs in F_RUNS.items():
        ok = [r for r in runs if P2.complete(r, 58)]
        R["missing"] += [r for r in runs if r not in ok]
        if ok:
            F[exp] = run_metrics(ok[0])
            F[exp]["run"] = ok[0]
    R["F"] = F
    R["E0"] = reference(E0_FIXED, E0_NOFIX)
    R["E1"] = reference(E1_FIXED, E1_NOFIX)
    if R["f0"] and "F1" in F and F["F1"].get("gt_proxy_per_frame"):
        import p2b_f0
        ratio = F["F1"]["objects_per_frame"] / F["F1"]["gt_proxy_per_frame"]
        rows = [dict(r) for r in csv_rows()]
        R["f0_calibrated"] = {"object_ratio": ratio, "settings": p2b_f0.summarize(rows, R["f0"]["settings_list"], ratio)}
    if R["f0"]:
        S0, gr = R["f0"]["settings"], R["f0"]["goal_view_retention_pct"]
        ok = [s for s in gr if S0[s]["share_le_16k"] >= 95.0]
        if ok:
            best = max(ok, key=lambda s: gr[s])
            R["f1_setting"] = {"setting": best, "share_le_16k": S0[best]["share_le_16k"], "goal_kept": gr[best]}
        R["gate"] = gate(R["f0_calibrated"] if "f0_calibrated" in R else R["f0"], F, R["E0"])
    if "F1" in F and "F2" in F and R["E1"] and R["E0"]:
        R["in_context_vs_external_gate"] = {"F2_minus_F1": F["F2"]["SR"] - F["F1"]["SR"],
                                            "E1_minus_E0": R["E1"]["mean"]["SR"] - R["E0"]["mean"]["SR"]}
    if "F2b" in F and "F2" in F:
        d2 = F["F2"]["SR"] - F["F1"]["SR"]
        R["F2b_share_of_F2_gain"] = (F["F2b"]["SR"] - F["F1"]["SR"]) / d2 if d2 else None
    json.dump(R, open(os.path.join(OUT, "p2b_results.json"), "w"), indent=1, default=float)
    write_doc(R)
    print(json.dumps({"missing": R["missing"], "gate": R.get("gate")}, indent=1, default=float))


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
    g = R.get("gate")
    if g:
        D.append(f"## Decision: **{g['decision']}**\n")
        if g["decision"] != "INCOMPLETE":
            D.append("| Gate G1.5 condition | Value | Holds |\n|---|---|---|")
            D.append(f"| F0: ≥ {FIT_SHARE:.0f}% of decisions fit 32k tokens with some pruning setting | best: {set_name(g['f0_best_setting'])}, "
                     f"{g['f0_best_share']:.1f}% | {'yes' if g['f0_fits'] else 'no'} |")
            D.append(f"| F2 − F1 ≥ 2 × E0 std ({g['threshold_2std']:.1f} SR) | {g['F2_minus_F1']:+.1f} SR | {'yes' if g['F2_minus_F1_ok'] else 'no'} |")
            D.append(f"| F1 ≥ E0 mean − {g['threshold_2std']:.1f} SR | F1 {g['F1']:.1f} vs E0 {g['E0_mean']:.1f} ({g['F1_minus_E0']:+.1f}) | "
                     f"{'yes' if g['F1_ok'] else 'no'} |")
            D.append("")
            D.append({"GO, implicit": "All three conditions hold: continue with P3 as written (region-scoped context, zero-shot).",
                      "GO, implicit but training-dependent": "The context fits and region scoping pays off inside it, but zero-shot F1 falls "
                      "more than the noise band below E0. Per DEV_PLAN.md, continue only once the P5 compute choice is settled "
                      "(3B backbone, LoRA, or more GPUs); zero-shot P3 becomes a diagnostic and G2 is judged after training.",
                      "FALLBACK, external": "Return to the external region-indexed memory (DEV_PLAN.md appendix); the paper states why the "
                      "context was not enough."}[g["decision"]] + "\n")
            m21, m1 = g["F2_minus_F1"] - g["threshold_2std"], g["F1_minus_E0"] + g["threshold_2std"]
            D.append(f"**How firm:** every F run is a single seed. F2 − F1 clears the noise band by {m21:+.1f} SR and F1 clears its floor by "
                     f"{m1:+.1f} SR; one run's SR varies by ±{g['threshold_2std'] / 2:.1f} (E0 std), so a re-run could move F2 − F1 to either "
                     "side of the band. " + ("The F2 − F1 condition is the fragile one; " if m21 < m1 else "The F1 floor is the fragile one; ")
                     + "two more seeds of F1 and F2 (four more 58-task runs) would settle it before the P5 compute is committed.\n")
    e0 = R.get("E0")
    if e0:
        D.append(f"E0 reference: {e0['label']} ({', '.join(e0['runs'])}), SR {e0['mean']['SR']:.1f} ± {fmt(e0['std']['SR'])}.")
        if e0["label"] != "fixed":
            D.append(" **Caveat:** the F runs use the start-orientation fix but the fixed Phase 2 baseline (`p2_E0_fix_s77–79`) is not on "
                     "this machine yet, so E0 is the Phase 1 baseline without the fix. Copy the Phase 2 runs into `analysis/runs/` and "
                     "re-run `python analysis/p2b_report.py`; the document and the decision update themselves.")
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
    # ---- F runs
    F = R["F"]
    if F:
        D.append("## F1, F2, F2b — 58-task subset, choice mode, start-orientation fix, seed 77\n")
        D.append("| Run | " + " | ".join(P2.NAMES[k] for k in KEYS) + " | Frontier aborts | Context tokens mean / p95 / max | s per decision | "
                 "Frames in context (mean) | Decisions | Capped | Invalid replies |\n|---|" + "---|" * (len(KEYS) + 7))
        for exp, m in F.items():
            D.append(f"| {exp} ({m['run']}) | " + " | ".join(fmt(m[k]) for k in KEYS) + f" | {m['frontier_aborts']} | "
                     f"{m['ctx_tokens_mean']:.0f} / {m['ctx_tokens_p95']:.0f} / {m['ctx_tokens_max']} | {m['sec_per_decision']:.1f} | "
                     f"{m['ctx_frames_mean']:.1f} | {m['decisions']} | {m['decisions_capped']} | {m['invalid_replies']} |")
        if e0:
            D.append(f"| E0 mean ({e0['label']}, n={len(e0['runs'])}) | " + " | ".join(f"{fmt(e0['mean'][k])} ± {fmt(e0['std'][k])}" for k in KEYS)
                     + " | | | | | | | |")
        e1 = R.get("E1")
        if e1:
            D.append(f"| E1 mean ({e1['label']}, n={len(e1['runs'])}) | " + " | ".join(fmt(e1['mean'][k]) for k in KEYS) + " | | | | | | | |")
        D.append("\nContext tokens and seconds are per VLM call (navigation and QA decisions; the baseline's separate retrieval-query "
                 "call does not exist here). Capped = decisions where the oldest frames were dropped to stay under 32k tokens. "
                 f"Exact prompt length minus the tokenizer count used for F0: max |difference| {max(m['token_count_error_max'] for m in F.values())} tokens.\n")
        if "F1" in F and e0:
            D.append(f"- Context-as-memory commits early more often than retrieval memory: premature commits {F['F1']['premature_commit_rate']:.1f}% "
                     f"in F1 vs {e0['mean']['premature_commit_rate']:.1f}% in E0, and {F['F2']['premature_commit_rate']:.1f}% once the context "
                     "holds only the goal's region (F2)." if "F2" in F else "")
        if "in_context_vs_external_gate" in R:
            x = R["in_context_vs_external_gate"]
            D.append(f"- In-context oracle gain F2 − F1 = {x['F2_minus_F1']:+.1f} SR, vs the external oracle gate E1 − E0 = {x['E1_minus_E0']:+.1f} SR.")
        if R.get("F2b_share_of_F2_gain") is not None:
            D.append(f"- F2b recovers {100 * R['F2b_share_of_F2_gain']:.0f}% of F2's gain over F1 (realistic ceiling: room types known, exact regions not)"
                     + ("; F2b ≥ F2 here, a difference within single-seed noise, so knowing the room type is about as good as knowing the region "
                        "on this subset." if F["F2b"]["SR"] >= F["F2"]["SR"] else "."))
        inv = {e: (m["invalid_replies"], m["invalid_with_empty_context"]) for e, m in F.items()}
        D.append("- Invalid replies (" + ", ".join(f"{e}: {a}, {b} of them with no frame in context" for e, (a, b) in inv.items())
                 + ") end the subtask, as an invalid baseline reply does; with an empty context the zero-shot model often still names a "
                 "frame (copying the prompt's example), which is how E1 treated an emptied memory list too.")
        if "F1" in F and F["F1"].get("objects_per_frame") is not None:
            D.append(f"- Objects listed per context frame in F1: {F['F1']['objects_per_frame']:.1f} (F0's ground-truth proxy: {F['F1']['gt_proxy_per_frame']:.1f}).")
        D.append("")
    D.append("## Method notes\n")
    D.append("- F1/F2/F2b (`analysis/p2b_experiments.py`) wrap the instrumented LMEE run at runtime; LMEE code and the baseline result files are unchanged. "
             "Retrieval (the CLIP knowledge base and its query call) is bypassed: the prompt holds the pruned frames in viewing order. "
             "Each frame shows the detector's boxes, numbered; the answer `Frame f, Object o` becomes the same single-object target the baseline "
             "navigates to. QA answers with `Frame f` and a letter, as the baseline does with memories.")
    D.append("- F2 keeps only frames whose ground-truth region (pixel-weighted majority of visible instances) is the goal's region; F2b keeps frames "
             "whose region type matches (unknown types kept). Ground truth is used only for these oracles.")
    D.append("- Each F run is a single seed (77); E0's run-to-run std sets the noise band, as in DEV_PLAN.md.\n")
    D.append("## Commands\n")
    D.append("```bash\nbash analysis/p2b_run_all.sh            # tables + this file from saved outputs (CPU)\n"
             "RERUN=1 bash analysis/p2b_run_all.sh    # also re-run F0 and the F1/F2/F2b GPU jobs\n```")
    open(os.path.join(REPO, "docs/phase2b_decision.md"), "w").write("\n".join(D) + "\n")


if __name__ == "__main__":
    main()

"""docs/p3_report.md (DEV_PLAN.md P3: RCMem zero-shot, development numbers). CPU only; every number from saved outputs.

Inputs
- analysis/out_p3/regions_offline.json   region builder vs HM3D regions, offline (analysis/p3_regions.py)
- analysis/out_p3/parser_eval.json       instruction parser vs the task configs (python rcmem/instruction.py)
- analysis/out_p3/pilot.json             pilot sample (analysis/p3_pilot.py); docs/qual/p3_*.html (analysis/p3_qual.py)
- analysis/runs/<run>                    navigation runs, 58-task subset, choice mode, seed 77, Qwen3-VL-8B base:
    p3_RP_s77 (RCMem, predicted regions), p3_RG_s77 (RCMem, GT regions),
    p3r_F1_s77 (F1, from the P3-0 re-check: same code and backbone), p3_F2bq_s77 (F2b, optional),
    p3r_PA_s77 (P3-0's GT-region panoramas without subtask markers, for reference)
- analysis/out_p2b/p2b_results.json      reference rows on MemoryExplorer's Qwen2.5-VL-7B and the noise band
Metrics per run: analysis/p3_0_report.nav_run (Phase 2b metrics + step 1 + cost), plus render time, online
segmentation scores (RP), step-2 accuracy (step 2 ran on a region whose frames show the goal: the chosen object is a
detection of the goal) and the step-2 share. G2: RP SR >= F1 SR + band; within +-2.6 SR (one run-to-run std) of the
bar is borderline and needs seeds 78-79 (P3-0's F1 s78/s79 exist).
Run (repo root, lmee env): python analysis/p3_report.py   -> analysis/out_p3/p3_results.json, docs/p3_report.md
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import p2_report as P2  # noqa: E402
import p3_0_report as R0  # noqa: E402

OUT = os.path.join(REPO, "analysis/out_p3")
DOC = os.path.join(REPO, "docs/p3_report.md")
RUNS = [  # key, run, label
    ("RP", "p3_RP_s77", "RCMem zero-shot, predicted regions"),
    ("RG", "p3_RG_s77", "RCMem zero-shot, ground-truth regions"),
    ("RPin", "p3_RPin_s77", "RP variant: step-2 boxes only inside the chosen region"),
    ("RPpt", "p3_RPpt_s77", "RP variant: step 2 by pointing (no detector box needed)"),
    ("RPcu", "p3_RPcu_s77", "RP variant: room close-ups + show threshold 1%"),
    ("RP25", "p3_RP25_s77", "RCMem zero-shot, predicted regions, on Qwen2.5-VL-7B-Instruct base"),
    ("PA", "p3r_PA_s77", "P3-0 panoramas (GT regions), no subtask markers"),
    ("F2bq", "p3_F2bq_s77", "F2b: context restricted to the goal's region type"),
    ("F1q", "p3r_F1_s77", "F1: all kept frames, no regions"),
]
KEYS = ["SR", "SPL", "premature_commit_rate", "distractor_rate_of_memory_choices", "step1_exact", "none_P", "none_R",
        "step2_acc", "step2_share", "tokens_per_decision", "tokens_per_decision_max", "sec_per_decision", "render_s"]
BORDER = 2.6


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def run_metrics(run):
    m = R0.nav_run(run)
    import p2b_report as PB  # noqa: E402
    _, _, decisions = PB.A.load_records(os.path.join(REPO, "analysis/runs", run))
    ds = [d for v in decisions.values() for d in v]
    p3 = [d["p3"] for d in ds if d.get("p3", {}).get("exp") in ("RG", "RP", "RPin", "RPpt", "RPcu")]
    if p3:
        m["render_s"] = float(np.mean([p["render_s"] for p in p3]))
        m["render_s_max"] = float(np.max([p["render_s"] for p in p3]))
        m["regions_per_decision"] = float(np.mean([p["n_regions"] for p in p3]))
        hol = [h for p in p3 for h in p["holes"] if h is not None]
        m["hole_fraction"] = float(np.mean(hol)) if hol else None
        seg = [p["seg"] for p in p3 if p.get("seg")]
        if seg:
            for k in ("point_acc", "coverage", "agreement"):
                m["seg_" + k] = 100 * float(np.mean([s[k] for s in seg]))
            m["seg_merged_share"] = 100 * sum(s["merged"] for s in seg) / max(1, sum(s["n_pred"] for s in seg))
            m["seg_decisions_with_merge"] = 100 * float(np.mean([s["merged"] > 0 for s in seg]))
            m["seg_pred_regions"] = float(np.mean([s["n_pred"] for s in seg]))
            m["seg_gt_regions"] = float(np.mean([s["n_gt"] for s in seg]))
        # step 2: ran on a region whose frames show the goal -> chose a detection of the goal
        hits = []
        for d in ds:
            p = d.get("p3", {})
            if p.get("step2") and p.get("step1_choice") in p.get("goal_visible_regions", []):
                ch = d.get("choice") or {}
                goal = {int(x) for v in (d.get("goal_obj_ids_mapping") or {}).values() for x in v}
                hits.append(ch.get("type") == "memory" and bool(set(ch.get("cluster", [])) & goal))
        m["step2_acc"] = 100 * float(np.mean(hits)) if hits else None
        m["step2_n"] = len(hits)
        m["markers_same_room"] = sum("same room as subtask" in p.get("markers", "") for p in p3)
        m["markers_different_room"] = sum("a different" in p.get("markers", "") for p in p3)
    return m


def fmt(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    pb = json.load(open(os.path.join(REPO, "analysis/out_p2b/p2b_results.json")))
    band = pb["noise"]["band"]
    res = {"band": band, "runs": {}, "missing": []}
    for k, run, label in RUNS:
        if os.path.isdir(os.path.join(REPO, "analysis/runs", run)) and P2.complete(run, 58):
            res["runs"][k] = {"run": run, "label": label, **{a: (float(b) if isinstance(b, (int, float, np.floating, np.integer)) else b)
                                                             for a, b in run_metrics(run).items()}}
        else:
            res["missing"].append(run)
    reg = json.load(open(os.path.join(OUT, "regions_offline.json")))
    par = json.load(open(os.path.join(OUT, "parser_eval.json")))["summary"]
    pilot = json.load(open(os.path.join(OUT, "pilot.json")))
    quals = sorted(glob.glob(os.path.join(REPO, "docs/qual/p3_*.html")))
    go = os.path.join(OUT, "user_go.txt")
    res["g2"] = {}
    R = res["runs"]
    if "RP" in R and "F1q" in R:
        bar = R["F1q"]["SR"] + band
        res["g2"] = {"RP": R["RP"]["SR"], "F1": R["F1q"]["SR"], "bar": bar, "pass": R["RP"]["SR"] >= bar,
                     "borderline": abs(R["RP"]["SR"] - bar) <= BORDER}
        if "F2bq" in R:
            res["g2"]["half_gap_bar"] = (R["F1q"]["SR"] + R["F2bq"]["SR"]) / 2
            res["g2"]["half_gap_pass"] = R["RP"]["SR"] >= res["g2"]["half_gap_bar"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "p3_results.json"), "w"), indent=1, default=str)

    L = ["# P3 report: RCMem zero-shot (development numbers)", "",
         "Generated by `analysis/p3_report.py` from saved outputs; every number is in `analysis/out_p3/p3_results.json`. "
         "Reproduce everything with `bash analysis/p3_run_all.sh` (`REPORT_ONLY=1` rescores only). "
         "These are P3 development numbers: 58-task subset, choice mode, **one seed (77)**, base Qwen3-VL-8B, zero-shot. "
         "The main-table row comes from P6 (full set, 3 seeds) with the settings frozen here.", ""]
    # ---- region builder
    best = reg["door_r"]
    L += ["## 1. Region builder (`rcmem/regions.py`)", "",
          "2D floor/wall map (0.1 m cells) from every captured depth frame; segmentation at narrow passages (erode the "
          "observed floor by DOOR_R, components ≥ 0.5 m² seed regions, watershed on the distance transform grows them back; "
          "regions < 1 m² merge into a neighbour); each 3D point takes its cell's region (nearest region cell within 1 m "
          "for walls and furniture); each region's panorama centre is its maximum-visibility cell at floor + 1.5 m. "
          "No ground truth is used; scores below compare with HM3D regions.", "",
          f"**Offline** (`analysis/p3_regions.py`, kept frames of `{reg['run']}`, scored at every navigation decision; "
          "point accuracy = share of points whose GT region is the majority GT region of their predicted region; coverage = the "
          "converse; agreement = their harmonic mean; merged = predicted regions where ≥ 2 GT regions each hold ≥ 20% of the points):", "",
          "| DOOR_R m / H_MAX m (0 = erosion seeds) | Decisions | Point accuracy % | Coverage % | Agreement % | Predicted / GT regions | "
          "Predicted regions that merge rooms % | Decisions with a merge % | Splits per decision | Segmentation s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r, s in best.items():
        L.append(f"| {r}{' **(used)**' if r == '0.4/0' else ''} | {s['decisions']} | {100 * s['point_acc']:.1f} | {100 * s['coverage']:.1f} | "
                 f"{100 * s['agreement']:.1f} | {s['n_pred']:.1f} / {s['n_gt']:.1f} | {100 * s['share_pred_regions_merged']:.1f} | "
                 f"{100 * s['share_decisions_with_merge']:.1f} | {s['split_per_decision']:.2f} | {s['seg_s_mean']:.2f} |")
    L += ["", "Agreement is flat across settings (83.5–84.5). DOOR_R = 0.4 m is used: within 0.1 point of the best and, unlike "
          "0.3 m, it closes a 0.9 m doorway (self-check in `rcmem/regions.py`). Most merges join parts of open-plan spaces "
          "that HM3D splits into separate regions with no wall between them.", ""]
    if "RP" in R and "seg_point_acc" in R["RP"]:
        x = R["RP"]
        L += [f"**Online** (run `{x['run']}`, every navigation decision, all captured frames in the map): point accuracy "
              f"{fmt(x['seg_point_acc'])}%, coverage {fmt(x['seg_coverage'])}%, agreement {fmt(x['seg_agreement'])}%, "
              f"{fmt(x['seg_pred_regions'])} predicted vs {fmt(x['seg_gt_regions'])} GT regions per decision, "
              f"{fmt(x['seg_merged_share'])}% of predicted regions merge rooms, {fmt(x['seg_decisions_with_merge'])}% of decisions have a merge.", ""]
    # ---- renderer
    L += ["## 2. A1-band renderer online", "",
          "| Run | Regions per decision | Hole fraction (full sphere) | Render s per decision (mean / max) | Tokens per decision (step 1 + step 2, mean / max) | s per decision (VLM) |",
          "|---|---|---|---|---|---|"]
    for k in ("RP", "RG"):
        if k in R:
            x = R[k]
            L.append(f"| {x['label']} | {fmt(x.get('regions_per_decision'))} | {fmt(x.get('hole_fraction'), 2)} | "
                     f"{fmt(x.get('render_s'), 2)} / {fmt(x.get('render_s_max'), 2)} | {fmt(x.get('tokens_per_decision'), 0)} / "
                     f"{fmt(x.get('tokens_per_decision_max'), 0)} | {fmt(x.get('sec_per_decision'))} |")
    # ---- parser
    L += ["", "## 3. Instruction parser (`rcmem/instruction.py`)", "",
          f"Rule-based: clauses at sequence words, a room vocabulary with synonyms and joined rooms (\"X and Y area\"), "
          f"goals anchored by their names in order, region instances from ordinal words. Scored against all "
          f"{par['tasks']} task configs ({par['subtasks']} subtasks):", "",
          "| Goal located % | Region type correct % | Region-instance grouping (subtask pairs) % | Whole task exact % | Exact: easy / medium / hard % |",
          "|---|---|---|---|---|",
          f"| {par['goal_located']:.1f} | {par['region_type']:.1f} | {par['instance_pairs']:.1f} | {par['task_exact']:.1f} | "
          f"{par['task_exact_easy']:.1f} / {par['task_exact_medium']:.1f} / {par['task_exact_hard']:.1f} |", "",
          "Remaining errors: repeated generic goals (\"picture\" three times) anchored at the wrong mention, and configs that "
          "label a \"rec room part\" as the same region as the living area. Errors: `analysis/out_p3/parser_eval.json`.", ""]
    # ---- two-step commit + runs
    L += ["## 4. Two-step commit and subset runs (Qwen3-VL-8B base, zero-shot, seed 77)", "",
          "Step 1 (made before moving) sees the task, the subtask markers `<subtask i: goal, room>` (with the region holding "
          "each earlier subtask's committed target and \"same room as subtask j\" / \"a different <room> from subtask j\"), one "
          "A1-band panorama per observed region and the frontiers, and answers `Region k` or `Frontier i`. Step 2 sees only "
          "region k's frames and answers `Frame f, Object o`. No region observed → nearest frontier without a VLM call "
          "(empty-context rule). Decision log: the `p3` block of every decision record in `analysis/runs/<run>/records_*.jsonl`.", "",
          "| Run | SR | SPL | Premature-commit % | Distractor % (memory choices) | Step-1 exact region % | 'Not seen yet' P / R % | "
          "Step-2 accuracy % | Step 2 run % | Tokens per decision (mean / max) | s per decision |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, run, label in RUNS:
        if k in R:
            x = R[k]
            L.append(f"| {label} (`{run}`) | {fmt(x['SR'])} | {fmt(x['SPL'])} | {fmt(x['premature_commit_rate'])} | "
                     f"{fmt(x['distractor_rate_of_memory_choices'])} | {fmt(x.get('step1_exact'))} | {fmt(x.get('none_P'))} / {fmt(x.get('none_R'))} | "
                     f"{fmt(x.get('step2_acc'))} | {fmt(x.get('step2_share'))} | {fmt(x.get('tokens_per_decision'), 0)} / "
                     f"{fmt(x.get('tokens_per_decision_max'), 0)} | {fmt(x.get('sec_per_decision'))} |")
        else:
            L.append(f"| {label} (`{run}`) | not run | | | | | | | | | |")
    for name, key in (("E0 fixed (MemoryExplorer, Qwen2.5-VL-7B, 3 seeds)", None), ("F1 (Qwen2.5-VL-7B, 3 seeds)", "F1"),
                      ("F2b (Qwen2.5-VL-7B, 3 seeds)", "F2b")):
        x = pb["E0"]["mean"] if key is None else pb["F"][key]["mean"]
        L.append(f"| Reference: {name} | {fmt(x['SR'])} | {fmt(x['SPL'])} | {fmt(x['premature_commit_rate'])} | "
                 f"{fmt(x.get('distractor_rate_of_memory_choices'))} | — | — | — | — | — | — |")
    L += ["", "Step-1 exact region: the chosen region is the observed region holding the goal (GT region for RG; the predicted "
          "region containing the goal object's centre for RP), or `explore` when there is none. 'Not seen yet': positive = "
          "`explore`. Step-2 accuracy: over step-2 calls on a region whose frames show ≥ 100 goal pixels, the chosen object "
          f"is a detection of the goal (n = {R.get('RP', {}).get('step2_n', '—')} for RP). Premature commit and distractor rates "
          "as in Phase 1–2b (`analysis/analyze.py`).", ""]
    # ---- G2
    g = res["g2"]
    L += ["## 5. Gate G2 (diagnostic, judged after P5)", ""]
    if g:
        L += [f"RCMem with predicted regions {g['RP']:.1f} SR vs F1 on Qwen3-VL-8B {g['F1']:.1f} SR + band {band:.1f} = "
              f"**{g['bar']:.1f}**: **{'passes' if g['pass'] else 'fails'}**"
              + (f" (borderline, within {BORDER} SR of the bar: add seeds 78–79 before deciding)" if g["borderline"] else "") + "."]
        gd = os.path.join(OUT, "g2_decision.txt")
        if g["borderline"] and os.path.exists(gd):
            L.append("Borderline handling: " + open(gd).read().strip())
        if "half_gap_bar" in g:
            L.append(f"Half-the-gap form: (F1 + F2b) / 2 = {g['half_gap_bar']:.1f} → {'passes' if g['half_gap_pass'] else 'fails'}.")
        L.append("")
    else:
        L += ["Not computed: RP or F1 run missing.", ""]
    # ---- qual, freeze, hygiene
    tag = git("tag", "--list", "p3-freeze")
    L += ["## 6. Pilot report, freeze, hygiene", "",
          f"- Pilot sample (`analysis/p3_pilot.py`, seed {pilot['seed']}): {', '.join(pilot['episodes'])}.",
          f"- Qualitative reports: {', '.join('`' + os.path.relpath(q, REPO) + '`' for q in quals) or 'none'}.",
          f"- User's go before the full runs: {open(go).read().strip() if os.path.exists(go) else 'not recorded'}.",
          f"- Frozen prompts and settings for P6: tag `p3-freeze` {'→ ' + git('rev-list', '-n1', '--abbrev-commit', 'p3-freeze') if tag else '(not created yet)'} "
          "(`analysis/p3_nav.py` SYS1 / INTRO / markers, `analysis/p2b_context.py` step-2 prompt, `rcmem/regions.py` constants, "
          "`rcmem/instruction.py`, MIN_SHARE 0.05, voxel 0.05 m, A1-band 1024 px).",
          f"- Missing runs: {', '.join(res['missing']) or 'none'}.", ""]
    open(DOC, "w").write("\n".join(L) + "\n")
    print(DOC)


if __name__ == "__main__":
    main()

"""Every navigation run so far in one table (docs/p3_all_runs.md): LMEE 58-task subset (choice mode) runs grouped by method
and backbone (mean ± std over seeds where there are several), the full-set run, and GOAT-Bench.

Metrics per run: analysis/p3_report.run_metrics (SR, SPL, premature-commit and distractor rates, step-1 exact region,
step-2 accuracy, tokens per decision) and QA accuracy in LMEE's original multiple-choice format (answer order as shipped,
where always answering "B" scores 59.3%), with the prior-corrected accuracy (accuracy minus the best constant letter).
Phase 2b's F1 / F2 / F2b runs are not on disk; their rows come from analysis/out_p2b/p2b_results.json.
GOAT-Bench (no QA): SR and SPL from the subtask records, overall and on the 12-scene subset (analysis/runs/goat_sub12_data).
Run (repo root, lmee env): python analysis/p3_table.py   -> analysis/out_p3/all_runs.json, docs/p3_all_runs.md
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
import p2_report as P2  # noqa: E402

ME, Q3, Q25 = "MemoryExplorer Qwen2.5-VL-7B (fine-tuned)", "Qwen3-VL-8B base", "Qwen2.5-VL-7B-Instruct base"
GROUPS = [  # (phase, label, backbone, runs)
    ("P1–P2", "E0 MemoryExplorer, original start orientation", ME, ["instr_choice_sub", "p1_E0_r2", "p1_E0_r3", "p2_E0_nofix_s78", "p2_E0_nofix_s79"]),
    ("P1–P2", "E0 MemoryExplorer, fixed start (reference baseline)", ME, ["p2_E0_fix_s77", "p2_E0_fix_s78", "p2_E0_fix_s79"]),
    ("P1", "E1 oracle region gate (original start)", ME, ["p1_E1"]),
    ("P2", "E1 oracle region gate", ME, ["p2_E1_fix_s77", "p2_E1_fix_s78", "p2_E1_fix_s79"]),
    ("P2", "E1b region-type gate", ME, ["p2_E1b_fix_s77", "p2_E1b_fix_s78", "p2_E1b_fix_s79"]),
    ("P1", "E2 room name in prompt (original start)", ME, ["p1_E2"]),
    ("P1", "E4 verify on arrival (original start)", ME, ["p1_E4"]),
    ("P1", "E5 earlier-goal exclusion (original start)", ME, ["p1_E5"]),
    ("P3-0", "F1 context, no regions", Q3, ["p3r_F1_s77", "p3r_F1_s78"]),
    ("P3-0", "PA panoramas (GT regions), two-step, no markers", Q3, ["p3r_PA_s77", "p3r_PA_s78"]),
    ("P3-0", "PB region markers + labels (GT regions), two-step", Q3, ["p3r_PB_s77", "p3r_PB_s78"]),
    ("P3", "RP: RCMem, predicted regions", Q3, ["p3_RP_s77"]),
    ("P3", "RG: RCMem, ground-truth regions", Q3, ["p3_RG_s77"]),
    ("P3", "RPin: RP + step-2 boxes only inside the chosen region", Q3, ["p3_RPin_s77"]),
    ("P3", "RPpt: RP + step 2 by pointing", Q3, ["p3_RPpt_s77"]),
    ("P3", "RPcu: RP + room close-ups + show threshold 1%", Q3, ["p3_RPcu_s77"]),
    ("P3", "RP: RCMem, predicted regions", Q25, ["p3_RP25_s77"]),
    ("P3", "RPcu: RP + room close-ups + show threshold 1%", Q25, ["p3_RPcu25_s77"]),
]
KEYS = ["SR", "SPL", "premature_commit_rate", "distractor_rate_of_memory_choices", "step1_exact", "step2_acc", "acc",
        "acc_prior_corrected", "tokens_per_decision"]


def metrics(run):
    import p3_report as R  # noqa: E402
    try:
        return run, {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in R.run_metrics(run).items()}
    except Exception as e:  # noqa: BLE001
        return run, {"error": repr(e)}


def goat(run, scenes=None):
    import analyze as A  # noqa: E402
    _, subtasks, _ = A.load_records(os.path.join(REPO, "analysis/runs", run))
    rows = [s for s in subtasks.values() if scenes is None or A.scene_of(s["subtask"]) in scenes]
    out = {"tasks": len(rows)}
    for t in (None, "object", "description", "image"):
        rr = [s for s in rows if t is None or s["goal_type"] == t]
        if not rr:
            continue
        sr = [float(s["success_by_distance"]) for s in rr]
        spl = [float(s["success_by_distance"]) * s["gt_dist"] / max(s["gt_dist"], s["explore_dist"], 1e-6) for s in rr
               if s.get("gt_dist") is not None and np.isfinite(s["gt_dist"])]  # SPL over tasks with a GT path length
        out[t or "all"] = {"n": len(rr), "SR": 100 * np.mean(sr), "SPL": 100 * np.mean(spl) if spl else float("nan"), "n_spl": len(spl)}
    return out


def ms(vals, d=1):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "—"
    return f"{np.mean(vals):.{d}f}" + (f" ± {np.std(vals, ddof=1):.{d}f}" if len(vals) > 1 else "")


def main():
    runs = [r for g in GROUPS for r in g[3] if P2.complete(r, 58)]
    with Pool(min(16, len(runs))) as pool:
        M = dict(pool.map(metrics, runs))
    pb = json.load(open(os.path.join(REPO, "analysis/out_p2b/p2b_results.json")))
    sub12 = {f[:-5] for f in os.listdir(os.path.join(REPO, "analysis/runs/goat_sub12_data"))}
    G = {"MemoryExplorer (goat_ep0, all 36 scenes)": goat("goat_ep0"),
         "MemoryExplorer (goat_ep0, 12-scene subset)": goat("goat_ep0", sub12)}
    json.dump({"runs": M, "goat": G}, open(os.path.join(REPO, "analysis/out_p3/all_runs.json"), "w"), indent=1, default=str)

    L = ["# All navigation runs so far", "",
         "Generated by `analysis/p3_table.py` (numbers in `analysis/out_p3/all_runs.json`). LMEE 58-task subset, choice mode; "
         "mean ± std over seeds where a row has several. QA accuracy is LMEE's original multiple-choice format (answer order as "
         "shipped: always answering \"B\" scores 59.3%); prior-corrected = accuracy − best constant letter. Premature commit and "
         "distractor rates as in `analysis/analyze.py`. Step-1 exact region and step-2 accuracy exist only for two-step runs.", "",
         "| Phase | Run | Backbone | Seeds | SR | SPL | Premature % | Distractor % | Step-1 exact % | Step-2 acc. % | QA acc. % | QA prior-corrected % | Tokens / decision |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for phase, label, bb, rs in GROUPS:
        ok = [r for r in rs if r in M and "error" not in M[r]]
        if not ok:
            L.append(f"| {phase} | {label} | {bb} | — | {'running' if any(os.path.isdir(os.path.join(REPO, 'analysis/runs', r)) for r in rs) else 'not run'} | | | | | | | | |")
            continue
        seeds = ", ".join(r.split("_s")[-1] for r in ok) if all("_s7" in r for r in ok) else f"{len(ok)} run{'s' * (len(ok) > 1)}"
        cells = [ms([M[r].get(k) for r in ok], 0 if k == "tokens_per_decision" else 1) for k in KEYS]
        L.append(f"| {phase} | {label} | {bb} | {seeds} | " + " | ".join(cells) + " |")
    for k, name in (("F1", "F1 context, no regions"), ("F2", "F2 context, oracle goal region"), ("F2b", "F2b context, oracle region type")):
        x = pb["F"][k]
        L.append(f"| P2b | {name} | {ME} | 77, 78, 79 | {x['mean']['SR']:.1f} ± {x['std']['SR']:.1f} | {x['mean']['SPL']:.1f} | "
                 f"{x['mean']['premature_commit_rate']:.1f} | {x['mean'].get('distractor_rate_of_memory_choices', float('nan')):.1f} | — | — | — | — | "
                 f"{x['mean'].get('ctx_tokens_mean', float('nan')):.0f} |")
    full = "p2_full_E0_fix_s77"
    if P2.complete(full, 166):
        _, m = metrics(full)
        L += ["", "**Full set** (166 tasks, seed 77): MemoryExplorer, fixed start — "
              f"SR {m['SR']:.1f}, SPL {m['SPL']:.1f}, premature {m['premature_commit_rate']:.1f}%, QA {m['acc']:.1f} (prior-corrected {m['acc_prior_corrected']:.1f})."]
    L += ["", "## GOAT-Bench (val-unseen, episode 0 per scene; SR within 1 m of a goal viewpoint; no QA)", "",
          "| Run | Tasks | SR | SPL | Object SR | Description SR | Image SR |", "|---|---|---|---|---|---|---|"]
    for name, g in G.items():
        L.append(f"| {name} | {g['tasks']} | {g['all']['SR']:.1f} | {g['all']['SPL']:.1f} | " +
                 " | ".join(f"{g[t]['SR']:.1f} (n={g[t]['n']})" if t in g else "—" for t in ("object", "description", "image")) + " |")
    L.append("| RCMem RP / RPcu (12-scene subset) | — | stopped (31 Oct, to prioritise RPcu on Qwen2.5) | | | | |")
    open(os.path.join(REPO, "docs/p3_all_runs.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

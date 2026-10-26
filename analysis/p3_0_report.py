"""P3-0a scoring and docs/p3_0_decision.md (DEV_PLAN.md P3-0). CPU only, seconds.

Reads analysis/out_p3_0/<run>__<model>.jsonl (analysis/p3_0a.py) and scores every region-unit option against ground
truth, per decision:
- target: the goal's GT region if it is among the observed regions, else "not seen yet" (none);
- exact-region accuracy: the answer is the target ("invalid" replies count as wrong);
- region-type accuracy (decisions whose goal region has a known type): a region whose GT type matches the goal's,
  or "not seen yet" when no observed region has that type;
- "not seen yet" precision / recall (positive = none);
- goal presence (options that say it): over decisions where a region was chosen, "visible" against
  >= GOAL_PX goal pixels in that region's panorama;
- random same-type: uniform choice among observed regions whose type matches the goal's (unknown types kept, as
  in E1b); "not seen yet" when there is none; scored in expectation.
Each metric is computed per seed and reported as mean ± std over the 3 seeds (ddof=1), for all navigation
decisions and split by goal type. Writes analysis/out_p3_0/p3_0_results.json and docs/p3_0_decision.md.
Run (repo root, any env with numpy): python analysis/p3_0_report.py
"""
import collections
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis/out_p3_0")
DOC = os.path.join(REPO, "docs/p3_0_decision.md")
RUNS = ["p2_E0_fix_s77", "p2_E0_fix_s78", "p2_E0_fix_s79"]
GOAL_PX = 100
OPTIONS = [  # (key, model file, label)
    ("A1", "qwen25", "A1: RGB panoramas, Qwen2.5-VL-7B (MemoryExplorer), all panoramas in one prompt"),
    ("A2", "onecanvas", "A2: OneCanvas feature canvases, OneCanvas-Qwen3-VL-8B, per-region yes/no"),
    ("C_multi", "qwen3", "Control: RGB panoramas, Qwen3-VL-8B base, all panoramas in one prompt"),
    ("C_pr", "qwen3", "Control: RGB panoramas, Qwen3-VL-8B base, per-region yes/no"),
    ("B", "qwen25", "B: region markers with VLM labels + 4 views per region, Qwen2.5-VL-7B (MemoryExplorer)"),
    ("C_B", "qwen3_B", "Control: option B on Qwen3-VL-8B base (labels + 4 views per region)"),
    ("rand", "qwen25", "Random same-type region (GT types)"),
]
METRICS = ["exact", "type", "none_P", "none_R", "vis_P", "vis_R", "invalid"]


def load(run, model):
    f = os.path.join(OUT, f"{run}__{model}.jsonl")
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def target(r):
    return next((i for i, x in enumerate(r["regions"]) if x in r["goal_regions"]), "none")


def type_ok(r, pred):
    gt = set(r["goal_types"])
    if pred == "none":
        return not any(set(t) & gt for t in r["region_types"])
    return isinstance(pred, int) and bool(set(r["region_types"][pred]) & gt)


def rand_cands(r):
    gt = set(r["goal_types"])
    return [i for i, t in enumerate(r["region_types"]) if not gt or not t or set(t) & gt]


def score(rows, key):
    """rows: decision rows of one run. Returns metric -> value (%), with counts."""
    ex, ty, n_ty, inv = [], [], 0, 0
    tp_n = fp_n = fn_n = 0.0
    vtp = vfp = vfn = 0
    for r in rows:
        t = target(r)
        if key == "rand":
            c = rand_cands(r)
            p_none = 1.0 if not c else 0.0
            ex.append(p_none if t == "none" else (1 / len(c) if t in c else 0.0))
            if r["goal_types"]:
                ty.append(1.0 if not c else np.mean([type_ok(r, i) for i in c]))
            tp_n += p_none * (t == "none")
            fp_n += p_none * (t != "none")
            fn_n += (1 - p_none) * (t == "none")
            continue
        p = r.get(f"{key}_choice")
        inv += p == "invalid"
        ex.append(float(p == t))
        if r["goal_types"]:
            ty.append(float(type_ok(r, p)))
        tp_n += p == "none" and t == "none"
        fp_n += p == "none" and t != "none"
        fn_n += p != "none" and t == "none"
        v = r.get(f"{key}_visible")
        if isinstance(p, int) and v is not None:
            g = r["goal_px"][p] >= GOAL_PX
            vtp, vfp, vfn = vtp + (v and g), vfp + (v and not g), vfn + ((not v) and g)
    pct = lambda a, b: 100.0 * a / b if b else float("nan")  # noqa: E731
    return {"exact": 100 * np.mean(ex), "type": 100 * np.mean(ty) if ty else float("nan"),
            "none_P": pct(tp_n, tp_n + fp_n), "none_R": pct(tp_n, tp_n + fn_n),
            "vis_P": pct(vtp, vtp + vfp), "vis_R": pct(vtp, vtp + vfn), "invalid": pct(inv, len(rows)), "n": len(rows)}


def ms(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def fmt(m, s):
    return "—" if np.isnan(m) else (f"{m:.1f}" if np.isnan(s) else f"{m:.1f} ± {s:.1f}")


def main():
    data = {(run, mdl): load(run, mdl) for run in RUNS for mdl in ("qwen25", "qwen3", "qwen3_B", "onecanvas")}
    R = {"subsets": {}, "stats": {}, "coverage": {}}
    for (run, mdl), rows in data.items():
        R["coverage"][f"{run}__{mdl}"] = len(rows)
    for sub, keep in [("all", lambda r: True), ("description", lambda r: r["goal_type"] == "description"),
                      ("image", lambda r: r["goal_type"] == "image"),
                      ("goal region observed", lambda r: target(r) != "none"),
                      ("goal region not observed", lambda r: target(r) == "none")]:
        S = {}
        for key, mdl, _ in OPTIONS:
            per = {run: score([r for r in data[(run, mdl)] if keep(r)], key) for run in RUNS if data[(run, mdl)]}
            if not per:
                continue
            S[key] = {"runs": per, **{m: ms([per[x][m] for x in per]) for m in METRICS},
                      "n": int(sum(per[x]["n"] for x in per))}
        R["subsets"][sub] = S
    # tokens, holes, render time
    st = {}
    q = [r for run in RUNS for r in data[(run, "qwen25")]]
    oc = [r for run in RUNS for r in data[(run, "onecanvas")]]
    if q:
        nz = [r for r in q if r["regions"]]
        st["A1 panorama image tokens per decision (Qwen2.5-VL)"] = [np.mean([r["A1_image_tokens"] for r in nz]), max(r["A1_image_tokens"] for r in nz)]
        st["B image tokens per decision (Qwen2.5-VL)"] = [np.mean([r["B_image_tokens"] for r in nz]), max(r["B_image_tokens"] for r in nz)]
        st["Regions per decision"] = [np.mean([len(r["regions"]) for r in q]), max(len(r["regions"]) for r in q)]
        st["Hole fraction per panorama (full sphere)"] = [np.mean([h for r in q for h in r["hole_fraction"]]), max(h for r in q for h in r["hole_fraction"])]
        st["Panorama update + render time per decision (s)"] = [np.mean([r["render_s"] for r in q]), max(r["render_s"] for r in q)]
        st["Kept views per decision (frame-level pruning, 0.05 m)"] = [np.mean([r["kept_frames"] for r in q]), max(r["kept_frames"] for r in q)]
        st["Seen views per decision"] = [np.mean([r["seen_frames"] for r in q]), max(r["seen_frames"] for r in q)]
    if oc:
        nz = [r for r in oc if r["regions"]]
        st["A2 canvas tokens per decision (sum over regions)"] = [np.mean([sum(r["A2_tokens"]) for r in nz]), max(sum(r["A2_tokens"]) for r in nz)]
        st["A2 canvas tokens per region"] = [np.mean([t for r in nz for t in r["A2_tokens"]]), max(t for r in nz for t in r["A2_tokens"])]
    R["stats"] = {k: [float(v[0]), float(v[1])] for k, v in st.items()}
    # P3-0a rule
    A = R["subsets"]["all"]
    D = R["subsets"]["description"]
    g = lambda S, k: S.get(k, {}).get("exact", (float("nan"),))[0]  # noqa: E731
    best = max(["A1", "A2"], key=lambda k: -1 if np.isnan(g(A, k)) else g(A, k))
    rule = {"A1_exact": g(A, "A1"), "A2_exact": g(A, "A2"), "B_exact": g(A, "B"), "rand_exact": g(A, "rand"),
            "A2_exact_description": g(D, "A2"), "A1_exact_description": g(D, "A1"), "B_exact_description": g(D, "B")}
    rule["best_A"] = best
    rule["margin_over_B"] = g(A, best) - g(A, "B")
    rule["beats_B_by_10"] = bool(rule["margin_over_B"] >= 10)
    rule["beats_random"] = bool(g(A, best) > g(A, "rand"))
    rule["pass"] = rule["beats_B_by_10"] and rule["beats_random"]
    rule["A1_A2_within_5"] = bool(abs(g(A, "A1") - g(A, "A2")) <= 5)
    R["rule"] = rule
    json.dump(R, open(os.path.join(OUT, "p3_0_results.json"), "w"), indent=1, default=float)
    write_doc(R)
    print(json.dumps(rule, indent=1, default=float))


def table(S, keys):
    L = ["| Option | Decisions | Exact region % | Region type % | 'Not seen yet' P / R % | Goal visible P / R % | Invalid % |",
         "|---|---|---|---|---|---|---|"]
    lab = {k: l for k, _, l in OPTIONS}
    for k in keys:
        if k not in S:
            continue
        x = S[k]
        L.append(f"| {lab[k]} | {x['n']} | {fmt(*x['exact'])} | {fmt(*x['type'])} | {fmt(*x['none_P'])} / {fmt(*x['none_R'])} | "
                 f"{fmt(*x['vis_P'])} / {fmt(*x['vis_R'])} | {fmt(*x['invalid'])} |")
    return "\n".join(L)


def write_doc(R):
    keys = [k for k, _, _ in OPTIONS]
    A, rule = R["subsets"]["all"], R["rule"]
    per_seed = ["| Option | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
    for k in keys:
        if k in A:
            per_seed.append(f"| {k} | " + " | ".join(f"{A[k]['runs'][r]['exact']:.1f}" if r in A[k]["runs"] else "—" for r in RUNS) + " |")
    stats = ["| Quantity | Mean | Max |", "|---|---|---|"] + [f"| {k} | {v[0]:.2f} | {v[1]:.2f} |" for k, v in R["stats"].items()]
    md = open(os.path.join(OUT, "decision_text.md")).read() if os.path.exists(os.path.join(OUT, "decision_text.md")) else ""
    doc = f"""# P3-0 decision: which region unit (DEV_PLAN.md P3-0)

Tables generated by `analysis/p3_0_report.py` from `analysis/out_p3_0/*.jsonl`; every number is in
`analysis/out_p3_0/p3_0_results.json`. Reproduce: `bash analysis/p3_0_run_all.sh` (all stages, about 3 h on 4 × RTX 3090)
or `REPORT_ONLY=1 bash analysis/p3_0_run_all.sh` (rescoring only). Rationale text: `analysis/out_p3_0/decision_text.md`.

{md}

## P3-0a scores (offline, 58-task subset, fixed-baseline decisions, seeds 77–79)

Exact-region accuracy per seed:

{chr(10).join(per_seed)}

### All navigation decisions

{table(A, keys)}

### Description subtasks (A2 sees the same subtask information as the others)

{table(R['subsets']['description'], keys)}

### Image subtasks (A2 gets only the text; the others also see the target image)

{table(R['subsets']['image'], keys)}

### Goal region already observed

{table(R['subsets']['goal region observed'], keys)}

### Goal region not observed yet (correct answer: "not seen yet")

{table(R['subsets']['goal region not observed'], keys)}

### Tokens, holes and render time (per decision unless stated)

{chr(10).join(stats)}

### P3-0a rule (provisional thresholds)

| Check | Value | Result |
|---|---|---|
| Best panorama option on exact-region accuracy | {rule['best_A']}: {rule['A1_exact' if rule['best_A'] == 'A1' else 'A2_exact']:.1f} | — |
| Beats option B by ≥ 10 points | {rule['margin_over_B']:+.1f} vs B {rule['B_exact']:.1f} | {'yes' if rule['beats_B_by_10'] else 'no'} |
| Beats random same-type | vs {rule['rand_exact']:.1f} | {'yes' if rule['beats_random'] else 'no'} |
| A1 and A2 within 5 points (prefer A1) | A1 {rule['A1_exact']:.1f}, A2 {rule['A2_exact']:.1f} | {'yes' if rule['A1_A2_within_5'] else 'no'} |
| **P3-0a passes** | | **{'yes' if rule['pass'] else 'no'}** |
"""
    open(DOC, "w").write(doc)


if __name__ == "__main__":
    main()

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
    # P3-0 re-check: base Qwen2.5-VL-7B-Instruct, and A1-band (panoramas cropped to the seen latitude band)
    ("A1_base", "qwen25base", "Re-check: RGB panoramas, Qwen2.5-VL-7B-Instruct base, all panoramas in one prompt"),
    ("band_base", "qwen25base", "Re-check: A1-band panoramas, Qwen2.5-VL-7B-Instruct base, all panoramas in one prompt"),
    ("B_base", "qwen25base", "Re-check: option B on Qwen2.5-VL-7B-Instruct base (labels + 4 views per region)"),
    ("C_band", "qwen3_band", "Re-check: A1-band panoramas, Qwen3-VL-8B base, all panoramas in one prompt"),
]
MODELS = ("qwen25", "qwen3", "qwen3_B", "onecanvas", "qwen25base", "qwen3_band")
# readers judged by the re-check's degenerate rule: (key, backbone, unit)
READERS = [("A1", "MemoryExplorer Qwen2.5-VL-7B", "A1"), ("B", "MemoryExplorer Qwen2.5-VL-7B", "B"),
           ("A1_base", "Qwen2.5-VL-7B-Instruct", "A1"), ("band_base", "Qwen2.5-VL-7B-Instruct", "A1-band"),
           ("B_base", "Qwen2.5-VL-7B-Instruct", "B"), ("C_multi", "Qwen3-VL-8B", "A1"),
           ("C_band", "Qwen3-VL-8B", "A1-band"), ("C_B", "Qwen3-VL-8B", "B"), ("A2", "OneCanvas-Qwen3-VL-8B", "A2")]
TOP_SHARE = 0.90  # degenerate: one answer in at least this share of decisions
BACKBONE_GAP = 5.0
# P3-0 re-check step 3: navigation runs (analysis/p3_0_nav.py, Qwen3-VL-8B base, 58-task subset, seeds 77-79)
NAV_SEEDS = (77, 78, 79)
NAV = {"PA": "Panoramas (A1-band), two-step commit", "PB": "B: region markers + labels + 4 views, two-step commit",
       "F1": "F1: all kept frames, no regions"}
NAV_RUNS = {k: [f"p3r_{k}_s{sd}" for sd in NAV_SEEDS] for k in NAV}
NAV_KEYS = ["SR", "SPL", "premature_commit_rate", "distractor_rate_of_memory_choices", "step1_exact", "none_P", "none_R",
            "tokens_per_decision", "tokens_per_decision_max", "sec_per_decision", "empty_context_explore", "invalid_replies"]
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
    data = {(run, mdl): load(run, mdl) for run in RUNS for mdl in MODELS}
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
    qb = [r for run in RUNS for r in data[(run, "qwen25base")] if r["regions"]]
    q3b = [r for run in RUNS for r in data[(run, "qwen3_band")] if r["regions"]]
    if qb:
        st["A1 panorama image tokens per decision (Qwen2.5-VL-7B-Instruct)"] = [np.mean([r["A1_base_image_tokens"] for r in qb]), max(r["A1_base_image_tokens"] for r in qb)]
        st["A1-band image tokens per decision (Qwen2.5-VL-7B-Instruct)"] = [np.mean([r["band_base_image_tokens"] for r in qb]), max(r["band_base_image_tokens"] for r in qb)]
        st["A1-band rows per panorama (of 512)"] = [np.mean([h for r in qb for h in r["band_rows"]]), max(h for r in qb for h in r["band_rows"])]
        st["Grey share of an A1 panorama image (unfilled holes)"] = [np.mean([h for r in qb for h in r["hole_fraction"]]), max(h for r in qb for h in r["hole_fraction"])]
        st["Grey share of an A1-band image (unfilled holes)"] = [np.mean([h for r in qb for h in r["band_hole_fraction"]]), max(h for r in qb for h in r["band_hole_fraction"])]
    if q3b:
        st["A1-band image tokens per decision (Qwen3-VL-8B)"] = [np.mean([r["C_band_image_tokens"] for r in q3b]), max(r["C_band_image_tokens"] for r in q3b)]
    R["stats"] = {k: [float(v[0]), float(v[1])] for k, v in st.items()}
    R["recheck"] = recheck(R, data)
    R["nav"] = nav_results()
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


def recheck(R, data):
    """P3-0 re-check rules (DEV_PLAN.md): degenerate readers, then the backbone."""
    seen = R["subsets"]["goal region observed"]
    rand_seen = seen["rand"]["exact"][0]
    deg = {}
    for key, bb, unit in READERS:
        mdl = next(m for k, m, _ in OPTIONS if k == key)
        ch = [str(r.get(f"{key}_choice")) for run in RUNS for r in data[(run, mdl)]]
        if not ch:
            continue
        top, n = collections.Counter(ch).most_common(1)[0]
        ex_seen = seen[key]["exact"][0]
        deg[key] = {"backbone": bb, "unit": unit, "top_answer": top, "top_share": n / len(ch), "decisions": len(ch),
                    "exact_seen": ex_seen, "rand_seen": rand_seen, "exact_all": R["subsets"]["all"][key]["exact"][0],
                    "degenerate": bool(n / len(ch) >= TOP_SHARE or ex_seen <= rand_seen)}
    q25 = max(["A1_base", "band_base"], key=lambda k: deg[k]["exact_all"])
    q3 = max(["C_multi", "C_band"], key=lambda k: deg[k]["exact_all"])
    gap = deg[q3]["exact_all"] - deg[q25]["exact_all"]
    keep = (not deg[q25]["degenerate"]) and gap <= BACKBONE_GAP
    return {"readers": deg, "best_q25_panorama": q25, "best_q3_panorama": q3, "gap_q3_minus_q25": gap,
            "q25_not_degenerate": not deg[q25]["degenerate"], "within_gap": bool(gap <= BACKBONE_GAP),
            "backbone": "Qwen2.5-VL-7B-Instruct" if keep else "Qwen3-VL-8B",
            "nav_panorama_variant": {"Qwen2.5-VL-7B-Instruct": q25, "Qwen3-VL-8B": q3}["Qwen2.5-VL-7B-Instruct" if keep else "Qwen3-VL-8B"]}


def nav_run(run):
    """Navigation metrics of one run: Phase 2b's (p2b_report.run_metrics) + step 1 + cost per decision."""
    import p2b_report as PB  # noqa: E402  (analysis/ is on sys.path when run as analysis/p3_0_report.py)
    m = PB.run_metrics(run)
    _, _, decisions = PB.A.load_records(os.path.join(REPO, "analysis/runs", run))
    ds = [d for v in decisions.values() for d in v if "p2b" in d]
    s1 = [d["p3"] for d in ds if "step1_reply" in d.get("p3", {})]
    if s1:
        tgt = [next((r for r in p["regions"] if r in p["goal_regions"]), "explore") for p in s1]
        ch = [p["step1_choice"] for p in s1]
        tp = sum(c == "explore" and t == "explore" for c, t in zip(ch, tgt))
        m.update(step1_decisions=len(s1), step1_exact=100.0 * np.mean([c == t for c, t in zip(ch, tgt)]),
                 none_P=100.0 * tp / max(1, sum(c == "explore" for c in ch)),
                 none_R=100.0 * tp / max(1, sum(t == "explore" for t in tgt)),
                 step2_share=100.0 * np.mean([p["step2"] for p in s1]))
    cost = [(d["p2b"]["prompt_tokens"] + d.get("p3", {}).get("step1_tokens", 0),
             d["p2b"]["latency_s"] + d.get("p3", {}).get("step1_s", 0.0)) for d in ds]
    cost = [c for c in cost if c[0] > 0]  # decisions that called the VLM
    if cost:
        m.update(tokens_per_decision=float(np.mean([c[0] for c in cost])), tokens_per_decision_max=float(max(c[0] for c in cost)),
                 sec_per_decision=float(np.mean([c[1] for c in cost])))
    return m


def nav_results():
    """Groups of complete navigation runs, the reference rows from Phase 2b, and the unit rule."""
    import p2_report as P2  # noqa: E402
    pb = json.load(open(os.path.join(REPO, "analysis/out_p2b/p2b_results.json")))
    band = pb["noise"]["band"]
    G = {}
    for k, runs in NAV_RUNS.items():
        ok = [r for r in runs if os.path.isdir(os.path.join(REPO, "analysis/runs", r)) and P2.complete(r, 58)]
        if not ok:
            continue
        per = {r: nav_run(r) for r in ok}
        keys = [x for x in NAV_KEYS if all(per[r].get(x) is not None for r in ok)]
        G[k] = {"runs": ok, "missing": [r for r in runs if r not in ok], "per_run": per,
                "mean": {x: float(np.mean([per[r][x] for r in ok])) for x in keys},
                "std": {x: float(np.std([per[r][x] for r in ok], ddof=1)) if len(ok) > 1 else float("nan") for x in keys}}
    ref = {"E0 fixed (MemoryExplorer, retrieval bank)": pb["E0"],
           "F1 (MemoryExplorer, Phase 2b)": pb["F"]["F1"], "F2b (MemoryExplorer, Phase 2b)": pb["F"]["F2b"]}
    rule = {"band": band}
    if all(k in G and len(G[k]["runs"]) == len(NAV_SEEDS) for k in ("PA", "PB")):
        a, b = G["PA"]["mean"]["SR"], G["PB"]["mean"]["SR"]
        rule.update(PA=a, PB=b, diff=a - b, unit="A (panoramas)" if a >= b + band else "B (region markers)",
                    row="Panoramas >= B + band" if a >= b + band else ("Panoramas < B - band" if a < b - band else "Within the band of B"))
    return {"groups": G, "reference": {k: {"runs": v["runs"], "mean": v["mean"], "std": v["std"]} for k, v in ref.items()},
            "rule": rule}


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


def recheck_md(R):
    rc = R["recheck"]
    L = ["| Reader (backbone) | Unit | Exact region % (all) | Exact region % (goal's region seen) | Random same-type (seen) | "
         "Most common answer | Share % | Degenerate |", "|---|---|---|---|---|---|---|---|"]
    for k, v in rc["readers"].items():
        L.append(f"| {v['backbone']} | {v['unit']} (`{k}`) | {v['exact_all']:.1f} | {v['exact_seen']:.1f} | {v['rand_seen']:.1f} | "
                 f"{'not seen yet' if v['top_answer'] == 'none' else 'region ' + str(int(v['top_answer']) + 1) if v['top_answer'].isdigit() else v['top_answer']} | "
                 f"{100 * v['top_share']:.1f} | {'**yes**' if v['degenerate'] else 'no'} |")
    q25, q3 = rc["readers"][rc["best_q25_panorama"]], rc["readers"][rc["best_q3_panorama"]]
    B = ["| Check | Value | Holds |", "|---|---|---|",
         f"| Base Qwen2.5-VL-7B-Instruct not degenerate on panoramas (best variant `{rc['best_q25_panorama']}`) | "
         f"top answer {100 * q25['top_share']:.1f}% (< 90%); seen {q25['exact_seen']:.1f} vs random {q25['rand_seen']:.1f} | "
         f"{'yes' if rc['q25_not_degenerate'] else 'no'} |",
         f"| Its best panorama score within {BACKBONE_GAP:.0f} points of base Qwen3-VL-8B's (`{rc['best_q3_panorama']}`) | "
         f"{q25['exact_all']:.1f} vs {q3['exact_all']:.1f}: gap {rc['gap_q3_minus_q25']:.1f} | {'yes' if rc['within_gap'] else 'no'} |",
         f"| **Backbone** | both hold → Qwen2.5-VL-7B-Instruct; otherwise Qwen3-VL-8B (+ F1 on it) | **{rc['backbone']}** |",
         f"| Panorama variant for navigation (better of A1 / A1-band on that backbone) | `{rc['nav_panorama_variant']}` | — |"]
    return "\n".join(L), "\n".join(B)


def nav_md(R):
    N = R["nav"]
    G, ref, rule = N["groups"], N["reference"], N["rule"]
    f = lambda g, k, nd=1: "—" if k not in g["mean"] else (f"{g['mean'][k]:.{nd}f}" + ("" if np.isnan(g["std"].get(k, np.nan)) else f" ± {g['std'][k]:.{nd}f}"))  # noqa: E731
    L = ["| Run (Qwen3-VL-8B base, 3 seeds) | SR | SPL | Premature-commit % | Distractor % (mem. choices) | Step-1 exact region % | "
         "'Not seen yet' (explore) P / R % | Tokens per decision (mean / max) | s per decision | Empty-context explores per run | Invalid replies per run |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, lab in NAV.items():
        if k in G:
            g = G[k]
            L.append(f"| {lab} ({len(g['runs'])} runs) | {f(g, 'SR')} | {f(g, 'SPL')} | {f(g, 'premature_commit_rate')} | "
                     f"{f(g, 'distractor_rate_of_memory_choices')} | {f(g, 'step1_exact')} | {f(g, 'none_P')} / {f(g, 'none_R')} | "
                     f"{f(g, 'tokens_per_decision', 0)} / {g['mean'].get('tokens_per_decision_max', float('nan')):.0f} | {f(g, 'sec_per_decision')} | "
                     f"{f(g, 'empty_context_explore', 0)} | {f(g, 'invalid_replies', 0)} |")
        else:
            L.append(f"| {lab} | not run | | | | | | | | | |")
    for k, g in ref.items():
        L.append(f"| Reference: {k} | {f(g, 'SR')} | {f(g, 'SPL')} | {f(g, 'premature_commit_rate')} | "
                 f"{f(g, 'distractor_rate_of_memory_choices')} | — | — | {f(g, 'ctx_tokens_mean', 0)} / — | {f(g, 'sec_per_decision')} | "
                 f"{f(g, 'empty_context_explore', 0)} | {f(g, 'invalid_replies', 0)} |")
    P = ["| Run | SR | SPL | Premature-commit % | Step-1 exact % | Decisions | Step-1 calls | Step 2 run % |", "|---|---|---|---|---|---|---|---|"]
    for k in NAV:
        for r, m in G.get(k, {}).get("per_run", {}).items():
            P.append(f"| {r} | {m['SR']:.1f} | {m['SPL']:.1f} | {m['premature_commit_rate']:.1f} | "
                     f"{m['step1_exact']:.1f} | {m.get('decisions', 0)} | {m.get('step1_decisions', 0)} | {m['step2_share']:.1f} |"
                     if "step1_exact" in m else f"| {r} | {m['SR']:.1f} | {m['SPL']:.1f} | {m['premature_commit_rate']:.1f} | — | {m.get('decisions', 0)} | — | — |")
    U = ["| Result (SR, 3 seeds) | Unit | Holds |", "|---|---|---|"]
    if "unit" in rule:
        for row, unit in [("Panoramas ≥ B + band", "A (panoramas)"), ("Within the band of B", "B (simpler)"), ("Panoramas < B − band", "B")]:
            U.append(f"| {row} | {unit} | {'**yes**' if rule['row'].replace('>=', '≥').replace(' - ', ' − ') == row else 'no'} |")
        U.append(f"| PA {rule['PA']:.1f} vs PB {rule['PB']:.1f}: difference {rule['diff']:+.1f}, band {rule['band']:.1f} | **{rule['unit']}** | — |")
    else:
        U.append("| navigation runs incomplete | — | — |")
    return "\n".join(L), "\n".join(P), "\n".join(U)


def write_doc(R):
    keys = [k for k, _, _ in OPTIONS]
    A, rule = R["subsets"]["all"], R["rule"]
    per_seed = ["| Option | " + " | ".join(RUNS) + " |", "|---|" + "---|" * len(RUNS)]
    for k in keys:
        if k in A:
            per_seed.append(f"| {k} | " + " | ".join(f"{A[k]['runs'][r]['exact']:.1f}" if r in A[k]["runs"] else "—" for r in RUNS) + " |")
    stats = ["| Quantity | Mean | Max |", "|---|---|---|"] + [f"| {k} | {v[0]:.2f} | {v[1]:.2f} |" for k, v in R["stats"].items()]
    md = open(os.path.join(OUT, "decision_text.md")).read() if os.path.exists(os.path.join(OUT, "decision_text.md")) else ""
    readers, bb = recheck_md(R)
    nav, nav_per, unit = nav_md(R)
    rc, nr = R["recheck"], R["nav"]["rule"]
    doc = f"""# P3-0 decision: backbone and region unit (DEV_PLAN.md P3-0 and the P3-0 re-check)

Tables generated by `analysis/p3_0_report.py` from `analysis/out_p3_0/*.jsonl` (offline) and `analysis/runs/p3r_*`
(navigation); every number is in `analysis/out_p3_0/p3_0_results.json`. Reproduce: `bash analysis/p3_0_run_all.sh`
(offline stages) and `bash analysis/p3_0r_run_all.sh` (navigation runs), or `REPORT_ONLY=1 bash analysis/p3_0_run_all.sh`
(rescoring only). Rationale text: `analysis/out_p3_0/decision_text.md`.

**Backbone: {rc['backbone']}** (re-check step 2). **Region unit: {nr.get('unit', 'pending (navigation runs incomplete)')}** (re-check step 3).

{md}

## P3-0 re-check, steps 1–2: backbone control (offline, same 2,543 decisions, seeds 77–79)

Degenerate reader (DEV_PLAN.md): one answer in at least 90% of decisions, or exact-region accuracy on decisions whose
goal region was already seen not above random same-type. Every option's full P3-0a tables (all slices) are below.

{readers}

Backbone rule:

{bb}

## P3-0 re-check, step 3: navigation, panoramas vs B (two-step commit, ground-truth regions, empty-context rule, 58-task subset)

`analysis/p3_0_nav.py` on the Phase 2b wrapper; mean ± std over seeds {', '.join(map(str, NAV_SEEDS))}. Tokens and seconds
are per decision that called the VLM (step 1 + step 2, PB labels included). Reference rows are Phase 2b's
(`docs/phase2b_decision.md`), on MemoryExplorer's Qwen2.5-VL-7B.

{nav}

Per run:

{nav_per}

Unit rule (band = Phase 2b's pooled noise band):

{unit}

## P3-0a (27 Oct): offline scores, 58-task subset, fixed-baseline decisions, seeds 77–79

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

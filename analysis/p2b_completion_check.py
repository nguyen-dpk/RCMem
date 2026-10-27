"""Prints the "Phase 2b completion check" table of DEV_PLAN.md, each row checked from the files and git.
Run (repo root): python analysis/p2b_completion_check.py     (exit code 1 if a row fails)
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
os.chdir(REPO)
SCRIPTS = ["analysis/p2b_context.py", "analysis/p2b_f0.py", "analysis/p2b_experiments.py", "analysis/p2b_report.py",
           "analysis/p2b_run_all.sh", "analysis/runs/p2b_queue.sh", "analysis/p2b_completion_check.py"]
DOC = "docs/phase2b_decision.md"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout.strip()


def main():
    import p2_report as P2
    import p2b_report as R
    rows = []
    # 1 F0
    f0 = json.load(open("analysis/out_p2b/f0.json"))
    kinds = {k: [s for s in f0["settings_list"] if s.startswith(p)] for k, p in
             (("(a) all frames", "a_"), ("(b) frame-level", "b_"), ("(c) token-level", "c_"), ("(d) 2x2 pooling", "d_"))}
    print("== F0: analysis/out_p2b/f0.json (written by analysis/p2b_f0.py)")
    print(f"   records: {f0['runs']}, {f0['decisions']} decisions, {f0['episodes']} episodes")
    for s in f0["settings_list"]:
        v = f0["settings"][s]
        print(f"   {s:14s} mean {v['mean']:7.0f}  p95 {v['p95']:7.0f}  max {v['max']:7.0f}  <=16k {v['share_le_16k']:5.1f}%  <=32k {v['share_le_32k']:5.1f}%")
    ok1 = all(kinds.values()) and f0["decisions"] > 0
    rows.append(("F0 token distributions reported for all four settings, from a script over the existing records",
                 "analysis/p2b_f0.py -> analysis/out_p2b/f0.json; " + ", ".join(f"{k}: {len(v)}" for k, v in kinds.items())
                 + f" settings; {f0['decisions']} decisions from {', '.join(f0['runs'])}", ok1))
    # 2-3 F runs
    res = json.load(open("analysis/out_p2b/p2b_results.json"))
    need = ["SR", "SPL", "distractor_rate_all", "premature_commit_rate", "acc", "acc_prior_corrected",
            "ctx_tokens_mean", "ctx_tokens_max", "sec_per_decision"]
    print("== F runs: analysis/runs/p2b_ec_*/records_*.jsonl (written by analysis/p2b_experiments.py)")
    ok2 = ok3 = True
    for e, runs in R.F_RUNS.items():
        for r in runs:
            done = P2.complete(r, 58)
            m = res["F"].get(e, {}).get("per_run", {}).get(r)
            has = m is not None and all(m.get(k) is not None for k in need)
            ok2 &= done and has and m["episodes"] == 58
            ok3 &= has
            print(f"   {r:16s} complete(58)={done} episodes={m and m['episodes']} nav={m and m['n_nav']} qa={m and m['n_qa']} "
                  + (f"SR {m['SR']:.1f} SPL {m['SPL']:.1f} distr {m['distractor_rate_all']:.1f} prem {m['premature_commit_rate']:.1f} "
                     f"acc {m['acc']:.1f} pc-acc {m['acc_prior_corrected']:.1f} | tokens mean {m['ctx_tokens_mean']:.0f} max "
                     f"{m['ctx_tokens_max']} | s/call {m['sec_per_decision']:.1f}" if has else "MISSING"))
    rows.append(("F1, F2 and F2b implemented as wrappers and run on all 58 tasks, with every metric listed above",
                 "analysis/p2b_experiments.py; 9 runs p2b_ec_{F1,F2,F2b}_s{77,78,79}, each complete on 58 episodes with SR, SPL, "
                 "distractor, premature-commit, acc, prior-corrected acc", ok2))
    rows.append(("Context-length and latency numbers reported per run",
                 "per-run tokens (mean / p95 / max) and s per VLM call in analysis/out_p2b/p2b_results.json and the per-run table of "
                 + DOC, ok3))
    # 4 decision doc
    doc = open(DOC).read() if os.path.exists(DOC) else ""
    m = re.search(r"## Decision: \*\*(.+?)\*\*", doc)
    dec = m.group(1) if m else None
    g = res.get("gate", {})
    ok4 = dec in ("GO, implicit", "GO, implicit but training-dependent", "FALLBACK, external") and dec == g.get("decision") \
        and "| Gate G1.5 condition | Value | Holds |" in doc
    print(f"== {DOC}: decision = {dec}; p2b_results.json gate = {g.get('decision')}")
    print(f"   F2 - F1 = {g.get('F2_minus_F1', float('nan')):+.1f} vs band {g.get('band', float('nan')):.1f}; "
          f"F1 {g.get('F1', float('nan')):.1f} vs floor {g.get('E0_mean', float('nan')) - g.get('band', float('nan')):.1f}; F0 fits {g.get('f0_fits')}")
    rows.append(("docs/phase2b_decision.md states GO-implicit, GO-implicit-training-dependent or FALLBACK, justified against the rule with numbers",
                 f"{DOC}: \"{dec}\", with the G1.5 table (band {g.get('band', 0):.1f} SR = 2 x pooled E0 std; F2 - F1 "
                 f"{g.get('F2_minus_F1', 0):+.1f}; F1 {g.get('F1', 0):.1f} vs floor {g.get('E0_mean', 0) - g.get('band', 0):.1f})", ok4))
    # 5 scripts committed + one command
    tracked = git("ls-files", *SCRIPTS).split("\n")
    dirty = git("status", "--porcelain", "--", *SCRIPTS, DOC, "analysis/out_p2b")
    ok5 = set(tracked) == set(SCRIPTS) and dirty == "" and "bash analysis/p2b_run_all.sh" in doc
    print(f"== git ls-files: {len(tracked)}/{len(SCRIPTS)} scripts tracked; git status --porcelain on scripts, doc, outputs: "
          f"{'clean' if not dirty else dirty}")
    rows.append(("Scripts committed; one documented command reproduces every table",
                 f"{len(tracked)}/{len(SCRIPTS)} scripts tracked and clean at {git('rev-parse', '--short', 'HEAD')}; "
                 f"`bash analysis/p2b_run_all.sh` documented in {DOC}", ok5))
    # 6 no LMEE / baseline change
    diff = git("diff", "--stat", "p2-hygiene..HEAD", "--", "evaluation/", "analysis/runs/instr_choice_sub", "evaluation/results")
    ok6 = diff == ""
    print(f"== git diff --stat p2-hygiene..HEAD -- evaluation/ analysis/runs/instr_choice_sub evaluation/results: "
          f"{'(empty)' if ok6 else diff}")
    rows.append(("git diff shows no change to LMEE evaluation code or the original baseline results",
                 "`git diff --stat p2-hygiene..HEAD -- evaluation/ analysis/runs/instr_choice_sub evaluation/results` is empty", ok6))
    print("\n## Phase 2b completion check\n\n| Check | Evidence | Holds |\n|---|---|---|")
    for c, ev, ok in rows:
        print(f"| {c} | {ev} | {'yes' if ok else 'NO'} |")
    return 0 if all(ok for _, _, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

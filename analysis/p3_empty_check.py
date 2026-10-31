"""P3 check: exercise RCMem's empty-context rule in the simulator (DEV_PLAN P3 completion check, two-step commit row).

On LMEE the initial 360-degree look-around always observes a region before the first decision, so the rule ("no region
observed -> nearest frontier, no VLM call") never fires in the subset runs. This forces it: the region show threshold is
set above 100%, so no region is ever observed, and every navigation decision must take the rule's path. The VLM is
replaced by a function that fails if called for a navigation decision. Pass: every navigation decision record has
mask "no_region_seen" and parse "empty_context_explore" (or "empty_context_no_frontier"), and no VLM call was made.
Run (repo root, lmee env, one GPU): CUDA_VISIBLE_DEVICES=2 python analysis/p3_empty_check.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import p3_nav as M  # noqa: E402

RUN = "p3_empty_check"
M.N.MIN_SHARE = 2.0  # no frame can have > 100% of its pixels in a region -> no region is ever observed
CALLS = []
_call = M.F.call_vlm


def call_vlm(sys_prompt, content, model, processor):
    CALLS.append(sys_prompt[:60])
    return _call(sys_prompt, content, model, processor)


M.F.call_vlm = M.N._call_vlm = call_vlm

if __name__ == "__main__":
    if "--score" not in sys.argv:  # --score: rescore the saved records only (no simulator)
        import shutil
        shutil.rmtree(os.path.join(REPO, "analysis/runs", RUN), ignore_errors=True)
        a = M.F.parse(["--exp", "RP", "--vlm", "Qwen/Qwen3-VL-8B-Instruct", "--run_name", RUN, "--seed", "77",
                       "--cfg_file", os.path.join(REPO, "analysis/cfg/p3_pilot.yaml"), "--start_ratio", "0", "--end_ratio", "0.2"])
        M.run_job(a)
    nav = []
    for f in glob.glob(os.path.join(REPO, "analysis/runs", RUN, "records_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("type") == "decision" and r.get("p3", {}).get("exp") == "RP":
                nav.append((r["p2b"]["mask"], r["p2b"]["parse"], r["p2b"]["prompt_tokens"], r["p3"]["n_regions"]))
    ok = nav and all(m == "no_region_seen" and p.startswith("empty_context") and t == 0 and n == 0 for m, p, t, n in nav)
    print(f"navigation decisions: {len(nav)}; masks: {sorted({m for m, *_ in nav})}; parses: {sorted({p for _, p, *_ in nav})}; "
          f"prompt tokens: {sorted({t for *_, t, _ in nav})}; VLM calls (all, incl. QA): {len(CALLS)}")
    print("empty-context rule check:", "PASS" if ok else "FAIL")
    json.dump({"navigation_decisions": len(nav), "masks": sorted({m for m, *_ in nav}), "parses": sorted({p for _, p, *_ in nav}),
               "nav_prompt_tokens": sorted({t for *_, t, _ in nav}), "regions_observed": sorted({n for *_, n in nav}), "pass": bool(ok)},
              open(os.path.join(REPO, "analysis/out_p3/empty_check.json"), "w"), indent=1)

"""P3: offline accuracy of the region builder (rcmem/regions.py) against HM3D regions, and the choice of DOOR_R and H_MAX.

Replays the kept frames of a logged run (the P3-0 cache: 448 px RGB-D + HM3D semantics + poses, analysis/p3_0_cache.py)
in viewing order. At every navigation decision the builder segments the map built from the frames seen so far, and
the kept frames' points (every STRIDE-th pixel) are scored against their GT region (rcmem.regions.score: point
accuracy, coverage, agreement, merged and split regions). Means are over decisions; "episodes with a merge" counts
episodes whose last decision has a merged region. The setting with the best mean agreement is reported as "best"; rcmem keeps DOOR_R 0.4 when it is within 1 point, since
smaller radii no longer close a 0.9 m doorway (the self-check in rcmem/regions.py).
The online numbers in docs/p3_report.md come from the agent runs themselves (analysis/p3_nav.py logs the same scores).

Run (repo root, lmee env, CPU): python analysis/p3_regions.py [--run p2_E0_fix_s77] [--radii 0.3/0 0.3/0.5 ...]
Output: analysis/out_p3/regions_offline.json
"""
import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, REPO)
import analyze as A  # noqa: E402
from rcmem import regions as RG  # noqa: E402
from rcmem.geometry import backproject_habitat  # noqa: E402

CACHE = os.environ.get("P3_CACHE", "/ssd1/quang/embodied/outputs/p3_0_cache")
OUT = os.path.join(REPO, "analysis/out_p3")
STRIDE = 4


def episode(job):
    run, ep, objs, steps, radii = job
    z = np.load(os.path.join(CACHE, run, ep + ".npz"))
    inst_region = np.full(max(objs) + 2, "", dtype=object)
    for k, o in objs.items():
        inst_region[k] = o["region"] if o["region"] not in (None, "_-1") else ""
    pts, gts = [], []
    for fi in range(len(z["step"])):
        d = z["depth_mm"][fi][::STRIDE, ::STRIDE] / 1000.0
        p, ok = backproject_habitat(d, z["cam"][fi], z["yaw"][fi], hfov_deg=float(z["hfov"]), tilt=float(z["tilt"]))
        g = inst_region[np.clip(z["sem"][fi][::STRIDE, ::STRIDE].astype(np.int64), 0, len(inst_region) - 1)].reshape(-1)
        pts.append(p[ok])
        gts.append(g[ok])
    floor = float(z["cam"][0][1]) - RG.EYE
    out = {}
    for r in radii:
        door_r, h_max = (float(x) for x in r.split("/"))
        rb = RG.RegionBuilder(z["cam"][0], floor, door_r=door_r, h_max=h_max)
        rows, fi, t_seg = [], 0, []
        for s in steps:
            while fi < len(z["step"]) and z["step"][fi] <= s:
                rb.add(pts[fi])
                fi += 1
            if fi == 0:
                continue
            t0 = time.time()
            rb.segment()
            P = np.concatenate(pts[:fi])
            pred = rb.assign(P)
            t_seg.append(time.time() - t0)
            sc = RG.score(pred, np.concatenate(gts[:fi]))
            if sc:
                rows.append(sc)
        out[r] = {"rows": rows, "seg_s": t_seg}
    return ep, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p2_E0_fix_s77")
    ap.add_argument("--radii", nargs="+", default=["0.3/0", "0.4/0", "0.5/0", "0.3/0.3", "0.3/0.5", "0.4/0.5"],
                    help="settings DOOR_R/H_MAX (H_MAX 0: erosion seeds)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    scenes, subtasks, decisions = A.load_records(os.path.join(REPO, "analysis/runs", a.run))
    steps = collections.defaultdict(list)
    for sid, st in subtasks.items():
        if st["goal_type"] != "QA":
            steps[st["episode"]] += [d["global_step"] for d in decisions.get(sid, [])]
    eps = sorted(e for e in steps if os.path.exists(os.path.join(CACHE, a.run, e + ".npz")))
    if a.limit:
        eps = eps[:a.limit]
    jobs = [(a.run, e, scenes[A.scene_of(e)], sorted(steps[e]), a.radii) for e in eps]
    with Pool(min(a.procs, len(jobs))) as pool:
        res = dict(pool.map(episode, jobs))
    summ = {}
    for r in a.radii:
        rows = [x for e in eps for x in res[e][r]["rows"]]
        last = [res[e][r]["rows"][-1] for e in eps if res[e][r]["rows"]]
        segs = [t for e in eps for t in res[e][r]["seg_s"]]
        summ[str(r)] = {
            "decisions": len(rows), "episodes": len(last),
            **{k: float(np.mean([x[k] for x in rows])) for k in ("point_acc", "coverage", "agreement", "n_pred", "n_gt")},
            "merged_per_decision": float(np.mean([x["merged"] for x in rows])),
            "share_pred_regions_merged": float(sum(x["merged"] for x in rows) / max(1, sum(x["n_pred"] for x in rows))),
            "share_decisions_with_merge": float(np.mean([x["merged"] > 0 for x in rows])),
            "share_episodes_with_merge_at_end": float(np.mean([x["merged"] > 0 for x in last])),
            "split_per_decision": float(np.mean([x["split"] for x in rows])),
            "seg_s_mean": float(np.mean(segs)), "seg_s_max": float(np.max(segs)),
        }
    best = max(summ, key=lambda r: summ[r]["agreement"])
    os.makedirs(OUT, exist_ok=True)
    json.dump({"run": a.run, "stride": STRIDE, "door_r": summ, "best": best,
               "params": {"cell": RG.CELL, "low_hi": RG.LOW_HI, "wall": [RG.WALL_LO, RG.WALL_HI], "wall_min": RG.WALL_MIN,
                          "min_seed_m2": RG.MIN_SEED_M2, "min_region_m2": RG.MIN_REGION_M2, "merge_share": RG.MERGE_SHARE}},
              open(os.path.join(OUT, "regions_offline.json"), "w"), indent=1)
    print("| DOOR_R / H_MAX | decisions | point acc % | coverage % | agreement % | pred / GT regions | merged regions % | "
          "decisions with a merge % | split per decision | seg s |")
    for r, s in summ.items():
        print(f"| {r} | {s['decisions']} | {100 * s['point_acc']:.1f} | {100 * s['coverage']:.1f} | {100 * s['agreement']:.1f} | "
              f"{s['n_pred']:.1f} / {s['n_gt']:.1f} | {100 * s['share_pred_regions_merged']:.1f} | "
              f"{100 * s['share_decisions_with_merge']:.1f} | {s['split_per_decision']:.2f} | {s['seg_s_mean']:.2f} |")
    print("best DOOR_R / H_MAX:", best)


if __name__ == "__main__":
    main()

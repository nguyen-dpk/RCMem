"""Phase 2: ground-truth check of E4's verify-on-arrival decisions (analysis/runs/p1_E4).

For every verification (record type p1_verify) the checked object is matched to ground truth with the same rule as
analyze.py: it is the goal if the detection was IoU-mapped to a goal instance during the run, or if its nearest
GT instance (within 1.5 m; restricted to the goal's category group when the detector called it that category) is a
goal instance. Positive class for precision/recall = "the checked object is NOT the goal" (what a rejection should catch).
Run (repo root): python analysis/p2_verifier.py   ->  analysis/out_p2/verifier.json
"""
import glob
import json
import os

import analyze as A

REPO = A.REPO
RUN = os.path.join(REPO, "analysis/runs/p1_E4")
OUT = os.path.join(REPO, "analysis/out_p2")


def main():
    os.makedirs(OUT, exist_ok=True)
    scenes, rows = {}, []
    for f in sorted(glob.glob(os.path.join(RUN, "records_*.jsonl"))):
        last_dec, sub_maps, pending = {}, {}, []
        recs = [json.loads(l) for l in open(f)]
        for r in recs:
            if r["type"] == "scene":
                scenes[r["scene_id"]] = {o["sid"]: o for o in r["objects"]}
            elif r["type"] == "subtask":
                sub_maps[r["subtask"]] = r
        for r in recs:
            if r["type"] == "decision":
                last_dec[r["subtask"]] = r
            elif r["type"] == "p1_verify":
                d, st = last_dec.get(r["subtask"]), sub_maps.get(r["subtask"])
                if d is None or st is None:
                    continue
                objs = scenes[A.scene_of(r["subtask"])]
                G = {int(x) for x in st["goal_obj_ids"]}
                gcat = objs.get(min(G), {}).get("cat", "?")
                mapped = {int(x) for v in st["goal_obj_ids_mapping"].values() for x in v}
                o = d["objects"].get(str(r["object"]))
                sid = None
                if o is not None:
                    sid, _ = A.nearest_gt(objs, o["center"], group=A.canon(gcat) if A.canon(o["cls"]) == A.canon(gcat) else None)
                is_goal = r["object"] in mapped or (sid is not None and sid in G)
                rows.append({"subtask": r["subtask"], "accepted": r["accepted"], "is_goal": is_goal,
                             "goal_type": st["goal_type"], "matched": o is not None})
    tp = sum(not r["accepted"] and not r["is_goal"] for r in rows)   # rejected a wrong object
    fp = sum(not r["accepted"] and r["is_goal"] for r in rows)       # rejected the true goal
    fn = sum(r["accepted"] and not r["is_goal"] for r in rows)       # accepted a wrong object
    tn = sum(r["accepted"] and r["is_goal"] for r in rows)           # accepted the true goal
    res = {"verifications": len(rows), "object_found_in_records": sum(r["matched"] for r in rows),
           "confusion": {"rejected_wrong_object": tp, "rejected_true_goal": fp, "accepted_wrong_object": fn, "accepted_true_goal": tn},
           "reject_precision": A.pct(tp, tp + fp), "reject_recall": A.pct(tp, tp + fn),
           "accept_precision": A.pct(tn, tn + fn), "accept_recall": A.pct(tn, tn + fp),
           "true_goal_rejected_share": A.pct(fp, fp + tn),
           "by_goal_type": {t: {"n": sum(r["goal_type"] == t for r in rows),
                                "rejected_true_goal": sum(r["goal_type"] == t and not r["accepted"] and r["is_goal"] for r in rows),
                                "accepted_true_goal": sum(r["goal_type"] == t and r["accepted"] and r["is_goal"] for r in rows)}
                            for t in sorted({r["goal_type"] for r in rows})}}
    json.dump(res, open(os.path.join(OUT, "verifier.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

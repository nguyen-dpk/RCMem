"""Drop records of episodes that did not finish (the LMEE logger only saves results at episode end and
re-runs unfinished episodes on restart). Usage: python analysis/clean_partial.py <start_ratio> <end_ratio> [run_name]"""
import json, os, pickle, sys
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", sys.argv[3] if len(sys.argv) > 3 else "instr_choice_sub")
s, e = float(sys.argv[1]), float(sys.argv[2])
rec = os.path.join(RUN, f"records_{s}_{e}.jsonl")
pk = os.path.join(RUN, f"success_by_snapshot_{s}_{e}_1.pkl")
done = set()
if os.path.exists(pk):
    done = {"_".join(k.split("_")[:-1]) for k in pickle.load(open(pk, "rb"))}  # "<scene>_<ep>"
if os.path.exists(rec):
    lines = open(rec).read().splitlines()
    keep = [l for l in lines if json.loads(l)["type"] == "scene" or
            json.loads(l)["episode"].replace("_ep_", "_") in done]
    open(rec, "w").write("".join(x + "\n" for x in keep))
    print(f"{rec}: kept {len(keep)}/{len(lines)} records; finished episodes {len(done)}")

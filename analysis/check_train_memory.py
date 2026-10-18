"""Check how memory entries in LMEE training data were built (needs pyarrow: run with the memoryexplorer env).
Prints: memory size stats, step-index spacing, share of entries that are 'stop_for' (goal) frames."""
import re, sys, collections
import numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data/LMEE-train/train_data.parquet")
step = lambda p: int(re.search(r"/(\d+)_", p).group(1))
sizes, n_stop, n_tot, gaps, has_pos, qa_rows, qa_with_stop = [], 0, 0, [], 0, 0, 0
for _, r in d.iterrows():
    mem = list(r["memory"])
    sizes.append(len(mem))
    st = [step(m["img_path"]) for m in mem]
    gaps += list(np.diff(sorted(st)))
    stops = ["stop_for" in m["img_path"] for m in mem]
    n_stop += sum(stops); n_tot += len(mem)
    has_pos += all("pos" in m for m in mem)
    if r["qa_question"]:
        qa_rows += 1; qa_with_stop += any(stops)
print("rows", len(d), "| memory size mean/min/max", np.mean(sizes), min(sizes), max(sizes))
print("entries that are stop_for (goal-arrival) frames:", n_stop, "/", n_tot, f"= {100*n_stop/n_tot:.1f}%")
print("step-gap histogram (top 8):", collections.Counter(gaps).most_common(8))
print("rows where every entry has 'pos':", has_pos, "/", len(d))
print("QA rows:", qa_rows, "| QA rows whose memory has >=1 stop_for frame:", qa_with_stop)
print("entry keys:", sorted(d.iloc[0]["memory"][0].keys()))

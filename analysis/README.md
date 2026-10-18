# Gap-search analysis (MemoryExplorer on LMEE-Bench, 58-task subset)

One command reproduces every number in `docs/gap_report.md` from the saved run outputs:

```bash
bash analysis/run_all.sh            # ~1 min, CPU only
RERUN=1 GPU=0 bash analysis/run_all.sh   # also redo the GPU stages (instrumented re-run, GT question views)
```

| File | What it does |
| --- | --- |
| `instrumented_run.py` | Re-runs the baseline (`data/MemoryExplorer`, choice answers, same config as `evaluation/results/exp_eval_lmee_choice`) and logs per-frame GT instance visibility (semantic sensor), full CLIP rankings at every retrieval, memory contents and choices. LMEE code is wrapped at runtime, not modified. |
| `runs/run_split.sh`, `clean_partial.py` | Run one scene split with OOM retries; resume per episode without duplicate records. |
| `gt_question_views.py` | Finds the pose each QA image was taken from (CLIP-ranked pose search + pixel refinement) and records the HM3D instances visible in it. |
| `labels/gt_view_ok.json` | Manual check of all 145 GT-view matches (106 verified). |
| `labels/nav_manual_check.json`, `labels/qa_manual_check.json` | Manual check of random failure samples (images from `render_cases.py`). |
| `analyze.py` | Failure taxonomy (navigation + QA) and diagnostics a–e → `out/summary.json`, `out/nav_items.csv`, `out/qa_items.csv`. |
| `check_baseline_facts.py` | Numbers behind the verified baseline facts. |
| `check_train_memory.py` | How training memories were built (needs pyarrow). |
| `render_cases.py` | Re-renders stored views for manual inspection → `out/cases/`. |

Outputs of the GPU stages are kept in `runs/instr_choice_sub/` (records_*.jsonl, logs, LMEE result files) and
`out/gt_question_views.json`, `out/gt_view_pairs/`.

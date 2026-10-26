## Decision: **B** — region markers with VLM labels and a few views per region

P3-0a fails, so by the plan's rule the region unit is option B (option C stays the fallback if B falls short in P3).
P3-0b is **skipped**: the plan runs it only when P3-0a passes.

**Against the rule (all navigation decisions, mean ± std over seeds 77–79, tables below):**

- **A1 vs B.** A1 is the best panorama option at 38.5 ± 2.6 exact-region accuracy. B scores 41.8 ± 1.9, so A1 is
  3.2 points *below* B. The rule needs A1 to be at least 10 points *above* B.
- **Random same-type.** A1 does beat random same-type (34.7 ± 0.5), but that is only the second condition and does
  not save the rule.
- **A2.** A2 scores 17.6 ± 1.6, below random, so the "prefer A1 if within 5 points" clause does not apply.
- **Per-seed results.** A1 never clears B by 10 points: 36.9 / 37.2 / 41.5 against 42.5 / 43.2 / 39.6.
- **Description subtasks.** This is where A2 sees the same information as the others. A1 scores 45.0 against B's
  42.2, which is still well under +10.

**Is B a fair comparator?** No, not on Qwen2.5-VL-7B. B on the MemoryExplorer checkpoint nearly always answers
"not seen yet": recall 99.8%, precision 41.7%, and 0.6% exact-region accuracy when the goal region was already
observed. Its 41.8 is therefore close to the share of decisions where "not seen yet" is the right answer.

I added the same comparison on the stronger backbone, which is not degenerate:

| Qwen3-VL-8B base | Exact region % |
| --- | --- |
| RGB panoramas, all regions in one prompt (control) | 53.4 ± 0.8 |
| Option B (labels + 4 views per region) | 44.8 ± 0.6 |

The margin is +8.6 points, so panoramas lose to B on both backbones under the ≥10-point bar. That makes the decision
robust to which backbone is used.

The Qwen3 margin is the closest call in this phase. If the provisional threshold is lowered, or P3 moves to
Qwen3-VL-8B, re-check A against B there before building on B.

**Why A2 is weak.** The canvas pipeline itself works. The sanity check in `analysis/out_p3_0/onecanvas_sanity.txt`
asked OneCanvas short questions on two GT regions and got 6/6 right, including room type and yes/no. A2's failure is
the per-region yes/no navigation instruction, which the checkpoint was not trained on. Its P(yes) stays high for every
region, so A2 never says "not seen yet" (recall 0.0). Multi-canvas prompts are not usable with the released checkpoint
(no `temporal_raw_frame_index`). A2 is therefore not a fair test of trained canvases, only of the zero-shot release.

**Cost.** Per decision, an RGB panorama costs about the same as B:

| Option | Image tokens per decision (Qwen2.5-VL) |
| --- | --- |
| A1 | 3549 |
| B | 3070 |

Building the panoramas takes 0.06 s per decision, but they are mostly holes: the mean hole fraction is 0.71 of the
full sphere. A2 canvases cost about 6385 tokens per decision.

**Setup and caveats.**

- **Decisions.** Logged navigation decisions of the fixed baseline (p2_E0_fix_s77/78/79, 58-task subset, 2543
  decisions in total). The F1 logs were not available, so only fixed-baseline decisions are used.
- **Regions.** Regions and region types are ground truth (HM3D regions; types from `analysis/out_p2/room_types.json`).
  Views were re-rendered at 448 px and pruned at 0.05 m voxels, keeping a view only if ≥10% of its voxels are new.
- **A1 and B backbone.** Both use MemoryExplorer's fine-tuned Qwen2.5-VL-7B, the baseline's own model. The
  Qwen3-VL-8B rows are base-model controls.
- **A2 setup.**
  - It uses OneCanvas-Qwen3-VL-8B (vendored at commit 5c9c587) with `sdpa` attention, not flash-attention.
  - The tokenizer printed a regex warning on load.
  - On image subtasks A2 gets only the text, not the target image.
- **Random same-type** is scored in expectation over observed regions whose type matches the goal type (or is unknown).

# Research-gap report: VLM memory mechanism on LMEE-Bench

Branch `gap-search`, 2026-10-16. Every number below is produced by a script in `analysis/` (`bash analysis/run_all.sh`; main output `analysis/out/summary.json`). Papers are listed with the arXiv URL that was opened. Anything not verified is marked **UNVERIFIED**.

## Summary

**Recommended gap: room-scoped commitment.** MemoryExplorer's memory has no notion of *which room a remembered object is in* or *which room the current subtask refers to*. So the agent commits to a remembered object anywhere in the house, usually in the wrong room and usually before it has even seen the subtask's room. This premature or wrong-instance commitment accounts for **98 of 214 navigation failures (45.8%)**. **87.5%** of same-category distractor choices and **97.4%** of other-category premature choices lie in a different HM3D region than the goal. The strict upper bound on the gain is **+36.0 SR points** (21.3 → at most 57.4). The QA side of the same gap is small: **8 of 145** questions (+5.5 points at most) are answered wrongly from another room while a view of the goal's room was in memory.

Several other hypotheses do not hold on LMEE:
- **Retrieval:** recall of the gold memory is 84–91% (≤ +2.8 points).
- **Multi-view evidence:** only **2 of 145** questions need more than one view (≤ +1.4 points).
- **QA accuracy barely measures memory:** answering "B" every time scores **59.3%**, and accuracy is 61.8% when the evidence was never observed versus 67.9% when it is in a single stored view.

## 1. Verified baseline facts

| Claim in GAP_SEARCH.md | Verdict | Evidence |
| --- | --- | --- |
| Memory entry per sampled step = position + CLIP text features (from an image-tagging model) + CLIP image features | **True for training, wrong for evaluation.** | **Evaluation:** an entry is a 3D-Mem-style *snapshot*, i.e. one stored frame chosen to cover a spatial cluster of detected 3D objects (`evaluation/src/scene.py:713-785`). Its text is the comma-joined **YOLO-World ScanNet-200 class names** of the cluster, not image-tagging output (`evaluation/src/eval_utils_qwen_lmee.py:563-564`; `run_lmee.py:63`; `cfg/eval_lmee_bench_choice.yaml:27`). Both indices use HF CLIP ViT-L/14 (`eval_utils_qwen_lmee.py:25-35`). The knowledge-base record (`:62-68`) stores no position; positions exist only as 3D boxes used for clustering. **Training:** one entry per sampled step with tagging-model tags, `pos`, and image (`check_train_memory.py`: every entry has `action, img_path, object, pos`; tags such as "room", "hardwood"). Retrieval uses only text and image (`git show HEAD:train/verl/tooluse/memory_tool.py:97-131`), so training and evaluation memories differ in unit and in text source. |
| Retrieval = single-round top-k (k = 3) CLIP cosine against a generated query; position used for writing, not retrieval | **Partly wrong.** | Single round: yes. The query comes from a separate MLLM call (`eval_utils_qwen_lmee.py:389-463`). But retrieval returns the **union of text-index top-3 and image-index top-3** (`:87-104`, `:107-130`, k from `cfg:22`), so **up to 6** memories reach the MLLM. Measured: 87.4% of 856 decisions passed more than 3 memories, mean 4.73, max 6 (`check_baseline_facts.py`, from the baseline log). Position is not used in retrieval: confirmed. Note that FAISS `IndexFlatL2` on L2-normalised vectors ranks the same as cosine. |
| QA prompt makes the agent choose one memory and answer from it | **True.** | `eval_utils_qwen_lmee.py:368-382` ("you must select one most relevant memory"). Only the ≤6 retrieved memories are shown. The answer is parsed from the last line of the reply (`:789-803`). The QA prompt contains only the question and choices (`logger_lmeebench.py:494-501`), with no task instruction and no indication of which subtask or object the question concerns. |
| Training samples had goal-related memories inserted by hand | **Consistent with the data; insertion code not in repo.** | The base memory is sampled every 10 steps (34,135 gaps of 10), but **34.3%** of training entries are off-grid `stop_for_<goal>` frames, and 10,794 of 11,684 rows (92.4%) contain at least one (`check_train_memory.py`). The insertion script is not released (**UNVERIFIED** as code). Paper Sec. 12 states it. |
| Backbone Qwen2.5-VL-7B, GRPO, 50-step episode cap | **True, with one correction.** | `data/MemoryExplorer/config.json` is `Qwen2_5_VLForConditionalGeneration` (hidden 3584, 28 layers, i.e. 7B). GRPO: `train/examples/lmee_config.yaml:22` (KL 0.1, `:26`). The upstream script uses lr 1e-6 and 5 rollouts (`git show HEAD:train/train_my.sh:25,28`). The cap is `max(int(sqrt(area)*2), 50)` steps **per subtask**, not per episode (`run_lmee.py:146-147,221`); it equals 50 in all 50 logged episodes. |

Implementation issues found along the way:
- **Start orientation is NaN in 58 of 58 episodes.** LMEE stores the start quaternion as `[w,x,y,z]`, but `get_pts_angle_goatbench` reads `[x,y,z,w]` (`evaluation/src/utils.py:47-54`). The yaw axis component is 0, so the angle is 0/0. `get_quaternion(NaN)` returns the identity (`src/habitat.py:123-131`), and all 7 initial look-around views are the same view.
- **Navigation goals are matched to detections** only if the goal covers more than 0.01% of the frame and IoU > 0.6 (`scene.py:463,474`).

**Reproduction check.** The user's baseline (`evaluation/results/exp_eval_lmee_choice`) scores SR 23.90, SPL 14.53, accuracy 64.83. The paper's Table 2 reports 23.53, 14.99 and 65.52, all within 5 points. My instrumented re-run of the same configuration and code (`analysis/instrumented_run.py`, LMEE code wrapped at runtime and not modified) scores **SR 21.32 and accuracy 65.52**, with per-goal success agreement of 84.9% against the baseline. All analysis below uses the re-run, because only it records ground truth.

## 2. Failure taxonomy (58-task subset: 272 goals, 145 questions)

Ground truth used:
- **Visibility:** semantic-sensor visibility of every HM3D instance in every stored view. "Visible" means at least 0.1% of the frame; results at 200 px and 0.5% are in `analysis/out/summary_minpx*.json` and the category order is stable.
- **Objects and rooms:** HM3D instance categories, centres and region IDs. Region IDs equal the task configs' "Region N" for 757 of 757 goals.
- **Retrieval and choices:** full CLIP rankings at every retrieval, and the agent's choices.
- **Distractors:** same-category groups merge HM3D synonyms such as picture, painting and wall-hanging decoration (`analyze.py:GROUPS`).

Each failure gets exactly one label, in priority order (`analyze.py:nav_category`, `qa_category`).

### Navigation: 214 failures out of 272 goals

| Category | Count | Share of failures | How labelled | Example subtask IDs |
| --- | --- | --- | --- | --- |
| Premature commit: chose a remembered object of another category before the goal was ever visible | 39 | 18.2% | GT (visibility, instance category). Manual: 4/4 agree. | 00820-mL8ThkuaVTM_1_1, 00813-svBbv1Pavdk_1_0 |
| Wrong instance, goal not yet observed: chose a same-category distractor | 36 | 16.8% | GT (instance positions and category groups). Manual: 4/4 agree. | 00829-QaLdnwvtxbs_0_0, 00820-mL8ThkuaVTM_0_0 |
| Perception: goal visible in a stored view but never matched to a detection | 26 | 12.1% | GT (visibility vs IoU mapping). Manual: 1/3 agree, 2 unclear. | 00829-QaLdnwvtxbs_0_4, 00813-svBbv1Pavdk_1_8 |
| Exploration: aborted because no frontier was available and no memory was chosen | 25 | 11.7% | Run record (frontier count 0) | 00820-mL8ThkuaVTM_2_2, 00821-eF36g7L6Z9M_0_0 |
| Wrong instance, goal already observed | 23 | 10.7% | GT. Manual: 4/4 agree. | 00813-svBbv1Pavdk_0_2, 00813-svBbv1Pavdk_1_4 |
| Perception: false-positive goal detection (no goal-category instance within 1.5 m) | 20 | 9.3% | GT | 00820-mL8ThkuaVTM_1_2, 00813-svBbv1Pavdk_0_5 |
| Task progress: chose an earlier subtask's goal again | 10 | 4.7% | GT | 00829-QaLdnwvtxbs_0_2, 00829-QaLdnwvtxbs_0_3 |
| Selection: gold memory retrieved and shown, but another object chosen | 10 | 4.7% | GT (mapping and rankings) | 00813-svBbv1Pavdk_0_0, 00813-svBbv1Pavdk_1_2 |
| Other: invalid MLLM output while frontiers existed | 10 | 4.7% | Run record | 00813-svBbv1Pavdk_0_4, 00808-y9hTuugGdiq_1_4 |
| Exploration: step cap reached | 6 | 2.8% | Run record | 00871-VBzV5z6i1WS_0_0, 00810-CrMo8WxCyVb_0_2 |
| Retrieval miss: gold memory existed but was never in the retrieved set | 3 | 1.4% | GT | 00824-Dd4bFSTQ8gi_0_5, 00873-bxsVRursffK_0_3 |
| Perception: goal detected but never written to a memory | 3 | 1.4% | GT | 00800-TEEsavR23oF_0_0, 00800-TEEsavR23oF_0_2 |
| Goal chosen but viewpoint not reached within 1 m | 3 | 1.4% | GT | 00824-Dd4bFSTQ8gi_0_3, 00810-CrMo8WxCyVb_1_4 |

"Other" is 4.7%. Of the hypothesised categories, "spatial-relation / counting" does not apply to navigation and "multi-view needed" is a QA category. The manual checks are in `analysis/labels/nav_manual_check.json`; the images are re-rendered with `analysis/render_cases.py`.

### QA: 50 wrong answers out of 145

| Category | Count | Share of wrong answers | How labelled | Example subtask IDs |
| --- | --- | --- | --- | --- |
| Exploration: the question's evidence was never visible in any stored view | 21 | 42.0% | GT (question-view instances, see below). Manual: 2/2 plausible. | 00829-QaLdnwvtxbs_0_6, 00820-mL8ThkuaVTM_1_3 |
| Selection: a gold memory was retrieved but another memory was chosen | 9 | 18.0% | GT. Manual: 2/3 agree, 1 unclear. | 00877-4ok3usBNeis_2_6, 00835-q3zU7Yy5E5s_0_5 |
| Counting error (gold chosen) | 5 | 10.0% | GT gold + benchmark category. Manual: 1 plausible. | 00813-svBbv1Pavdk_0_6, 00808-y9hTuugGdiq_0_5 |
| Spatial-relation error (gold chosen) | 5 | 10.0% | GT gold + benchmark category. Manual: 2/2 agree. | 00877-4ok3usBNeis_0_4, 00808-y9hTuugGdiq_1_7 |
| Retrieval miss | 4 | 8.0% | GT. Manual: 2/2 agree. | 00877-4ok3usBNeis_1_5, 00871-VBzV5z6i1WS_0_7 |
| Perception or attribute error (gold chosen) | 3 | 6.0% | GT gold + benchmark category | 00820-mL8ThkuaVTM_2_5, 00877-4ok3usBNeis_0_3 |
| Multi-view needed | 2 | 4.0% | GT | 00800-TEEsavR23oF_0_5, 00853-5cdEh9F2hJL_0_6 |
| Memory writing: evidence observed but not kept in any memory view | 1 | 2.0% | GT | 00853-5cdEh9F2hJL_0_7 |

"Other" is 0%. **Gold memory definition.** Each LMEE question was generated from one trajectory image (`task_test/.../front.png`). `analysis/gt_question_views.py` recovers that image's pose: a CLIP-ranked search over navigable poses within 6 m of the goal, refined by pixel error. All 145 matches were checked by eye, and **106 are correct** (`analysis/labels/gt_view_ok.json`). The question's relevant instances are those visible in the verified view whose HM3D category is named in the question, correct choice or open answer (84 questions). For the other 61 questions the fallback is the QA's own goal instance. A memory is gold if its stored view shows every relevant instance.

## 3. Ground-truth diagnostics

All computed by `analysis/analyze.py` from `analysis/runs/instr_choice_sub/records_*.jsonl` and `analysis/out/gt_question_views.json`.

**a. QA answers that need more than one stored view.**

| Evidence status | Questions | Accuracy |
| --- | --- | --- |
| A single stored memory view covers the evidence | 81 (55.9%) | 67.9% |
| Multi-view needed (covered by the union of memory views, not by any single one) | 2 (1.4%) | 0% |
| Observed but not kept in memory | 7 (4.8%) | 85.7% |
| Never observed | 55 (37.9%) | 61.8% |

- On the 84 questions with a verified view, multi-view is 2 of 84.
- With stricter visibility (0.5% of the frame), multi-view rises to 5 of 145.
- **Multi-view need is rare on LMEE**, because the questions were generated from single frames.

**b. Same-category distractors.**
- **80.1%** of goals (218/272) have a same-category instance elsewhere in the scene (median 6); 50.0% have one in the same region.
- The agent picks a distractor in **72 of 222** memory choices (32.4%); **87.5%** of those are in another region.
- SR is **16.1%** with distractors against **42.6%** without.
- In 32.2% of wrong-instance failures the goal's region had already been observed; in the rest it had not.
- The goal's room type appears somewhere in the instruction for 98.2% of goals. This is weak evidence: it only shows the word occurs, not that it is bound to the goal.

**c. Retrieval recall of the gold memory.** Recall counts the union of text top-3 and image top-3, which is what the baseline passes on.

| | Recall (union) | Text top-3 | Image top-3 |
| --- | --- | --- | --- |
| QA (81 questions with a single-view gold) | **84.0%** | 60.5% | 71.6% |
| Navigation, gold defined by visibility (135 goals) | **91.1%** | 71.9% | 86.7% |
| Navigation, gold defined by detection mapping (73 goals) | 86.3% | – | – |

- On average 23.4 memories exist and 5.1 are retrieved.
- QA accuracy is about the same whether the gold was retrieved or not: 67.6% versus 69.2%.

**d. Exploration and step-cap failures by difficulty.**

| Difficulty | Navigation failures | Exploration failures | Step cap reached | Goal never seen |
| --- | --- | --- | --- | --- |
| Easy | 33 | 39.4% | 0% | 81.8% |
| Medium | 143 | 31.5% | 14.0% | 53.8% |
| Hard | 38 | 31.6% | 0% | 57.9% |

"Exploration failures" here are premature commits plus aborts plus step cap. The step cap is rarely binding. Most "goal never seen" failures are commitments to a remembered object or aborts, not running out of steps.

**e. Other patterns.**
1. **The answer-letter prior dominates QA.** Gold letters are B 86, A 29, C 28, D 2, so answering "B" every time scores **59.3%**. The model predicts B 103 times. Evidence never observed gives 61.8% and single-view gold gives 67.9%. QA accuracy is 63.6% when the question's goal was reached in navigation and 66.1% when it was not.
2. **Missing QA evidence follows from navigation failure.** Evidence was never observed for 48.2% of questions whose goal was not reached, against 3.0% when it was.
3. **Room confusion in QA.** Only 44.1% of answers come from a view whose dominant region is the goal's region (accuracy 68.8%, against 63.0% for other regions). For 18.6% of questions a view of the goal's room was in memory but not used; 5.5% of all questions are wrong for that reason.
4. **The frontier map empties and every later subtask aborts.** This happens in 12 episodes and causes 25 aborts, 20 of them at step 1 (`diag_e_aborts`). The link to the NaN start-pose bug is **UNVERIFIED**.
5. **Training and evaluation memories differ** (Section 1): per-step tag memory with inserted gold frames in training, cluster snapshots without inserted gold in evaluation.

## 4. Candidate gaps

### G1 (recommended): room-scoped commitment in episodic memory

**Statement.** An episodic memory that tags every stored view and object instance with its 3D region, and binds each subtask of a compound instruction to a room and visit order, should stop the agent committing to remembered objects outside the subtask's room. It should also send the agent to explore that room when it has not yet been observed.

**LMEE evidence.**
- Premature commit and wrong instance account for 98 of 214 navigation failures (45.8%) (`candidates.G1_room_scoped_commitment_nav`).
- 87.5% of distractor choices and 97.4% of premature choices are in a non-goal region.
- In only 32.2% of wrong-instance failures had the goal's region already been observed.
- On the QA side, 8 of 145 are wrong because the answer came from another room although the goal room was in memory.
- Manual check: 12 of 12 sampled cases confirm, e.g. a doormat in the hall chosen for a bedroom rug.

**Upper bound.** Navigation: **+36.0 SR points** (every such failure becomes a success; strict upper bound). QA: **+5.5 points**.

**Closest prior work (methods read):**

| Paper | What it does | Why the gap remains |
| --- | --- | --- |
| STEGNav, [2608.28279](https://arxiv.org/html/2608.28279) | Instance scene graph with a room label per node; VLM grounding over ≤10 candidates "judged by appearance and spatial context"; stop verification; ≤8 verified outcome anchors across subtasks; GOAT-Bench 66.3 SR. | Training-free modular pipeline that disambiguates *after* candidate retrieval and verifies on arrival. Subtasks are given one at a time (GOAT), so there is no binding of a compound instruction's room sequence. No memory QA. |
| EvoMemNav, [2606.03509v1](https://arxiv.org/html/2606.03509v1) | Room → view → object memory graph with CLIP room-type labels; low-confidence anchor choice sends the agent back to exploration; STOP/RESELECT verification; GOAT 59.6 SR. | Closest mechanism. Goal signatures are per modality or class and never bound to an instruction's room order. No QA. Not tested with a learned memory policy. |
| Context-Nav, [2603.09506](https://arxiv.org/html/2603.09506) | Caption-conditioned value map; checks the candidate and context objects are in the same room; viewpoint-based relation checks; discards and keeps exploring. | Single-goal instance navigation with no persistent multi-goal memory and no QA. |
| MCNav, [2605.19594](https://arxiv.org/html/2605.19594) | Object-node cognitive map; re-validation, blacklist, double-check before confirming. | One goal per episode; no room binding; no QA. |
| MemoryEQA, [2505.13948](https://arxiv.org/html/2505.13948) | Memory entries carry a VLM room label and pose; entropy-adaptive retrieval; confidence-based stopping. | QA-only exploration with no navigation subtasks and no instance commitment. |
| R4, [2512.15940](https://arxiv.org/html/2512.15940) (already on the ruled-out list) | Separate semantic, spatial (3D centroid) and temporal keys. | No rooms or regions and no multi-goal commitment. G1 adds region identity plus subtask binding, not separate keys per se. |

Also read and not covering G1:
- LangNav/MLFM, [2507.07299](https://arxiv.org/html/2507.07299): goals disclosed one at a time; no rooms in the map.
- RAGNav, [2603.03745](https://arxiv.org/html/2603.03745): orders subtasks but no room binding and no instance handling.
- HM3D-M³, [2610.20121](https://arxiv.org/html/2610.20121): benchmark only; finds "declares goal from the wrong place".

**Is it still open?** Partly. Room-aware instance verification for *navigation* exists in STEGNav, EvoMemNav and Context-Nav. What none of them does is:
1. gate *commitment* by the room a compound sequential instruction assigns to the current subtask, using region coverage over time to decide "not yet observed, explore";
2. do this inside a *learned* memory-query policy like MemoryExplorer, whose training memories always contained the goal (Section 1), so it never learned to reject memory;
3. share the same region-indexed memory between navigation and the later memory QA.

The risk is real: a reviewer may see G1 as porting room labels and verification into MemoryExplorer. The claim must therefore rest on (1)–(3) and on the falsifying experiment below.

**Role of each memory part.**
- **3D:** region membership of each view and instance, from position and room segmentation. This decides "outside the subtask's room" and "subtask's room not yet observed". Essential.
- **Semantic:** room type of each region and category/attributes of each instance, to match "the painting in the bathroom". Essential.
- **Temporal:** binds subtask *k* to the *k*-th room in the instruction, records which rooms have been visited or covered so far, and which instances already served as earlier goals (the 10 task-progress failures). Essential for navigation.
- **For QA, the temporal part does little:** the QA prompt gives no subtask reference (`logger_lmeebench.py:494-501`), so only the 3D and semantic parts help there.

**Falsifying experiment.** An oracle region gate, run as inference only through the same runtime-wrapper approach as `analysis/instrumented_run.py`:
- At each navigation decision, hide memories and objects whose GT region is not the goal's region, so the agent must explore if none remain.
- For QA, restrict the candidate memories to views of the goal's region.
- If SR rises by **less than 5 points** and accuracy by less than 2 points, the gap is not worth a new mechanism; the bottleneck would then be perception and exploration (23% and 14.5% of navigation failures).
- **Cost:** about 2.5 hours on 4×24 GB GPUs for the 58 tasks, no training.
- A cheaper check (about the same cost, no oracle): add the subtask's room name from the task config to the navigation prompt and measure the change in distractor-choice rate.

### G2: Evidence sufficiency before answering memory QA

**Statement.** The agent answers every question even when no stored view contains the evidence. A memory that knows what it has and has not observed, and where, could trigger re-observation or abstention.

**LMEE evidence.**
- Evidence was never observed for 55 of 145 questions (37.9%); these account for 21 of 50 wrong answers (42%).
- Accuracy on them (61.8%) is at the always-"B" level (59.3%).

**Upper bound.** +14.5 points if all were answered correctly. But if those questions reached single-view accuracy, the expected gain is **+2.3 points**, because the letter prior already answers most of them. On the current QA metric the measured benefit would be small.

**Closest prior work:**

| Paper | What it does | Why the gap remains |
| --- | --- | --- |
| MemoryEQA, [2505.13948](https://arxiv.org/html/2505.13948) | Confidence-rated stopping before answering. | Single-question exploration, not post-hoc QA over a multi-goal episode memory. |
| EmbodiedMemory-Bench, [2609.28236](https://arxiv.org/html/2609.28236) | Shows 61.3% of failures start from acting on memory contradicted by history; proposes the EMem memory. | Interactive household tasks, not static-scene memory QA. |
| FindingDory, [2506.15635](https://arxiv.org/html/2506.15635) | Shows VLMs fail at memory-referenced instructions. | Navigation targets, not abstention. Also on the ruled-out list for past-interaction references. |
| VL-KnG, [2510.01483](https://arxiv.org/html/2510.01483) | Retrieves ranked frames from an object knowledge graph. | No sufficiency check. |

**Role of each memory part.** 3D: coverage of the region the question concerns. Semantic: whether the asked object or attribute was seen. Temporal: weak, because the question carries no time or subtask reference.

**Falsifier.** An oracle "evidence view" run: insert the verified GT view into memory for the 55 questions. If accuracy rises by less than 3 points, the metric cannot reward the mechanism. Cost: about 1 GPU-hour.

**Status.** The mechanism is plausible, but the LMEE QA metric is dominated by the answer-letter prior. **This is a benchmark issue more than a memory gap.**

### G3: Multi-view evidence aggregation for QA. Falsified on LMEE.

**Statement.** Answers needing evidence spread over several stored views fail when a single memory is chosen.

**LMEE evidence.** Only 2 of 145 questions (5 of 145 at stricter visibility); upper bound +1.4 points.

**Closest prior work:**
- Informative Viewpoint Selection, [2609.33288](https://arxiv.org/abs/2609.33288) (abstract only, via the literature sub-agent): selects a diverse set of views.
- MemTree3D, [2608.18009](https://arxiv.org/html/2608.18009): about 3 key frames chosen per location; no instance linking across locations.
- VL-KnG, [2510.01483](https://arxiv.org/html/2510.01483): persistent object identities; answers over multiple frames, text only.

**Why it remains open elsewhere but not here.** LMEE questions are generated from single frames, so the gap is real on OpenEQA-style data but not on LMEE.

**Role of each memory part.** 3D links views of the same instance; semantic does the matching; temporal plays no role.

**Falsifier.** Already run (diagnostic a). Further cost: none.

### G4: Retrieval recall (CLIP top-3). Falsified on LMEE.

**LMEE evidence.** Gold recall is 84.0% for QA and 91.1% for navigation. Retrieval misses are 3 navigation failures (+1.1 points) and 4 QA failures (+2.8 points). QA accuracy does not depend on gold retrieval (67.6% vs 69.2%).

**Closest prior work:**
- FRAME, [2609.08886](https://arxiv.org/html/2609.08886): learned per-attribute readouts; offline retrieval only.
- Memory Over Maps, [2603.20530](https://arxiv.org/html/2603.20530): SigLIP top-10 plus VLM re-ranking plus multi-view depth fusion; no QA.
- NavProbe, [2609.27526](https://arxiv.org/html/2609.27526): already on the ruled-out list for multi-round retrieval.

**Role of each memory part.** Semantic and 3D keys; temporal plays no role.

**Falsifier.** Done (diagnostic c).

### G5: Task-progress tracking across subtasks

**LMEE evidence.** 10 navigation failures (4.7%) re-select an earlier subtask's goal; upper bound +3.7 points.

**Closest prior work:**
- AGM, [2608.29537](https://arxiv.org/html/2608.29537): progress pointer advanced only on verified achievement; manipulation only.
- STEGNav, [2608.28279](https://arxiv.org/html/2608.28279): verified outcome anchors.
- NavProbe, [2609.27526](https://arxiv.org/html/2609.27526): task-state history.

**Why it is still open.** Only the navigation-side verified pointer is new for LMEE, and the measured share is small, so it is best folded into G1's temporal part.

**Role of each memory part.** Temporal is essential, 3D helps (the instance identity of earlier goals), semantic is minor.

**Falsifier.** Exclude GT earlier-goal instances from the candidate memories (cost as for G1). If the gain is under 2 points, drop it.

Not memory gaps but large:
- **Perception** (22.9% of navigation failures: the goal is visible but not detected or IoU-mapped, or a false-positive detection).
- **Frontier aborts** (11.7%).

These should be fixed or controlled before claiming memory gains.

## 5. Recommendation and risks

**Recommendation.** Pursue **G1, room-scoped commitment**. It is the only candidate that meets all four validity criteria with numbers behind it:

| Criterion | How G1 meets it |
| --- | --- |
| (1) Evidence on LMEE | 45.8% of navigation failures, labelled from GT and confirmed 12/12 by hand; upper bound +36.0 SR points |
| (2) Not already solved | Open in the narrowed form above. The prior navigation work verifies instances but does not gate a learned memory policy's commitment by the room bound to each subtask of a compound instruction, and none shares that memory with QA. |
| (3) Fits the mechanism | Needs all three parts for navigation, with clear roles (temporal is weak for QA) |
| (4) Falsifiable | Oracle region-gate experiment, about 2.5 GPU-hours on 4 GPUs, no training |

Evaluate it mainly on navigation SR/SPL. Treat QA accuracy as secondary until the letter prior is controlled, for example by reporting accuracy minus the best constant-letter baseline, or by balancing answer letters.

**Risks.**
1. **Novelty.** STEGNav, EvoMemNav and Context-Nav already use room context for instance verification. If the oracle gate gives most of its gain through simple verification-on-arrival, the contribution reduces to engineering. Run the oracle and a STEGNav-style verify-on-arrival ablation side by side.
2. **Confounds that cap any memory gain.** Perception (22.9%) and frontier aborts (11.7%) both limit what a memory fix can achieve. The NaN start-pose bug may contribute to the aborts (**UNVERIFIED**); fixing it changes the baseline, so report both.
3. **The re-run differs from the user's baseline run.** SR is 21.32 vs 23.90 and per-goal agreement is 84.9%. The non-determinism comes from the simulator and detection, since generation is greedy. Conclusions are based on shares, not small differences.
4. **Gold definition for QA.** 61 of 145 questions fall back to goal-instance gold because the question-view match was not verified or no question noun matched an instance. The QA numbers are therefore less precise than the navigation ones; the multi-view and retrieval conclusions hold on the verified subset too (2/84 multi-view).
5. **Only 58 tasks.** Full-set (166-task) runs exist under `evaluation/results/exp_eval_lmee_full*`, but they are unfinished. The taxonomy was not extended to them.
6. **Paper reading depth.** Method sections were read for all papers in the G1, G2 and G4 tables. For Informative Viewpoint Selection (2609.33288), Memory-R1 (arXiv ID 2508.19828, ID **UNVERIFIED**) and the MemCtrl, EvolvingNav, EvoNav-Bench and LiSoNav entries, only the abstract or listing was read, through the literature sub-agent.

## 6. Reproducibility

```bash
bash analysis/run_all.sh                 # all numbers from saved outputs (~1 min)
RERUN=1 GPU=0 bash analysis/run_all.sh   # also redo the GPU stages (instrumented re-run, GT question views)
```

**Scripts** (all under `analysis/`):

| Script | Purpose |
| --- | --- |
| `instrumented_run.py` + `runs/run_split.sh` + `clean_partial.py` | Re-run of the baseline with ground-truth logging; resumable, retries after out-of-memory errors |
| `gt_question_views.py` | Recovers the pose of each QA source image |
| `analyze.py` | Taxonomy and diagnostics a–e |
| `check_baseline_facts.py` | Numbers behind Section 1 |
| `check_train_memory.py` | Training-memory construction (needs pyarrow) |
| `render_cases.py` | Re-renders stored views for manual checks |

**Labels:** `analysis/labels/*.json`. **Outputs:** `analysis/out/` (`summary.json`, `nav_items.csv`, `qa_items.csv`, sensitivity files) and `analysis/runs/instr_choice_sub/` (records, logs, LMEE result files). LMEE evaluation code and the user's baseline result files were not modified.

# RCMem: region-scoped context memory

*Method as implemented in P3 (zero-shot, Qwen3-VL-8B base), updated 31 Oct 2026 with every P3 run. Code: `rcmem/`, `analysis/p3_nav.py`. Diagram version with figures: `docs/rcmem_method.html`.*

The agent keeps everything it has seen inside the VLM's own context, organised by room. Before it moves, it first picks a room (or decides to explore), and only then picks an object, looking at that room's views alone.

## The problem it targets

On LMEE, memory-based agents often commit to an object in the wrong room before they have seen the goal's room: 45.8% of MemoryExplorer's navigation failures are such commitments, and 87.5% of its same-category distractor choices lie in another region (`docs/gap_report.md`). Putting every view in context without structure makes it worse (F1: premature commits 28.9% → 40.6%), while restricting the context to the goal's room removes almost all of them (F2 oracle: 1.3%) (`docs/phase2b_decision.md`). RCMem tries to get that effect without ground truth: it builds rooms itself and makes the room choice an explicit first step.

## Overview

```mermaid
flowchart LR
  subgraph every captured view
    V[Captured views<br/>RGB-D + pose] -->|voxel pruning 0.05 m,<br/>keep if ≥25% new| K[Kept frames<br/>indexed · frame→pose list]
    V -->|all views| M[Floor map<br/>0.1 m: floor, walls]
    M -->|segment at doorways| R[Regions]
    K -->|pixels| P[One A1-band panorama<br/>per region]
    R -->|labels| P
  end
  subgraph each decision, before moving
    T[Task instruction] --> IP[Instruction parser<br/>&lt;subtask i: goal, room&gt;]
    IP -->|markers| S1{{Step 1<br/>Region k or Frontier i}}
    P --> S1
    FR[Frontier images] --> S1
    S1 -->|Region k| S2{{Step 2<br/>Frame f, Object o}}
    K -.->|frames of region k only| S2
    S1 -->|Frontier i| EX[Explore frontier i]
    S2 --> NAV[Navigate via frame→pose]
    NAV -.->|target position → later markers| IP
  end
```

With no region observed yet, step 1 is skipped and the agent goes to the nearest frontier without a VLM call (empty-context rule).

## Step by step

### 1. Parse the instruction into ordered (goal, room) pairs
*Once per subtask · `rcmem/instruction.py`*

An LMEE task is one compound sentence ("Start in the first bedroom to look at the painting, then head to the bathroom to check the sink, and finally go to the second bedroom…"). A rule-based parser splits it into clauses at sequence words (then, next, finally, followed by, sentence ends), finds room mentions from a vocabulary with synonyms (hallway → hall, study → office, …) and shared spaces ("dining room and living room area", "laundry room & kitchen"), and gives each goal the room of its clause, or the previous goal's room when its clause names none. Region instances: every mention opens a new instance, except a mention of the same room type without "another / other / second / next / last / different", which reuses the earlier instance ("back to the kitchen").

Scored against the 166 task configs (`analysis/out_p3/parser_eval.json`): goal located 100%, room type correct 97.0% of 824 subtasks, same/different-room grouping correct for 96.1% of subtask pairs, whole task exact 82.5%.

### 2. Keep a frame only if it shows new space
*Every captured view · `analysis/p2b_context.py` (`FramePruner`)*

Each view's depth is back-projected and quantised into 0.05 m voxels; the frame is kept if at least 25% of its voxels are new. Kept frames get an index in viewing order and stay in the context as images. The only state outside the context is the frame-to-pose list used to drive to a chosen frame: no features, never searched. Context grows with the size of the house, not the length of the episode (F0: 100% of decisions under 16k tokens).

### 3. Build a floor map from depth
*Every captured view · `rcmem/regions.py` (`RegionBuilder.add`)*

Every view (kept or not) adds its 3D points to a 0.1 m grid over the floor plane, centred on the episode start:

| Height above floor | Meaning |
|---|---|
| below 1.0 m | observed floor or furniture |
| 1.3–1.9 m | wall (≥ 2 points; dilated by one cell) |

Doorways stay open because their opening reaches above 1.9 m.

### 4. Split the map into regions at narrow passages
*Every decision · `rcmem/regions.py` (`RegionBuilder.segment`)*

1. Domain = observed cells that are not wall (closed once to fill sampling holes).
2. Erode the domain by `DOOR_R` = 0.4 m: a passage narrower than about 0.8 m (a doorway) disappears.
3. Each remaining component of at least 0.5 m² seeds a region.
4. A watershed on the domain's distance transform grows the seeds back, so borders fall on the narrowest cells.
5. Parts of the domain no seed reached (small closed rooms) become their own regions; regions under 1 m² merge into the neighbour they share the longest border with.

No learned model and no ground truth: a region is a connected part of the floor between narrow passages. Region ids change between decisions as the map grows, so the method identifies regions by their cells, never by id.

Accuracy against HM3D rooms (offline, `analysis/p3_regions.py`, 845 decisions): point accuracy 78.4%, agreement 84.4%; online in the RP run: 76.4% and 83.2%. The main error is merging: 30.8% of predicted regions join open-plan spaces that HM3D divides without a wall.

### 5. Assign every kept frame's pixels to regions
*Every decision · `analysis/p3_nav.py` (`frame_labels`, `observed`)*

Each pixel's 3D point falls in a map cell and takes that cell's region; points on walls or furniture take the nearest region cell within 1 m. A region is **observed** (shown in step 1) once some kept frame has at least 5% of its pixels in it, and its **step-2 frames** are those kept frames. Regions are listed in the order they were first observed.

### 6. Render one panorama per region
*Every decision · `rcmem/panorama.py` (OneCanvas's projection, `third_party/onecanvas` at 5c9c587)*

- **Centre:** the region's maximum-visibility cell (among its most open cell and a sample of cells ≥ 0.3 m from a border, the one that sees the most region cells along lines that cross no wall), at floor + 1.5 m.
- **Splat:** every kept frame's pixels of that region are projected onto a 1024 × 512 equirectangular image around the centre; the nearest point wins each pixel.
- **Holes:** parts not yet seen stay grey, so the image shows what has *not* been seen.
- **A1-band:** cropped to the rows holding the central 98% of observed pixels (full 360° width).

Panoramas are rebuilt at every decision because regions change; this takes about 0.3 s (mean hole fraction 0.6 of the full sphere).

### 7. Step 1: choose a room or explore
*Every decision, before moving · Qwen3-VL-8B, zero-shot*

The prompt holds the task, the subtask markers, one panorama per observed region, and one image per frontier:

```text
Task Instruction: The complete task is: … Now you need to perform the subtask of finding the painting, …
Subtasks of the instruction, in order:
<subtask 1: painting, bedroom> done; its target is in Region 0
<subtask 2: sink, bathroom> done; its target is in Region 2
<subtask 3: painting, bedroom> current; a different bedroom from subtask(s) 1 (not Region 0)
Regions: each image below is one room of the house as seen so far, rendered over 360 degrees from the room's
centre and cropped to the band of heights seen so far; grey areas have not been seen yet.
Region 0: [panorama]   Region 1: [panorama]   Region 2: [panorama]
The followings are all the Frontiers that you can explore:
Frontier 0: [image]    Frontier 1: [image]
THOUGHT: …
ANSWER: Region 1        ← or "ANSWER: Frontier i"
```

The markers carry the region binding: where earlier subtasks' committed targets lie now, and whether the current goal is in the same room as an earlier subtask or a different room of the same type. With no region observed, the agent goes to the nearest frontier without asking the model (empty-context rule). An invalid reply ends the subtask, as an invalid baseline reply does.

### 8. Step 2: choose the object inside that room only
*Only after "Region k" · the Phase 2b decision prompt (`analysis/p2b_context.py`)*

The model sees only region k's kept frames, each with the detector's numbered boxes (LMEE's YOLO-World + SAM, run by LMEE on every captured view), and answers `Frame f, Object o`, or a frontier if the target is not there after all. Objects in other rooms, including same-category distractors, are not in this prompt.

### 9. Navigate, then remember where the target was
*After a commit*

The chosen object becomes LMEE's navigation target (a one-object snapshot); the frame-to-pose list gives the viewpoint. The target's 3D position is stored per subtask and, at later decisions, mapped to whichever region now contains it, so later markers can say "its target is in Region k" even though region ids change. QA subtasks read all kept frames (as F1).

## Inputs and what the method does not use

| Input | Source | Used for |
|---|---|---|
| RGB and depth | Habitat simulator sensors (exact, metric, noise-free), the same LMEE and MemoryExplorer use | frames, pruning, floor map, panoramas |
| Camera pose | simulator | back-projection, frame→pose list |
| Object boxes | LMEE's own YOLO-World + SAM, unchanged | step 2 only |
| HM3D rooms and instance labels | ground truth | **only** the oracle run (RG) and scoring; never shown to the model in RP |

- **No memory bank:** no stored features, no retrieval index, no similarity search.
- **No room labels:** the model compares each panorama with the subtask's wording (labelled region markers, option B, collapsed to "not seen yet" on 7B models in P3-0).
- **No training yet:** P4 builds training streams where the goal is often absent, so that "explore" is sometimes the right answer; P5 fine-tunes (LoRA SFT, then GRPO on the decision).

## Variants under test (P3, seed 77)

| Variant | Change | Targets |
|---|---|---|
| **RG** | ground-truth HM3D rooms instead of predicted ones (oracle) | how much segmentation errors cost |
| **RPin** | step 2 boxes only objects whose 3D centre lies in region k | distractors leaking in from frames that mostly show another room (the 5% rule takes whole frames) |
| **RPpt** | step 2 sees plain frames and answers `Frame f, Point (x, y)` (0–1000 relative); the point is back-projected with the frame's depth and pose to the 3D target | detector misses: the goal seen but never boxed (26–34% of goal-visible decisions, `analysis/p3_glimpse.py`) |
| **RPcu** | step 1 shows a room once a kept frame has ≥ 1% of its pixels in it, and adds up to two full-resolution close-up crops per room after its panorama | small or brief sightings: the goal covers < 1 image token of its room's panorama in 37–45% of goal-visible decisions; small glimpses from another room are never usable with RP |
| **RP25** | RP on Qwen2.5-VL-7B-Instruct base | the backbone's share of the gain |

## Results so far (58-task subset, choice mode, seed 77, zero-shot)

| Run | Backbone | SR | SPL | Premature commit % | Step-1 exact room % | Step-2 accuracy % |
|---|---|---|---|---|---|---|
| **RP: RCMem, predicted regions** | Qwen3-VL-8B | **34.6** | 20.5 | 22.8 | 46.2 | 20.0 |
| RG: RCMem, ground-truth regions | Qwen3-VL-8B | 33.1 | 21.4 | 23.9 | 45.3 | 18.2 |
| F1: all kept frames, no regions | Qwen3-VL-8B | 31.2 | 19.2 | 29.8 | — | — |
| MemoryExplorer, fixed start (3 seeds) | Qwen2.5-VL-7B (fine-tuned) | 25.9 | 16.2 | 28.9 | — | — |
| RPin: RP + step-2 boxes only inside the chosen room | Qwen3-VL-8B | 35.7 | 20.7 | 16.9 | 43.8 | 18.6 |
| RPpt: RP + step 2 by pointing | Qwen3-VL-8B | 24.6 | 15.0 | 35.7 | 43.4 | 35.1 |
| **RPcu: RP + room close-ups + show threshold 1%** | Qwen3-VL-8B | **38.6** | **21.5** | 18.8 | **48.7** | 17.0 |
| RP on base Qwen2.5 | Qwen2.5-VL-7B-Instruct | 18.0 | 10.8 | 32.0 | 37.5 | 10.1 |
| RPcu on base Qwen2.5 | Qwen2.5-VL-7B-Instruct | 18.0 | 9.5 | 20.6 | 37.9 | 9.5 |

- RP beats MemoryExplorer by +8.7 SR and +4.3 SPL, but most of that comes from the backbone: F1 alone on Qwen3-VL-8B scores 31.2. RCMem's own gain over F1 is +3.3 SR with 7 points fewer premature commits.
- **Gate G2** (RP ≥ F1 + 5.1 = 36.4 SR) is borderline on one seed; by the user's decision (30 Oct) no extra seeds now, judged after P5.
- **RPcu is the best zero-shot setting** (+4.0 SR over RP, +7.4 over F1): close-ups and the lower show threshold raise step-1 accuracy and "not seen yet" recall. RPin removes most cross-room commits but doubles invalid replies; RPpt finds the goal more often in step 2 but commits far too early without the detector as a brake (P5 trains both RP and RPpt to test whether training fixes that).
- On base Qwen2.5-VL-7B, RCMem reaches only 18.0 SR (RP and RPcu alike), below MemoryExplorer's fine-tuned model: zero-shot RCMem depends on a strong reader, which is why training (P5) and the backbone controls (P6) matter.
- QA (LMEE's multiple-choice format): every Qwen3 run is at or below the answer-letter prior (MemoryExplorer: +8.7 over it), so no QA claim is made.
- Ground-truth regions do not beat predicted ones, so segmentation is not the bottleneck. The decision is: step 1 picks the right room about 45% of the time, and step 2 picks the goal only about 20% of the time when the room's frames show it.

Numbers come from `analysis/out_p3/p3_results.json` (`python analysis/p3_report.py`); the full report is `docs/p3_report.md`.

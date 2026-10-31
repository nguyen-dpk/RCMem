"""Figure 1, MemoryExplorer row: re-render the memories it retrieved at its last decision on the figure's subtask, from the
camera poses in its run's records, and draw its path on the navigable map. No VLM. Writes draw/assets/me_*.jpg, me_map.jpg.

Run from the repo root, lmee env: python paper/draw/render_fig1_me.py
Checks each render against the per-object pixel counts recorded in the run, so a wrong pose or camera fails loudly.
"""
import glob, json, os
import cv2, habitat_sim, numpy as np, quaternion

SUB, SCENE = "00877-4ok3usBNeis_0_2", "00877-4ok3usBNeis"
RUN, CHAIR, GOAL = "analysis/runs/p2_E0_fix_s77", 8, 201
OUT = "paper/draw/assets"
SIZE, HFOV = 1280, 120  # analysis/cfg/p3_pilot.yaml: img_width, img_height, hfov

sub, decs, objs = None, [], {}
for f in sorted(glob.glob(f"{RUN}/records_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("subtask") == SUB:
            if r["type"] == "decision":
                decs.append(r)
            elif r["type"] == "subtask":
                sub = r
        elif r["type"] == "scene" and r["scene_id"] == SCENE:
            objs = {o["sid"]: o for o in r["objects"]}
returned = decs[-1]["retrieval"][0]["returned"]
chosen = decs[-1]["choice"]["image"]
assert decs[-1]["choice"]["type"] == "memory" and chosen in returned

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_dataset_config_file = "data/hm3d_annotated_basis.scene_dataset_config.json"
sim_cfg.scene_id = f"data/hm3d-0.2/hm3d/val/{SCENE}/{SCENE.split('-')[1]}.basis.glb"
sensors = []
for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("sem", habitat_sim.SensorType.SEMANTIC)):
    s = habitat_sim.CameraSensorSpec()
    s.uuid, s.sensor_type, s.resolution, s.hfov, s.position = uuid, kind, [SIZE, SIZE], HFOV, [0.0, 0.0, 0.0]
    sensors.append(s)
agent_cfg = habitat_sim.agent.AgentConfiguration(sensor_specifications=sensors)
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def view(cam, fwd):
    z = -np.array(fwd) / np.linalg.norm(fwd)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    state = habitat_sim.AgentState()
    state.position = np.array(cam)
    state.rotation = quaternion.from_rotation_matrix(np.stack([x, np.cross(z, x), z], axis=1))
    sim.get_agent(0).set_state(state)
    o = sim.get_sensor_observations()
    return o["rgb"][..., :3][..., ::-1].copy(), o["sem"]


for i, name in enumerate(returned):
    meta = sub["frames"][name]
    bgr, sem = view(meta["cam"], meta["fwd"])
    ids, counts = np.unique(sem, return_counts=True)
    got = dict(zip(ids.tolist(), counts.tolist()))
    want = {int(k): v for k, v in meta["vis"].items()}
    top = sorted(want, key=want.get, reverse=True)[:5]
    err = max(abs(got.get(k, 0) - want[k]) / want[k] for k in top)
    print(name, "chosen" if name == chosen else "", "goal px", want.get(GOAL, 0), "chair px", want.get(CHAIR, 0), f"worst pixel-count error {err:.1%}")
    assert err < 0.05, name  # the render must reproduce what the run saw
    thumb = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
    if name == chosen:
        ys, xs = np.nonzero(sem == CHAIR)
        s = 256 / SIZE
        cv2.rectangle(thumb, (int(xs.min() * s), int(ys.min() * s)), (int(xs.max() * s), int(ys.max() * s)), (43, 57, 192), 3)
    cv2.imwrite(f"{OUT}/me_top{i}.jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])

for n in ("15-view_0.png", "15-view_1.png", "15-view_2.png"):
    print(n, "goal px", sub["frames"][n]["vis"].get(str(GOAL), 0), "of", SIZE * SIZE)
print("decision global steps", [d["global_step"] for d in decs], "chosen", chosen, "returned", returned)

# map: navigable floor, MemoryExplorer's path to the chair, the unseen goal
MPP = 0.04
y = decs[-1]["pts"][1]
PAD = 30  # the goal sits at the map's edge: keep its marker whole
floor = np.pad(sim.pathfinder.get_topdown_view(MPP, y), PAD)
lo = sim.pathfinder.get_bounds()[0]
px = lambda p: (int((p[0] - lo[0]) / MPP) + PAD, int((p[2] - lo[2]) / MPP) + PAD)
path = [d["pts"] for d in decs] + [sub["final_pts"]]
chair, goal = objs[CHAIR]["center"], objs[GOAL]["center"]
img = np.where(floor[..., None], np.array([236, 236, 236], np.uint8), np.array([255, 255, 255], np.uint8)).astype(np.uint8).copy(order="C")  # pathfinder arrays are not C-ordered; cv2 needs it
border = floor ^ (cv2.erode(floor.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
img[border] = (170, 170, 170)
RED, GREEN = (43, 57, 192), (73, 132, 30)
cv2.polylines(img, [np.array([px(p) for p in path], np.int32)], False, RED, 2, cv2.LINE_AA)
cv2.circle(img, px(path[0]), 6, (120, 120, 120), -1, cv2.LINE_AA)
cv2.drawMarker(img, px(chair), RED, cv2.MARKER_TILTED_CROSS, 22, 4, cv2.LINE_AA)
cv2.drawMarker(img, px(goal), GREEN, cv2.MARKER_STAR, 24, 3, cv2.LINE_AA)
pts = np.array([px(p) for p in path] + [px(chair), px(goal)])
m = 40
x0, y0 = np.maximum(pts.min(0) - m, 0)
x1, y1 = pts.max(0) + m
crop = img[y0:y1, x0:x1]
cv2.imwrite(f"{OUT}/me_map.jpg", cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("map", crop.shape, "chair at", (px(chair)[0] - x0, px(chair)[1] - y0), "goal at", (px(goal)[0] - x0, px(goal)[1] - y0))
sim.close()

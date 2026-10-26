"""Habitat -> OneCanvas conversion tested on one real scene (P3-0 completion check, row 1).

For every kept view of one cached episode: OneCanvas's own lifting (`lift_to_3d_with_intrinsics`, depth in mm,
OpenCV camera-to-world pose from rcmem.geometry.c2w_onecanvas) must (1) match our Habitat back-projection after the
inverse world map, and (2) put each HM3D instance's pixels inside that instance's ground-truth box. Then one region
panorama is built and every seen pixel is mapped back to 3D (pixel_to_point); the instance stored at that pixel must
contain the point. OneCanvas's feature-canvas geometry (`compute_scene_geometry`, used by A2) must place each patch
on the same ray as our panorama does.
Run (onecanvas env, repo root): python rcmem/test_onecanvas_conversion.py [run] [episode]
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, os.path.join(REPO, "third_party/onecanvas"))
import analyze as A  # noqa: E402
from geometry import lift_to_3d_with_intrinsics  # noqa: E402
from reprojection import compute_scene_geometry  # noqa: E402
from rcmem.geometry import backproject_habitat, c2w_habitat, c2w_onecanvas, intrinsics, to_habitat, to_onecanvas  # noqa: E402
from rcmem.panorama import RegionPanorama  # noqa: E402

CACHE = os.environ.get("P3_CACHE", "/ssd1/quang/embodied/outputs/p3_0_cache")
MARGIN = 0.05  # m around a GT box


def main(run="p2_E0_fix_s77", ep="00829-QaLdnwvtxbs_ep_0"):
    scenes, _, _ = A.load_records(os.path.join(REPO, "analysis/runs", run))
    objs = scenes[A.scene_of(ep)]
    z = np.load(os.path.join(CACHE, run, ep + ".npz"))
    n, H, W = z["depth_mm"].shape
    K = intrinsics(W, H)
    k_oc = torch.tensor([[K[0], K[1], K[2] - 0.5, K[3] - 0.5]], dtype=torch.float32)
    max_err, inside, total = 0.0, 0, 0
    for i in range(n):
        T = c2w_onecanvas(z["cam"][i], z["yaw"][i])
        p_oc, _ = lift_to_3d_with_intrinsics(torch.as_tensor(z["rgb"][i]), torch.as_tensor(z["depth_mm"][i].astype(np.float32)),
                                             torch.as_tensor(T, dtype=torch.float32), k_oc)
        p_oc = p_oc.double().numpy()
        p_h, ok = backproject_habitat(z["depth_mm"][i] / 1000.0, z["cam"][i], z["yaw"][i])
        ok &= np.isfinite(p_oc).all(1)
        max_err = max(max_err, float(np.abs(to_habitat(p_oc[ok]) - p_h[ok]).max()))
        s = z["sem"][i].reshape(-1).astype(np.int64)
        for k in np.unique(s[ok]):
            o = objs.get(int(k))
            if o is None or o["cat"] in A.STRUCT:
                continue
            m = ok & (s == k)
            inside += int(np.all(np.abs(to_habitat(p_oc[m]) - o["center"]) <= np.asarray(o["size"]) / 2 + MARGIN, 1).sum())
            total += int(m.sum())
    print(f"views {n}: OneCanvas lift vs Habitat back-projection, max |diff| = {max_err * 1000:.2f} mm")
    print(f"instance pixels inside their own GT box (+{MARGIN} m): {inside}/{total} = {inside / total:.4f}")
    assert max_err < 1e-3 and inside / total > 0.95

    # panorama round trip on the largest region of the episode
    reg = np.vectorize(lambda k: (objs.get(int(k)) or {}).get("region") or "")(z["sem"])
    r = max(set(reg.reshape(-1).tolist()) - {"", "_-1"}, key=lambda x: (reg == x).sum())
    pts = np.concatenate([backproject_habitat(z["depth_mm"][i] / 1000.0, z["cam"][i], z["yaw"][i])[0][(reg[i] == r).reshape(-1)]
                          for i in range(n)])
    centre = pts.mean(0)
    pano = RegionPanorama(to_onecanvas(centre), 1024)
    for i in range(n):
        pano.add(z["rgb"][i], z["depth_mm"][i], (reg[i] == r) & (z["depth_mm"][i] > 100), c2w_onecanvas(z["cam"][i], z["yaw"][i]), K,
                 inst=z["sem"][i])
    vs, us = np.nonzero(pano.seen().numpy())
    sel = np.random.default_rng(0).choice(len(vs), min(2000, len(vs)), replace=False)
    hit = tot = 0
    for v, u in zip(vs[sel], us[sel]):
        o = objs.get(int(pano.inst[v * pano.w + u]))
        if o is None or o["cat"] in A.STRUCT:
            continue
        p = to_habitat(pano.pixel_to_point(u, v))
        hit += bool(np.all(np.abs(p - o["center"]) <= np.asarray(o["size"]) / 2 + MARGIN))
        tot += 1
    print(f"region {r}: hole fraction {pano.hole_fraction():.3f}; pixel -> 3D lands in the pixel's GT box: {hit}/{tot} = {hit / tot:.4f}")
    assert hit / tot > 0.95

    # A2 canvas geometry: OneCanvas patch rays vs our panorama's angles for the same 3D points (one view)
    i = int(np.argmax([(reg[j] == r).sum() for j in range(n)]))
    d = torch.as_tensor(np.where(reg[i] == r, z["depth_mm"][i], 0).astype(np.float32))[None]
    g = compute_scene_geometry(d, torch.as_tensor(c2w_onecanvas(z["cam"][i], z["yaw"][i])[None], dtype=torch.float32),
                               torch.as_tensor(K[None], dtype=torch.float32), torch.tensor([[W, H]]), 14, 14,
                               center_override=torch.as_tensor(to_onecanvas(centre), dtype=torch.float32))
    vv, uu = np.divmod(g.valid_indices.numpy(), 14)
    # the same ray OneCanvas casts: patch centre (v + 0.5) * 32 px, depth read there ('nearest-exact'), Habitat camera
    yc, xc = (vv + 0.5) * H / 14, (uu + 0.5) * W / 14
    zz = z["depth_mm"][i][yc.astype(int), xc.astype(int)] / 1000.0
    pc = np.stack([(xc - K[2]) / K[0] * zz, -(yc - K[3]) / K[1] * zz, -zz], 1)
    T = c2w_habitat(z["cam"][i], z["yaw"][i])
    p = to_onecanvas(pc @ T[:3, :3].T + T[:3, 3] - centre)
    lon = np.arctan2(p[:, 0], p[:, 1])
    dlon = np.abs((g.longitude.numpy() - lon + np.pi) % (2 * np.pi) - np.pi)
    print(f"A2 canvas: {g.n_valid} patches of view {i}, max longitude difference to our projection {np.degrees(dlon.max()):.2f} deg")
    assert np.degrees(dlon.max()) < 0.01
    print("conversion test ok")


if __name__ == "__main__":
    main(*sys.argv[1:])

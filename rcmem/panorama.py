"""Region panoramas (P3-0 option A1): a region's posed RGB-D frames -> one equirectangular RGB panorama with holes.

Lifting and projection are OneCanvas's own code (third_party/onecanvas, commit 5c9c587):
`geometry.lift_to_3d_with_intrinsics` (depth in mm + camera-to-world pose -> 3D points) and
`geometry.compute_equirectangular_mapping` (3D points -> full-sphere 2:1 panorama pixels around a centre). This module
only adds the z-buffer (nearest point per pixel wins), grey holes, and the inverse map pixel -> 3D point. Poses and
points are in OneCanvas conventions (Z-up, OpenCV cameras); rcmem.geometry converts from Habitat.
Longitude 0 (panorama centre column) looks along OneCanvas +Y = Habitat -Z; the top row is the zenith.
Run `python rcmem/panorama.py` (onecanvas env) for the self-check.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party/onecanvas"))
from geometry import compute_equirectangular_mapping, lift_to_3d_with_intrinsics  # noqa: E402

GREY = 128


class RegionPanorama:
    def __init__(self, centre_oc, width=1024, device="cpu"):
        self.c = torch.as_tensor(np.asarray(centre_oc), dtype=torch.float32, device=device)
        self.w, self.h, self.dev = width, width // 2, device
        self.rng = torch.full((self.h * self.w,), float("inf"), device=device)
        self.rgb = torch.full((self.h * self.w, 3), GREY, dtype=torch.uint8, device=device)
        self.inst = torch.full((self.h * self.w,), -1, dtype=torch.int64, device=device)

    def add(self, rgb, depth_mm, mask, c2w, K, inst=None):
        """Splat one frame's pixels where `mask` (H, W) is true. K = (fx, fy, cx, cy) with pixel centres at u + 0.5."""
        d = torch.as_tensor(np.where(mask, depth_mm, 0).astype(np.float32), device=self.dev)
        # OneCanvas's lift puts pixel u at x = (u - cx) / fx: shift cx, cy by half a pixel to keep pixel centres
        k = torch.tensor([[K[0], K[1], K[2] - 0.5, K[3] - 0.5]], dtype=torch.float32)
        pts, _ = lift_to_3d_with_intrinsics(torch.as_tensor(rgb, device=self.dev), d,
                                            torch.as_tensor(c2w, dtype=torch.float32, device=self.dev), k)
        ok = torch.isfinite(pts).all(1)
        idx, valid, r = compute_equirectangular_mapping(torch.nan_to_num(pts), self.w, self.h, self.c)
        ok &= valid
        if not ok.any():
            return
        idx, r = idx[ok].long(), r[ok]
        col = torch.as_tensor(rgb, device=self.dev).reshape(-1, 3)[ok]
        ins = (torch.as_tensor(inst.astype(np.int64), device=self.dev).reshape(-1)[ok] if inst is not None
               else torch.full_like(idx, -1))
        best = torch.full((self.h * self.w,), float("inf"), device=self.dev).scatter_reduce(0, idx, r, "amin")
        win = (r == best[idx]) & (r < self.rng[idx])  # nearest point of this frame, and nearer than what is there
        idx = idx[win]
        self.rng[idx], self.rgb[idx], self.inst[idx] = r[win], col[win], ins[win]

    def seen(self):
        return torch.isfinite(self.rng).reshape(self.h, self.w)

    def image(self):
        """(H, W, 3) uint8; a hole with >= 5 of its 8 neighbours seen takes their mean (closes splatting gaps)."""
        img = self.rgb.reshape(self.h, self.w, 3).float()
        s = self.seen().float()
        ps, pi = torch.nn.functional.pad(s, (1, 1, 1, 1)), torch.nn.functional.pad(img * s[..., None], (0, 0, 1, 1, 1, 1))
        nb = sum(ps[1 + dy:self.h + 1 + dy, 1 + dx:self.w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        acc = sum(pi[1 + dy:self.h + 1 + dy, 1 + dx:self.w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        fill = (s == 0) & (nb >= 5)
        img[fill] = acc[fill] / nb[fill][:, None]
        return img.round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    def band_image(self, trim=0.01, min_h=56):
        """image() cropped to the latitude band actually observed (A1-band): the rows holding the central 1 - 2 * trim
        of the seen pixels (stray splats near the zenith or nadir would otherwise stretch it), at least min_h rows."""
        img, cnt = self.image(), self.seen().sum(1).cpu().numpy()
        if not cnt.any():
            return img
        c = np.cumsum(cnt) / cnt.sum()
        a, b = int(np.searchsorted(c, trim, side="right")), int(np.searchsorted(c, 1 - trim)) + 1
        a = max(0, min(a - max(0, min_h - (b - a)) // 2, self.h - min_h))
        return img[a:max(b, a + min_h)]

    def hole_fraction(self):
        return float(1 - self.seen().float().mean())

    def instance_pixels(self, ids):
        return int(torch.isin(self.inst, torch.as_tensor(list(ids), device=self.dev)).sum())

    def pixel_to_point(self, u, v):
        """Panorama pixel -> 3D point (OneCanvas world), None on a hole. Inverts compute_equirectangular_mapping."""
        r = float(self.rng[int(v) * self.w + int(u)])
        if not np.isfinite(r):
            return None
        lon = (u + 0.5) / self.w * 2 * np.pi - np.pi
        lat = (v + 0.5) / self.h * np.pi - np.pi / 2
        # canvas axes: x_c = X, y_c = -Z (down), z_c = Y (forward); lon = atan2(x_c, z_c), lat = atan2(y_c, horiz)
        x_c, y_c, z_c = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
        return self.c.cpu().numpy() + r * np.array([x_c, z_c, -y_c])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rcmem.geometry import backproject_habitat, c2w_onecanvas, intrinsics, to_habitat, to_onecanvas
    # a Habitat camera 3 m from a wall at z = -3 (Habitat), seen from a canvas centre 0.5 m off the camera
    depth = np.full((64, 64), 3.0)
    cam, yaw = np.array([0.0, 1.5, 0.0]), 0.0
    K, T = intrinsics(64, 64), c2w_onecanvas(cam, yaw, tilt=0.0)
    pts_h, _ = backproject_habitat(depth, cam, yaw, tilt=0.0)
    pano = RegionPanorama(to_onecanvas([0.5, 1.5, 0.2]), width=512)
    rgb = np.full((64, 64, 3), 200, np.uint8)
    pano.add(rgb, depth * 1000, np.ones((64, 64), bool), T, K, inst=np.full((64, 64), 7))
    # OneCanvas lifting through our pose must reproduce our own Habitat back-projection
    k = torch.tensor([[K[0], K[1], K[2] - 0.5, K[3] - 0.5]])
    p_oc, _ = lift_to_3d_with_intrinsics(torch.as_tensor(rgb), torch.as_tensor(depth * 1000, dtype=torch.float32),
                                         torch.as_tensor(T, dtype=torch.float32), k)
    assert np.allclose(to_habitat(p_oc.numpy()), pts_h, atol=1e-4), "Habitat -> OneCanvas conversion"
    vv, uu = np.argwhere(pano.seen().numpy())[len(np.argwhere(pano.seen().numpy())) // 2]
    p = to_habitat(pano.pixel_to_point(uu, vv))
    assert abs(p[2] + 3.0) < 0.05, p  # back on the wall
    # a wall straight ahead (Habitat -z) sits at the centre column: longitude 0
    assert abs(np.argwhere(pano.seen().numpy())[:, 1].mean() - 256) < 30
    # the camera is level and at canvas height: the wall straddles the horizon row
    assert abs(np.argwhere(pano.seen().numpy())[:, 0].mean() - 128) < 5
    before = pano.seen().reshape(-1).clone()
    pano.add(rgb // 2, np.full((64, 64), 5000.0), np.ones((64, 64), bool), T, K)  # a farther wall must not win
    assert (pano.rgb[before] == 200).all()
    assert pano.instance_pixels({7}) > 0 and 0 < pano.hole_fraction() < 1 and pano.image().shape == (256, 512, 3)
    band, cnt = pano.band_image(), pano.seen().numpy().sum(1)
    rows = np.nonzero(cnt)[0]
    assert band.shape[1] == 512 and 56 <= band.shape[0] <= max(56, rows.max() + 1 - rows.min())
    assert (band != GREY).any(2).sum() >= 0.97 * cnt.sum()  # at least the central 98% of seen pixels are kept
    print("panorama self-check ok")

"""Camera geometry for LMEE views and the Habitat -> OneCanvas coordinate conversion (numpy only, py3.9+).

Habitat: Y-up world; OpenGL cameras (x right, y up, -z forward); the agent turns by yaw about +y and the sensor is
tilted by TILT about its x axis; depth is z-depth in metres.
OneCanvas: Z-up world; OpenCV cameras (x right, y down, z forward); camera-to-world poses; depth in millimetres.
World map: (x, y, z)_habitat -> (x, -z, y)_onecanvas, a proper rotation (det +1), so handedness is kept.
Run `python rcmem/geometry.py` for the self-check.
"""
import numpy as np

HFOV = 120.0                  # degrees, evaluation camera
TILT = np.deg2rad(-30.0)      # camera pitch
H2O = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # habitat world -> onecanvas world
GL2CV = np.diag([1.0, -1.0, -1.0])                         # OpenGL camera axes -> OpenCV camera axes


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def focal(w, hfov_deg=HFOV):
    return (w / 2) / np.tan(np.deg2rad(hfov_deg) / 2)


def intrinsics(w, h, hfov_deg=HFOV):
    """(fx, fy, cx, cy) with pixel centres at u + 0.5 (square pixels)."""
    f = focal(w, hfov_deg)
    return np.array([f, f, w / 2, h / 2])


def c2w_habitat(cam, yaw, tilt=TILT):
    """4x4 camera-to-world, Habitat world and OpenGL camera axes."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot_y(yaw) @ rot_x(tilt), cam
    return T


def c2w_onecanvas(cam, yaw, tilt=TILT):
    """4x4 camera-to-world in OneCanvas conventions (Z-up world, OpenCV camera)."""
    T = np.eye(4)
    T[:3, :3] = H2O @ rot_y(yaw) @ rot_x(tilt) @ GL2CV
    T[:3, 3] = H2O @ np.asarray(cam, dtype=np.float64)
    return T


def to_onecanvas(p):
    return np.asarray(p, dtype=np.float64) @ H2O.T


def to_habitat(p):
    return np.asarray(p, dtype=np.float64) @ H2O  # H2O is orthonormal: inverse = transpose


def backproject_habitat(depth, cam, yaw, hfov_deg=HFOV, tilt=TILT):
    """depth (H, W) z-depth in metres -> (H*W, 3) Habitat world points and a validity mask."""
    h, w = depth.shape
    f = focal(w, hfov_deg)
    v, u = np.mgrid[0:h, 0:w]
    z = depth.reshape(-1).astype(np.float64)
    pc = np.stack([(u.reshape(-1) + 0.5 - w / 2) / f * z, -(v.reshape(-1) + 0.5 - h / 2) / f * z, -z], 1)
    T = c2w_habitat(cam, yaw, tilt)
    return pc @ T[:3, :3].T + T[:3, 3], (z > 0.1) & (z < 10.0)


def voxel_keys(pts, voxel=0.05):
    q = np.floor(pts / voxel).astype(np.int64) + (1 << 20)  # 21 bits per axis packed into one int64
    return set(np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]).tolist())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 6.0, (32, 32))
    cam, yaw = np.array([1.0, 1.5, -2.0]), 0.7
    p_hab, _ = backproject_habitat(depth, cam, yaw)
    # same points through the OneCanvas pose: OpenCV camera ray (x right, y down, z forward) * depth
    K = intrinsics(32, 32)
    v, u = np.mgrid[0:32, 0:32]
    z = depth.reshape(-1)
    pc = np.stack([(u.reshape(-1) + 0.5 - K[2]) / K[0] * z, (v.reshape(-1) + 0.5 - K[3]) / K[1] * z, z], 1)
    T = c2w_onecanvas(cam, yaw)
    assert np.allclose(to_habitat(pc @ T[:3, :3].T + T[:3, 3]), p_hab), "OneCanvas pose must give the same 3D points"
    assert np.isclose(np.linalg.det(H2O), 1.0) and np.allclose(to_onecanvas([0, 1, 0]), [0, 0, 1]), "Z-up"
    fwd = c2w_habitat(cam, yaw)[:3, :3] @ [0, 0, -1]
    assert np.isclose(np.arctan2(-fwd[0], -fwd[2]), yaw) and np.isclose(fwd[1], np.sin(TILT)), "yaw/tilt convention"
    print("geometry self-check ok")

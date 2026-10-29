"""Region builder (P3): segment the observed floor plan into regions at narrow passages (doorways indoors), assign 3D
points to regions, and place each region's panorama centre at its point of maximum visibility. Depth + pose only; no
ground truth (the scoring helpers at the bottom take GT labels for diagnostics).

Map: a 2D grid of CELL m over the Habitat x-z plane, centred on the episode start. Each frame's points, by height
above the floor:
  < LOW_HI              observed floor or low furniture -> observed cell
  WALL_LO .. WALL_HI    wall (tall structure; doorways stay open, since a door's opening reaches above WALL_HI)
Domain = observed cells that are not wall (walls dilated by one cell to close sampling gaps).
Segmentation: erode the domain by DOOR_R (a passage narrower than about 2 * DOOR_R vanishes); each connected component
of at least MIN_SEED_M2 seeds a region; a watershed on the domain's distance transform grows the seeds back over the
domain, so region borders fall on the narrowest cells between seeds. Regions under MIN_REGION_M2 are merged into the
neighbour they share the longest border with. Region ids are 1..K and change between calls; callers identify regions
by their cells, never by id.
Run `python rcmem/regions.py` for the self-check (two rooms joined by a doorway, plus an L-shaped room).
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

CELL = 0.1          # m, as LMEE's planner grid (tsdf_grid_size)
SIZE = 1000         # cells per side: 100 m
LOW_HI = 1.0        # m above the floor: floor and furniture points mark a cell observed
WALL_LO, WALL_HI = 1.3, 1.9
WALL_MIN = 2        # points for a wall cell
DOOR_R = 0.4        # m, erosion radius that closes a 0.9 m doorway (self-check); offline agreement ties 0.3 (analysis/p3_regions.py)
H_MAX = 0.0         # m; > 0: seeds are distance-transform peaks of this dynamic instead of eroded components
MIN_SEED_M2 = 0.5
MIN_REGION_M2 = 1.0
EYE = 1.5           # m above the floor: panorama centres at camera height


def disk(r_cells):
    y, x = np.mgrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
    return x * x + y * y <= r_cells * r_cells


class RegionBuilder:
    def __init__(self, origin, floor_y, door_r=DOOR_R, h_max=H_MAX, cell=CELL, size=SIZE):
        """origin: a Habitat point (x, y, z) on the floor, e.g. the agent's start; floor_y: the floor's Habitat y."""
        self.cell, self.size, self.floor, self.door_r, self.h_max = cell, size, float(floor_y), door_r, h_max
        self.x0, self.z0 = origin[0] - size * cell / 2, origin[2] - size * cell / 2
        self.obs_n = np.zeros(size * size, np.int32)
        self.wall_n = np.zeros(size * size, np.int32)
        self.labels, self.dt = None, None

    # ------------------------------------------------------------------ map
    def cells(self, pts):
        """(N, 3) Habitat points -> flat cell indices (-1 outside the grid)."""
        i = np.floor((pts[:, 2] - self.z0) / self.cell).astype(np.int64)
        j = np.floor((pts[:, 0] - self.x0) / self.cell).astype(np.int64)
        ok = (i >= 0) & (i < self.size) & (j >= 0) & (j < self.size)
        return np.where(ok, i * self.size + j, -1)

    def add(self, pts):
        """Add one frame's valid world points (N, 3)."""
        h, c = pts[:, 1] - self.floor, self.cells(pts)
        ok = c >= 0
        self.obs_n += np.bincount(c[ok & (h > -0.3) & (h < LOW_HI)], minlength=self.obs_n.size).astype(np.int32)
        self.wall_n += np.bincount(c[ok & (h >= WALL_LO) & (h <= WALL_HI)], minlength=self.wall_n.size).astype(np.int32)
        self.labels = None

    # ------------------------------------------------------------------ segmentation
    def segment(self):
        """-> (size, size) int32 labels, 0 = not a region cell. Cached until the next add()."""
        if self.labels is not None:
            return self.labels
        S = self.size
        lab = np.zeros((S, S), np.int32)
        self.dt = np.zeros((S, S), np.float32)
        obs = (self.obs_n > 0).reshape(S, S)
        wall = ndimage.binary_dilation((self.wall_n >= WALL_MIN).reshape(S, S))
        rows, cols = np.nonzero(obs | wall)
        if len(rows) == 0:
            self.labels = lab
            return lab
        m = int(np.ceil(self.door_r / self.cell)) + 2  # work on the bounding box of what was seen
        a, b = max(0, rows.min() - m), min(S, rows.max() + m + 1)
        c, d = max(0, cols.min() - m), min(S, cols.max() + m + 1)
        dom = ndimage.binary_closing(obs[a:b, c:d], iterations=1) & ~wall[a:b, c:d]  # close sampling holes in the floor
        dt = ndimage.distance_transform_edt(dom) * self.cell
        if self.h_max:  # seeds: DT peaks standing >= h_max above the saddle that joins them to a higher peak
            seeds, n = ndimage.label(h_maxima(dt, self.h_max) & (dt >= self.door_r))
        else:
            seeds, n = ndimage.label(ndimage.binary_erosion(dom, disk(int(round(self.door_r / self.cell)))))
        if n:
            area = np.bincount(seeds.ravel(), minlength=n + 1) * self.cell ** 2
            small = area < (0 if self.h_max else MIN_SEED_M2)
            small[0] = False
            seeds[small[seeds]] = 0
            seeds, n = ndimage.label(seeds > 0)
        ws = watershed(-dt, seeds, mask=dom).astype(np.int32) if n else np.zeros(dom.shape, np.int32)
        rest, m_rest = ndimage.label(dom & (ws == 0))  # parts of the domain no seed reached (small closed rooms)
        ws[rest > 0] = rest[rest > 0] + ws.max()
        if ws.any():
            ws = self._merge_small(ws)
            lab[a:b, c:d] = ws
        self.dt[a:b, c:d] = dt
        self.labels = lab
        return lab

    def _merge_small(self, ws):
        while True:
            ids, area = np.unique(ws[ws > 0], return_counts=True)
            small = ids[area * self.cell ** 2 < MIN_REGION_M2]
            if len(ids) <= 1 or not len(small):
                break
            k = small[np.argmin(area[np.isin(ids, small)])]
            ring = ndimage.binary_dilation(ws == k) & (ws > 0) & (ws != k)
            if not ring.any():  # isolated: drop it
                ws[ws == k] = 0
                continue
            ws[ws == k] = np.bincount(ws[ring]).argmax()
        ids = np.unique(ws[ws > 0])  # relabel 1..K
        lut = np.zeros(ws.max() + 1, np.int32)
        lut[ids] = np.arange(1, len(ids) + 1)
        return lut[ws]

    # ------------------------------------------------------------------ points -> regions
    def assign(self, pts):
        """(N, 3) Habitat points -> region label per point (0 = none): the region of the point's cell, or of the
        nearest region cell within 1 m (points on walls and furniture)."""
        lab = self.segment()
        if not hasattr(self, "_near") or self._near[0] is not lab:
            if lab.any():
                idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
                near = lab[idx[0], idx[1]]
                dist = np.hypot(idx[0] - np.arange(self.size)[:, None], idx[1] - np.arange(self.size)[None, :])
                near[dist * self.cell > 1.0] = 0
            else:
                near = lab
            self._near = (lab, near.ravel())
        return self.assign_cells(self.cells(pts))

    def assign_cells(self, c):
        """Flat cell indices (from cells(), -1 = outside) -> region label (0 = none); see assign()."""
        if not hasattr(self, "_near") or self._near[0] is not self.segment():
            self.assign(np.zeros((0, 3)))
        return np.where(c >= 0, self._near[1][np.maximum(c, 0)], 0)

    # ------------------------------------------------------------------ centres
    def centre(self, k, n_cand=40, n_tgt=300, n_samp=24, seed=0):
        """Maximum-visibility point of region k (Habitat x, floor + EYE, z): among candidate cells (the most open cell
        plus a sample of cells at least 0.3 m from a border), the one that sees the most region cells along
        straight lines not crossing a wall or leaving the observed area."""
        lab = self.segment()
        cells = np.argwhere(lab == k)
        if not len(cells):
            return None
        rng = np.random.default_rng(seed)
        dtk = self.dt[cells[:, 0], cells[:, 1]]
        inner = cells[dtk >= 0.3] if (dtk >= 0.3).any() else cells
        cand = np.vstack([cells[np.argmax(dtk)][None], inner[rng.choice(len(inner), min(n_cand, len(inner)), replace=False)]])
        tgt = cells[rng.choice(len(cells), min(n_tgt, len(cells)), replace=False)]
        t = np.linspace(0, 1, n_samp)[None, None, :, None]
        seg = cand[:, None, None, :] + t * (tgt[None, :, None, :] - cand[:, None, None, :])  # (C, T, S, 2)
        si = np.clip(np.round(seg).astype(int), 0, self.size - 1)
        free = lab[si[..., 0], si[..., 1]] > 0  # inside some region: not a wall, not unobserved
        vis = free.all(-1).sum(1)
        best = cand[np.lexsort((-self.dt[cand[:, 0], cand[:, 1]], -vis))[0]]
        return np.array([self.x0 + (best[1] + 0.5) * self.cell, self.floor + EYE, self.z0 + (best[0] + 0.5) * self.cell])

    def cell_point(self, i, j):
        return np.array([self.x0 + (j + 0.5) * self.cell, self.floor, self.z0 + (i + 0.5) * self.cell])


# ---------------------------------------------------------------------- scoring against ground truth (diagnostics)
MERGE_SHARE = 0.2   # a predicted region merges GT regions when >= 2 of them each hold this share of its points


def score(pred, gt, min_pts=200):
    """pred: int region per point (0 = none); gt: GT region id per point ('' = none). Points with both labels count.
    point_acc: share of points whose GT region is the majority GT region of their predicted region (purity);
    coverage: share of points whose predicted region is the majority predicted region of their GT region;
    agreement: harmonic mean of the two; merged: predicted regions (>= min_pts points) where >= 2 GT regions each hold
    >= MERGE_SHARE of the points; split: the same for GT regions over predicted ones."""
    pred, gt = np.asarray(pred), np.asarray(gt, dtype=object)
    ok = (pred > 0) & (gt != "")
    p, g = pred[ok], gt[ok]
    if not len(p):
        return None
    gi = {x: i for i, x in enumerate(sorted(set(g.tolist())))}
    g = np.array([gi[x] for x in g])
    M = np.zeros((p.max() + 1, len(gi)), np.int64)
    np.add.at(M, (p, g), 1)
    rows, cols = M.sum(1), M.sum(0)
    purity = M.max(1).sum() / M.sum()
    coverage = M.max(0).sum() / M.sum()
    big_p, big_g = rows >= min_pts, cols >= min_pts
    merged = int(((M >= MERGE_SHARE * rows[:, None]) & (M > 0)).sum(1)[big_p].__ge__(2).sum())
    split = int(((M >= MERGE_SHARE * cols[None, :]) & (M > 0)).sum(0)[big_g].__ge__(2).sum())
    return {"point_acc": float(purity), "coverage": float(coverage),
            "agreement": float(2 * purity * coverage / (purity + coverage)),
            "n_pred": int(big_p.sum()), "n_gt": int(big_g.sum()), "merged": merged, "split": split}


if __name__ == "__main__":
    # synthetic plan (Habitat x-z, metres): room A [0,4]x[0,4], room B [4.1,8]x[0,4] joined by a 0.9 m doorway in the
    # wall x = 4..4.1; room C an L joined to A by a 0.9 m doorway in the wall z = 4..4.1
    rng = np.random.default_rng(0)

    def wall_pts(x0, x1, z0, z1, gap=None):
        xs, zs = np.meshgrid(np.arange(x0, x1, 0.03), np.arange(z0, z1, 0.03))
        p = np.stack([xs.ravel(), zs.ravel()], 1)
        if gap is not None:
            p = p[~((p[:, gap[0]] >= gap[1]) & (p[:, gap[0]] <= gap[2]))]
        ys = rng.uniform(0.0, 2.4, len(p))
        return np.stack([p[:, 0], ys, p[:, 1]], 1)

    def floor_pts(x0, x1, z0, z1):
        xs, zs = np.meshgrid(np.arange(x0, x1, 0.05), np.arange(z0, z1, 0.05))
        return np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], 1)

    walls = np.vstack([wall_pts(-0.1, 8.1, -0.1, 0.0), wall_pts(-0.1, 0.0, -0.1, 9.1), wall_pts(8.0, 8.1, -0.1, 4.1),
                       wall_pts(4.0, 4.1, 0.0, 4.0, gap=(1, 1.5, 2.4)),           # A|B doorway z in [1.5, 2.4]
                       wall_pts(0.0, 8.1, 4.0, 4.1, gap=(0, 1.0, 1.9)),           # A|C doorway x in [1.0, 1.9]
                       wall_pts(3.0, 3.1, 4.1, 9.1), wall_pts(-0.1, 3.1, 9.0, 9.1)])
    floor = np.vstack([floor_pts(0, 4, 0, 4), floor_pts(4.1, 8, 0, 4), floor_pts(0, 3, 4.1, 9),
                       floor_pts(4.0, 4.1, 1.5, 2.4), floor_pts(1.0, 1.9, 4.0, 4.1)])
    rb = RegionBuilder(origin=(2.0, 0.0, 2.0), floor_y=0.0)
    rb.add(np.vstack([walls, floor]))
    lab = rb.segment()
    probe = np.array([[2, 0.5, 2], [6, 0.5, 2], [1.5, 0.5, 7], [2.5, 0.5, 4.5]])
    r = rb.assign(probe)
    assert len(set(r[:3].tolist())) == 3 and 0 not in r[:3], ("three rooms", r, np.unique(lab))
    assert r[3] == r[2], "C's corner cell belongs to C"
    assert lab.max() == 3, np.unique(lab)
    # score: a perfect prediction, then one that merges A and B
    gt = np.array(["A", "B", "C", "C"] * 100, dtype=object)
    s = score(np.array([1, 2, 3, 3] * 100), gt, min_pts=50)
    assert s["point_acc"] == 1 and s["merged"] == 0, s
    s = score(np.array([1, 1, 3, 3] * 100), gt, min_pts=50)
    assert s["merged"] == 1 and s["point_acc"] == 0.75, s
    # centre of room B lies inside B, away from the walls
    cB = rb.centre(r[1])
    assert 4.4 < cB[0] < 7.7 and 0.3 < cB[2] < 3.7 and abs(cB[1] - EYE) < 1e-9, cB
    print("regions self-check ok", np.unique(lab).tolist(), np.round(cB, 2).tolist())

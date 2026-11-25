"""Inside/outside (occupancy) labels for messy, non-watertight meshes.

Objaverse meshes are usually NOT watertight (open seats, single-sided planes, gaps), so
"is this point inside?" is not directly defined. Our method (same idea as mesh2sdf / ManifoldPlus):

  1. udf(p) = distance from p to the surface (KD-tree over dense surface samples)
  2. "wall" voxels = voxels whose centre has udf < LEVEL  (LEVEL ~ 0.9 voxel, so walls have no gaps)
  3. flood-fill all non-wall voxels from the grid border  -> EXTERIOR
  4. a query point p is OUTSIDE if  udf(p) >= LEVEL  and  p is on the exterior side,
     INSIDE otherwise.

The result is the watertight "outer skin" of the object pushed out by LEVEL (~0.4% of object size
at RES=256). Thin surfaces (a chair seat made of one plane) become plates of thickness 2*LEVEL.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

BOUND = 0.55  # grid covers [-BOUND, BOUND]^3; normalised object lives in [-0.5, 0.5]^3


class Occupancy:
    def __init__(self, surface_points, res=256, level_voxels=0.9):
        self.res = res
        self.voxel = 2 * BOUND / res
        self.level = level_voxels * self.voxel
        self.tree = cKDTree(surface_points)

        # candidate wall voxels: those containing samples, dilated by 1 (superset of udf < level)
        cand = np.zeros((res, res, res), dtype=bool)
        i = self.to_index(surface_points)
        cand[i[:, 0], i[:, 1], i[:, 2]] = True
        cand = ndimage.binary_dilation(cand, iterations=1)

        # exact wall test only on candidates (cheap)
        ci = np.argwhere(cand)
        d, _ = self.tree.query(self.centres(ci), workers=-1)
        wall = np.zeros_like(cand)
        w = ci[d < self.level]
        wall[w[:, 0], w[:, 1], w[:, 2]] = True

        labels, _ = ndimage.label(~wall)
        border = np.unique(np.concatenate([
            labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
            labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
        self.exterior = np.isin(labels, border[border > 0])

    def to_index(self, p):
        idx = np.floor((p + BOUND) / self.voxel).astype(np.int64)
        return np.clip(idx, 0, self.res - 1)

    def centres(self, idx):
        return (idx + 0.5) * self.voxel - BOUND

    def _is_ext(self, p):
        i = self.to_index(p)
        return self.exterior[i[:, 0], i[:, 1], i[:, 2]]

    def query(self, points):
        # we only need exact distances near the surface -> bounded search is ~50x faster
        d, nn = self.tree.query(points, workers=-1, distance_upper_bound=3 * self.voxel)
        inside = d < self.level                                   # inside the skin
        far = ~inside
        ext = self._is_ext(points)
        # points with udf >= level whose own voxel is a wall voxel: step 1.5 voxels away from
        # the nearest surface point and look whether we land in the exterior
        amb = far & ~ext & (nn < len(self.tree.data))
        if amb.any():
            p = points[amb]
            away = p - self.tree.data[nn[amb]]
            away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-12
            ext[amb] = self._is_ext(p + 1.5 * self.voxel * away)
        inside |= far & ~ext
        return inside

    def to_mesh(self, res=128):
        """Marching cubes of the occupancy field (for visual checking only)."""
        import trimesh
        from skimage import measure

        lin = (np.arange(res) + 0.5) / res * 2 * BOUND - BOUND
        g = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        vol = self.query(g).reshape(res, res, res).astype(np.float32)
        verts, faces, _, _ = measure.marching_cubes(np.pad(vol, 1), 0.5)
        verts = (verts - 1 + 0.5) / res * 2 * BOUND - BOUND
        return trimesh.Trimesh(verts, faces[:, ::-1], process=False)

"""Load an Objaverse GLB, normalise it, and sample coloured surface points.

Coordinate frame used everywhere in GEN3D: glTF frame (Y-up, as trimesh loads it),
normalised so the object's bounding-box centre is at the origin and its longest side = 1
(object fits in [-0.5, 0.5]^3).
"""`
import numpy as np
import trimesh


def load_parts(path):
    """Return list of world-space Trimesh parts (scene transforms applied)."""
    scene = trimesh.load(path, force="scene", process=False)
    return [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]


def normalisation(parts):
    v = np.concatenate([p.vertices for p in parts])
    lo, hi = v.min(0), v.max(0)
    center = (lo + hi) / 2
    scale = 1.0 / float((hi - lo).max())
    return center, scale


def apply_norm(parts, center, scale):
    out = []
    for p in parts:
        q = p.copy()
        q.vertices = (q.vertices - center) * scale
        out.append(q)
    return out


# ---------------------------------------------------------------- colour lookup
def _material_rgba(mat):
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        c = getattr(mat, attr, None)
        if c is not None:
            c = np.asarray(c, dtype=np.float64).reshape(-1)[:4]
            if c.max() > 1.0:
                c = c / 255.0
            return np.pad(c, (0, 4 - len(c)), constant_values=1.0)
    return np.ones(4)


def _texture_image(mat):
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        img = getattr(mat, "image", None)
    if img is None:
        return None
    return np.asarray(img.convert("RGBA"), dtype=np.float64) / 255.0


def _sample_colours(part, face_idx, points):
    """RGB in [0,1] for points lying on faces `face_idx` of `part`."""
    n = len(points)
    vis = part.visual
    if isinstance(vis, trimesh.visual.ColorVisuals):
        return vis.face_colors[face_idx, :3] / 255.0
    if not isinstance(vis, trimesh.visual.TextureVisuals):
        return np.full((n, 3), 0.8)

    base = _material_rgba(vis.material)[:3]
    img = _texture_image(vis.material)
    if img is None or vis.uv is None or len(vis.uv) != len(part.vertices):
        return np.tile(base, (n, 1))

    tri = part.triangles[face_idx]
    bary = trimesh.triangles.points_to_barycentric(tri, points)
    uv_tri = vis.uv[part.faces[face_idx]]                       # (n, 3, 2)
    uv = (uv_tri * bary[:, :, None]).sum(1) % 1.0               # wrap (glTF REPEAT)
    h, w = img.shape[:2]
    px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)  # glTF v=0 is bottom
    return img[py, px, :3] * base


def sample_surface(parts, n, with_colour=True, seed=0):
    """Area-weighted surface samples over all parts -> points, normals, colours."""
    rng = np.random.default_rng(seed)
    areas = np.array([p.area for p in parts])
    counts = rng.multinomial(n, areas / areas.sum())
    P, N, C = [], [], []
    for part, k in zip(parts, counts):
        if k == 0:
            continue
        pts, fidx = trimesh.sample.sample_surface(part, k, seed=int(rng.integers(1 << 31)))
        P.append(pts)
        N.append(part.face_normals[fidx])
        if with_colour:
            C.append(_sample_colours(part, fidx, pts))
    P, N = np.concatenate(P), np.concatenate(N)
    C = np.clip(np.concatenate(C), 0, 1) if with_colour else None
    return P, N, C

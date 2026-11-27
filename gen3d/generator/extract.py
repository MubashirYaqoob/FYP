"""Trained generator + one photo -> coloured 3D mesh.

Steps: encode the photo once -> ask the decoder "inside?" on a RES^3 grid (in chunks) ->
marching cubes -> ask the texture field for the colour of every vertex -> trimesh mesh.
(UV unwrapping + texture baking with xatlas comes later; vertex colours for now.)
"""
import numpy as np
import torch
import trimesh
from PIL import Image
from skimage import measure

from gen3d.data.dataset import image_to_tensor

BOUND = 0.55  # same box as preprocessing: the object lives in [-0.5, 0.5]^3


def default_camera(cfg, device):
    """Camera used for real photos: same lens as the renders, object centred 2.0 units away."""
    r = cfg["render"]
    f = r["lens_mm"] / r["sensor_mm"] * r["resolution"]
    c = r["resolution"] / 2
    K = torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1]], dtype=torch.float32, device=device)
    t = torch.tensor([0.0, 0.0, -r["inference_distance"]], device=device)
    return t, K


def background_mask(rgb, light=222, grey_tol=14, min_blob=0.002):
    """Photo without transparency -> True where background.
    Background = light, colourless pixels (plain white or a fake grey/white checkerboard).
    Small specks are cleaned up and only the biggest object regions are kept."""
    from scipy import ndimage

    rgb = rgb.astype(np.int16)
    bg = (rgb.min(-1) >= light) & ((rgb.max(-1) - rgb.min(-1)) <= grey_tol)
    fg = ndimage.binary_opening(~bg, iterations=2)
    labels, n = ndimage.label(fg)
    if n > 1:                                               # drop tiny islands (noise, watermark dots)
        sizes = ndimage.sum(fg, labels, range(1, n + 1))
        keep = np.isin(labels, 1 + np.nonzero(sizes >= min_blob * fg.size)[0])
        fg = keep if keep.any() else fg
    return ~fg


def prepare_photo(path, size, object_fraction=0.55):
    """Any photo -> RGBA square like our renders: object cropped, centred, ~55% of the frame.
    Uses the alpha channel if present; otherwise removes a plain white or checkerboard
    background (for busy real-world backgrounds, cut the object out first)."""
    img = np.asarray(Image.open(path).convert("RGBA")).copy()
    if img[..., 3].min() == 255:                                 # no transparency -> find background
        img[..., 3] = np.where(background_mask(img[..., :3]), 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(img[..., 3] > 16)
    if len(ys) == 0:
        raise ValueError("no object found in the photo")
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(crop.shape[:2]) / object_fraction)
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.LANCZOS))


@torch.no_grad()
def extract_mesh(model, image, t, K, res=128, chunk=65536, level=0.5):
    """image (3,H,W) normalised tensor, t (3,), K (3,3) -> trimesh.Trimesh with vertex colours
    (in the VIEW frame), or None if the model predicts nothing inside.
    level = inside-probability threshold for the surface (lower keeps thin parts, adds blobs)."""
    device = next(model.parameters()).device
    model.eval()
    img, t, K = image[None].to(device), t[None].to(device), K[None].to(device)
    enc = model.encode(img)

    lin = torch.linspace(-BOUND, BOUND, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    probs = torch.cat([torch.sigmoid(model.occupancy_logits(enc, grid[i:i + chunk][None], t, K))[0]
                       for i in range(0, len(grid), chunk)])
    vol = probs.reshape(res, res, res).cpu().numpy()
    if vol.max() < level or vol.min() > level:
        return None

    verts, faces, _, _ = measure.marching_cubes(vol, level)
    verts = verts / (res - 1) * 2 * BOUND - BOUND
    v = torch.from_numpy(verts.astype(np.float32)).to(device)
    rgb = torch.cat([model.colour(enc, v[i:i + chunk][None], t, K)[0] for i in range(0, len(v), chunk)])
    colours = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], vertex_colors=colours, process=False)
    return mesh


def clean_mesh(mesh, min_part=0.05, smooth_iters=10):
    """Make a generated mesh look clean without changing its overall shape.
    1. drop floating pieces smaller than `min_part` of the biggest piece (noise blobs)
    2. Taubin smoothing (removes the marching-cubes staircase, keeps the volume)"""
    import trimesh

    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        sizes = np.array([len(p.faces) for p in parts])
        mesh = trimesh.util.concatenate([p for p, n in zip(parts, sizes) if n >= min_part * sizes.max()])
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    return mesh


def photo_colours(mesh, rgba, t, K, depth_tol=0.02, splat=2):
    """Give the mesh the REAL colours of the photo where the camera can see it.

    Every vertex is projected into the photo (same camera as the model used). A vertex is visible if it
    faces the camera and is the closest surface along its pixel (a small z-buffer built from the
    vertices themselves). Visible vertices take the photo pixel colour; hidden ones (back side) keep
    the texture-field colour; the two are blended at grazing angles so no hard seam appears.
    rgba: (S, S, 4) photo prepared like the model input (any size S); K is for that size.
    """
    v = np.asarray(mesh.vertices, np.float64)
    p = v + np.asarray(t, np.float64)
    depth = -p[:, 2]
    k = np.asarray(K, np.float64)
    u = k[0, 0] * p[:, 0] / depth + k[0, 2]
    w = k[1, 2] - k[1, 1] * p[:, 1] / depth
    size = rgba.shape[0]
    ui, wi = np.clip(u.round().astype(int), 0, size - 1), np.clip(w.round().astype(int), 0, size - 1)

    zbuf = np.full((size, size), np.inf)
    for dx in range(-splat, splat + 1):          # splat each vertex a little so the surface has no holes
        for dy in range(-splat, splat + 1):
            np.minimum.at(zbuf, (np.clip(wi + dy, 0, size - 1), np.clip(ui + dx, 0, size - 1)), depth)
    visible = depth <= zbuf[wi, ui] + depth_tol

    view_dir = -p / np.linalg.norm(p, axis=1, keepdims=True)          # vertex -> camera
    facing = np.clip((np.asarray(mesh.vertex_normals) * view_dir).sum(1), 0, 1)
    inside_photo = rgba[wi, ui, 3] > 128
    weight = (visible & inside_photo) * np.clip(facing * 3, 0, 1)    # full weight unless grazing

    photo = rgba[wi, ui, :3].astype(np.float64)
    field = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float64)
    rgb = weight[:, None] * photo + (1 - weight[:, None]) * field
    colours = np.c_[rgb.round().astype(np.uint8), np.full(len(rgb), 255, np.uint8)]
    mesh.visual.vertex_colors = colours
    return mesh


def photo_to_tensor(rgba):
    return image_to_tensor(rgba, np.ones(3))

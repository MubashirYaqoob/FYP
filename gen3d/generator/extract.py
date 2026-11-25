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


def prepare_photo(path, size, object_fraction=0.55):
    """Any photo -> RGBA square like our renders: object cropped, centred, ~55% of the frame.
    Uses the alpha channel if present, else treats near-white pixels as background
    (for real photos run rembg first, e.g. in the demo)."""
    img = np.asarray(Image.open(path).convert("RGBA")).copy()
    if img[..., 3].min() == 255:                                 # no transparency -> key out white
        img[..., 3] = np.where(img[..., :3].min(-1) > 240, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(img[..., 3] > 16)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(crop.shape[:2]) / object_fraction)
    canvas = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.LANCZOS))


@torch.no_grad()
def extract_mesh(model, image, t, K, res=128, chunk=65536):
    """image (3,H,W) normalised tensor, t (3,), K (3,3) -> trimesh.Trimesh with vertex colours
    (in the VIEW frame), or None if the model predicts nothing inside."""
    device = next(model.parameters()).device
    model.eval()
    img, t, K = image[None].to(device), t[None].to(device), K[None].to(device)
    enc = model.encode(img)

    lin = torch.linspace(-BOUND, BOUND, res, device=device)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    probs = torch.cat([torch.sigmoid(model.occupancy_logits(enc, grid[i:i + chunk][None], t, K))[0]
                       for i in range(0, len(grid), chunk)])
    vol = probs.reshape(res, res, res).cpu().numpy()
    if vol.max() < 0.5 or vol.min() > 0.5:
        return None

    verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    verts = verts / (res - 1) * 2 * BOUND - BOUND
    v = torch.from_numpy(verts.astype(np.float32)).to(device)
    rgb = torch.cat([model.colour(enc, v[i:i + chunk][None], t, K)[0] for i in range(0, len(v), chunk)])
    colours = (rgb.cpu().numpy() * 255).round().astype(np.uint8)
    mesh = trimesh.Trimesh(verts, faces[:, ::-1], vertex_colors=colours, process=False)
    return mesh


def photo_to_tensor(rgba):
    return image_to_tensor(rgba, np.ones(3))

"""Show what happens INSIDE the generator for one photo, step by step (figure for report / demo).

  1 input photo (background removed, 224 x 224)
  2 DINOv2 patch features (16 x 16 x 384) shown as colours (PCA: similar features -> similar colour)
  3 3D query points projected onto the photo (each point reads the feature under it)
  4 occupancy decoder: inside-probability on slices through the 3D box (front + side cut)
  5 points the model says are inside, coloured by the texture field
  6 final mesh (marching cubes), front + side

Usage: python scripts/visualize_pipeline.py --image app/examples/wooden_chair_photo.jpg
           [--ckpt outputs/checkpoints/best_v1_chair.pt] [--out outputs/pipeline_inside.png]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.generator.extract import BOUND, default_camera, extract_mesh, photo_to_tensor, prepare_photo  # noqa: E402
from gen3d.generator.model import Generator, project  # noqa: E402
from gen3d.utils.common import project_path  # noqa: E402


def pca_rgb(tokens):
    """(N, C) features -> (N, 3) colours from the first 3 principal components."""
    x = tokens - tokens.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:3].T
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-8)
    return p


def draw_mesh(ax, m, el, az):
    v = m.vertices[:, [0, 2, 1]] * [1, -1, 1]
    f = m.faces
    fc = np.asarray(m.visual.vertex_colors)[:, :3][f].mean(1) / 255
    sh = 0.5 + 0.5 * np.clip(m.face_normals @ [0.3, 0.5, 0.8], 0, 1)
    ax.add_collection3d(Poly3DCollection(v[f], facecolors=np.clip(fc * sh[:, None], 0, 1), linewidths=0))
    c, r = v.mean(0), (v.max(0) - v.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r), ax.set_ylim(c[1] - r, c[1] + r), ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(el, az)
    ax.axis("off")


@torch.no_grad()
def pipeline_figure(model, cfg, image_path, res=96):
    """Run one photo through `model` and return a matplotlib figure of every internal step."""
    model.eval()
    size = cfg["generator"]["encoder"]["img_size"]
    rgba = prepare_photo(image_path, size)
    img = photo_to_tensor(rgba)[None]
    t, K = default_camera(cfg, torch.device("cpu"))
    t, K = t[None], K[None]

    # step 2: encoder -- raw DINOv2 patch tokens (before our projection) for the PCA picture
    enc = model.encoder
    tokens = enc.backbone.forward_features(img)[0, enc.n_prefix:].numpy()        # (256, 384)
    g = enc.grid
    fg = (rgba[..., 3].reshape(g, size // g, g, size // g).mean((1, 3)) > 16).reshape(-1)  # object patches
    feat_rgb = np.ones((g * g, 3))
    feat_rgb[fg] = pca_rgb(tokens[fg])                     # PCA over the object only -> parts get colours
    feat_rgb = feat_rgb.reshape(g, g, 3)
    feats = model.encode(img)

    # step 3: a few 3D points and where they land in the photo
    rng = np.random.default_rng(0)
    pts = torch.from_numpy(rng.uniform(-0.45, 0.45, (1, 400, 3)).astype(np.float32))
    u, v, depth = project(pts, t, K)

    # step 4 + 5: query a full grid once
    lin = torch.linspace(-BOUND, BOUND, res)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(1, -1, 3)
    prob = torch.cat([torch.sigmoid(model.occupancy_logits(feats, grid[:, i:i + 65536], t, K))
                      for i in range(0, grid.shape[1], 65536)], 1)[0].reshape(res, res, res).numpy()
    inside = grid[0][prob.reshape(-1) > 0.5]
    keep = inside[torch.randperm(len(inside))[:6000]]
    rgb = model.colour(feats, keep[None], t, K)[0].numpy()

    # step 6: mesh
    mesh = extract_mesh(model, img[0], t[0], K[0], res=128)

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Inside GEN3D-Gen: one photo, step by step", fontsize=16, weight="bold")

    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(rgba)
    ax.set_title("1. Input photo\n(background removed, 224×224)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(feat_rgb, interpolation="nearest")
    ax.set_title("2. DINOv2 encoder features\n16×16 patches × 384 numbers (PCA colours)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(rgba)
    sc = ax.scatter(u[0].numpy(), v[0].numpy(), c=depth[0].numpy(), cmap="viridis", s=6)
    ax.set_xlim(0, size), ax.set_ylim(size, 0)
    ax.set_title("3. Pixel-aligned features\n3D points projected onto the photo (colour = depth)")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="distance from camera")

    mid = res // 2
    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(prob[:, ::-1, mid].T, cmap="magma", vmin=0, vmax=1, extent=[-BOUND, BOUND, -BOUND, BOUND])
    ax.set_title("4a. Occupancy decoder: front slice\nP(inside) through the middle (z = 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y (up)")

    ax = fig.add_subplot(2, 4, 5)
    im = ax.imshow(prob[mid, ::-1, :], cmap="magma", vmin=0, vmax=1, extent=[-BOUND, BOUND, -BOUND, BOUND])
    ax.set_title("4b. Occupancy decoder: side slice\nP(inside) at x = 0 (right = camera side)")
    ax.set_xlabel("z (depth)")
    ax.set_ylabel("y (up)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(inside)")

    ax = fig.add_subplot(2, 4, 6, projection="3d")
    k = keep.numpy()
    ax.scatter(k[:, 0], -k[:, 2], k[:, 1], c=np.clip(rgb, 0, 1), s=2)
    ax.view_init(20, -35)
    ax.axis("off")
    ax.set_title("5. Points the model calls 'inside'\ncoloured by the texture field")

    for i, (el, az, lab) in enumerate([(8, -90, "front"), (20, -35, "side")]):
        ax = fig.add_subplot(2, 4, 7 + i, projection="3d")
        if mesh is not None:
            draw_mesh(ax, mesh, el, az)
        ax.set_title(f"6. Final mesh (marching cubes)\n{lab} view")

    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="outputs/checkpoints/best_v1_chair.pt")
    ap.add_argument("--out", default="outputs/pipeline_inside.png")
    ap.add_argument("--res", type=int, default=96)
    args = ap.parse_args()
    ck = torch.load(project_path(args.ckpt), map_location="cpu", weights_only=False)
    model = Generator(ck["config"])
    model.load_state_dict(ck["model"])
    fig = pipeline_figure(model, ck["config"], args.image, args.res)
    out = project_path(args.out)
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()

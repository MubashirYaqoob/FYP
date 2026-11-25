"""Look at what preprocess.py produced for a few objects.

For each object, one row of 3 pictures (front view, Y-up):
  1. coloured surface points   -> what the texture field learns
  2. INSIDE occupancy points   -> what the decoder learns ("this is solid")
  3. watertight mesh (if --save-mesh was used) as points

Usage:  python scripts/visualize_sample.py [--n 6]   -> outputs/preprocess_check.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path  # noqa: E402


def view(ax, p, c, title, s=0.3):
    # simple oblique projection: x right, y up, z towards viewer (shifted to give depth)
    order = np.argsort(p[:, 2])
    p, c = p[order], c[order] if isinstance(c, np.ndarray) and c.ndim == 2 else c
    ax.scatter(p[:, 0] + 0.35 * p[:, 2], p[:, 1] + 0.2 * p[:, 2], c=c, s=s, linewidths=0)
    ax.set_xlim(-0.75, 0.75), ax.set_ylim(-0.7, 0.7), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config()
    meta = project_path(cfg["data"]["meta_dir"])
    names = pd.read_csv(meta / "checked.csv", keep_default_na=False).set_index("uid")
    objs = sorted((project_path(cfg["data"]["processed_dir"]) / "objects").glob("*/data.npz"))[:args.n]

    fig, axes = plt.subplots(len(objs), 3, figsize=(9, 3 * len(objs)))
    axes = np.atleast_2d(axes)
    for row, f in zip(axes, objs):
        uid = f.parent.name
        z = np.load(f)
        label = f"{names.loc[uid, 'category']}: {str(names.loc[uid, 'name'])[:30]}"
        sp, sc = z["surf_points"].astype(np.float32), z["surf_colors"] / 255.0
        view(row[0], sp[::3], sc[::3], label + "\nsurface colours")
        pts, occ = z["points"].astype(np.float32), z["occ"]
        view(row[1], pts[occ], "tab:blue", f"inside points ({occ.mean():.1%} of samples)", s=0.5)
        wt = f.parent / "watertight.obj"
        if wt.exists():
            import trimesh
            m = trimesh.load(wt, process=False)
            p, _ = trimesh.sample.sample_surface(m, 15000)
            view(row[2], p, "tab:gray", f"watertight ({len(m.faces)} faces)")
        else:
            row[2].axis("off")
        n = json.load(open(f.parent / "norm.json"))
        row[2].text(0, -0.68, f"scale={n['scale']:.3g}", fontsize=6, ha="center")

    out = project_path("outputs/preprocess_check.png")
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

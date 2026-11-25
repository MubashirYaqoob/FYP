"""Photo -> 3D mesh with a trained generator.

Usage:
  python scripts/generate_mesh.py --image my_chair.png                 # any photo (white or transparent bg)
  python scripts/generate_mesh.py --uid <object id> [--view 3]         # a dataset object, with its true camera
Options: --ckpt outputs/checkpoints/best.pt  --res 128  --out outputs/meshes
Output: <out>/<name>.glb (vertex colours) and, for --uid, <name>_gt.glb (true shape, same frame) to compare.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.generator.extract import default_camera, extract_mesh, photo_to_tensor, prepare_photo  # noqa: E402
from gen3d.generator.model import Generator  # noqa: E402
from gen3d.utils.common import load_config, project_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--uid")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    ap.add_argument("--res", type=int, default=None)
    ap.add_argument("--out", default="outputs/meshes")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(project_path(args.ckpt), map_location=device)
    cfg = ck["config"]
    model = Generator(cfg).to(device)
    model.load_state_dict(ck["model"])
    res = args.res or cfg["generator"]["mc_resolution"]
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.image:
        rgba = prepare_photo(args.image, cfg["generator"]["encoder"]["img_size"])
        t, K = default_camera(cfg, device)
        name = Path(args.image).stem
        gt = None
    else:
        obj = project_path(cfg["data"]["processed_dir"]) / "objects" / args.uid
        cams = json.load(open(obj / "views" / "cameras.json"))
        frame = cams["frames"][args.view]
        rgba = np.asarray(Image.open(obj / "views" / frame["file"]).convert("RGBA"))
        w2c = np.linalg.inv(np.asarray(frame["c2w"]))
        t = torch.tensor(w2c[:3, 3], dtype=torch.float32)
        K = torch.tensor(cams["K"], dtype=torch.float32)
        name = f"{args.uid[:8]}_v{args.view}"
        z = np.load(obj / "data.npz")
        gt = (z["surf_points"].astype(np.float32) @ w2c[:3, :3].T.astype(np.float32), z["surf_colors"])

    mesh = extract_mesh(model, photo_to_tensor(rgba), t, K, res=res)
    if mesh is None:
        print("model predicted no inside points (not trained enough?)")
        return
    mesh.export(out / f"{name}.glb")
    Image.fromarray(rgba).save(out / f"{name}_input.png")
    print(f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces -> {out / (name + '.glb')}")
    if gt is not None:  # true surface as a coloured point cloud in the same (view) frame
        import trimesh
        trimesh.PointCloud(gt[0], colors=np.c_[gt[1], np.full(len(gt[1]), 255)]).export(out / f"{name}_gt.glb")


if __name__ == "__main__":
    main()

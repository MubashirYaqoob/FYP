"""Evaluate a trained generator the way Occupancy Networks (Mescheder et al., CVPR 2019) does.

Per test object (one fixed view, default view 0):
  IoU               inside/outside on the object's stored query points (for ShapeNet/ONet data these are
                    100k uniform points in the box = ONet's volumetric IoU), at thresholds 0.5 and 0.2
  Chamfer-L1        0.5 * (mean dist predicted surface -> true surface + mean dist true -> predicted)
  Normal consistency 0.5 * (mean |n_pred . n_true| both ways)   (needs surf_normals, ShapeNet has them)
  F-score@1%        harmonic mean of precision/recall of surface points within 0.01 (box size ~1.1)
Surfaces: 100k points sampled on our marching-cubes mesh vs the object's stored surface points.

Usage: python scripts/evaluate.py --ckpt best.pt --meta-dir M --objects-dir O --views-zip-dir V
           [--split test] [--name-regex shapenet] [--limit 500] [--out eval.csv]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.data.dataset import GEN3DDataset, load_object_table  # noqa: E402
from gen3d.generator.extract import extract_mesh  # noqa: E402
from gen3d.generator.model import Generator  # noqa: E402

N_SURF = 100000
F_TAU = 0.01


def surface_metrics(mesh, gt_pts, gt_nrm):
    pts, fid = mesh.sample(N_SURF, return_index=True)
    nrm = mesh.face_normals[fid]
    d_pred, i_pred = cKDTree(gt_pts).query(pts)        # accuracy: predicted -> true
    d_true, i_true = cKDTree(pts).query(gt_pts)        # completeness: true -> predicted
    out = {"chamfer_l1": 0.5 * (d_pred.mean() + d_true.mean())}
    if gt_nrm is not None:
        gn = gt_nrm / (np.linalg.norm(gt_nrm, axis=1, keepdims=True) + 1e-9)
        out["normal_consistency"] = 0.5 * (np.abs((nrm * gn[i_pred]).sum(1)).mean()
                                           + np.abs((gn * nrm[i_true]).sum(1)).mean())
    prec, rec = (d_pred < F_TAU).mean(), (d_true < F_TAU).mean()
    out["fscore_1pct"] = 2 * prec * rec / (prec + rec + 1e-9)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--views-zip-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--name-regex", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default="eval.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    cfg["train"]["val_view"] = args.view
    model = Generator(cfg).to(device).eval()
    model.load_state_dict(ck["model"])

    tab = load_object_table(args.meta_dir, args.objects_dir, args.split, views_zip_dir=args.views_zip_dir)
    if args.categories:
        tab = tab[tab.category.isin(args.categories.split(","))]
    if args.name_regex:
        tab = tab[tab.name.str.contains(args.name_regex, case=False, regex=True)]
    tab = tab.reset_index(drop=True)
    if args.limit:
        tab = tab.head(args.limit)
    ds = GEN3DDataset(tab, args.objects_dir, cfg, train=False, views_zip_dir=args.views_zip_dir)
    print(f"evaluating {len(tab)} objects ({args.split}) on {device}", flush=True)

    rows, t0 = [], time.time()
    for i, uid in enumerate(tab.uid):
        item = ds[i]                                   # fixed view: image, camera
        z = np.load(Path(args.objects_dir) / uid / "data.npz")
        cams = ds.cameras(uid)
        w2c = np.linalg.inv(np.asarray(cams["frames"][args.view]["c2w"], np.float64))
        rot = w2c[:3, :3].astype(np.float32)
        img, t, K = item["image"][None].to(device), item["t"][None].to(device), item["K"][None].to(device)
        enc = model.encode(img)

        pts = torch.from_numpy(z["points"].astype(np.float32) @ rot.T).to(device)
        prob = torch.cat([torch.sigmoid(model.occupancy_logits(enc, pts[None, j:j + 50000], t, K))[0]
                          for j in range(0, len(pts), 50000)]).cpu().numpy()
        occ = z["occ"].astype(bool)
        row = {"uid": uid, "category": tab.category[i]}
        for th in (0.5, 0.2):
            pred = prob > th
            row[f"iou_{th}"] = (pred & occ).sum() / max((pred | occ).sum(), 1)

        mesh = extract_mesh(model, item["image"], item["t"], item["K"], res=args.res, level=0.5)
        if mesh is not None and len(mesh.faces):
            gt = z["surf_points"].astype(np.float32) @ rot.T
            gn = z["surf_normals"].astype(np.float32) @ rot.T if "surf_normals" in z.files else None
            row.update(surface_metrics(mesh, gt, gn))
        rows.append(row)
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(rows)
            print(f"{i + 1}/{len(tab)}  IoU@0.5 {df['iou_0.5'].mean():.3f}  IoU@0.2 {df['iou_0.2'].mean():.3f}  "
                  f"CD {df.get('chamfer_l1', pd.Series([np.nan])).mean():.4f}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.drop(columns=["uid"]).groupby("category").mean().round(4)
    summary["n"] = df.groupby("category").size()
    print("\n=== EVALUATION (" + Path(args.ckpt).name + ") ===\n" + summary.to_string())
    json.dump(summary.reset_index().to_dict("records"), open(Path(args.out).with_suffix(".json"), "w"), indent=1)


if __name__ == "__main__":
    main()

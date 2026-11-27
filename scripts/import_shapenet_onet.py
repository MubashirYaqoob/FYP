"""Third data source: ShapeNet as preprocessed by Occupancy Networks (Mescheder et al., CVPR 2019).

We follow the ONet setup exactly: same objects, same train/val/test lists, same inside/outside
points (100k uniform samples in [-0.55, 0.55]^3) and the same 24 renders per object (Choy et al. 2016,
137 x 137). This script only converts their files into our format, so our training and evaluation
code reads ShapeNet like any other dataset:

  objects/<uid>/data.npz   points (float16), occ (uint8), surf_points + normals (from pointcloud.npz)
  views/<uid>.zip          the 24 JPEG renders + cameras.json (our OpenGL camera convention)
  meta/candidates.csv, meta/preprocessed.csv

Camera conversion (checked by projecting the point cloud onto the renders):
  ONet world_mat = [R | t] maps (scale * p + loc) into an OpenCV camera (x right, y down, z forward).
  Projection is scale-invariant, so for normalised points p:  p_cam = R p + (R loc + t) / scale.
  OpenCV -> OpenGL camera: flip y and z.  K is scaled from 137 px to 224 px (same lens as our renders).

Usage: python scripts/import_shapenet_onet.py --src <dir with ShapeNet/<synset>/> --synset 03001627
           --category chair --out <dir>   [--limit 20]
"""
import argparse
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT_RES = 224
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def convert(job):
    src_dir, uid, out = job
    src = Path(src_dir)
    try:
        p = np.load(src / "points.npz")
        pc = np.load(src / "pointcloud.npz")
        cams = np.load(src / "img_choy2016" / "cameras.npz")
        pts = p["points"]
        occ = np.unpackbits(p["occupancies"])[:len(pts)].astype(np.uint8)
        sel = np.random.default_rng(0).choice(len(pc["points"]), 50000, replace=False)
        obj = Path(out) / "objects" / uid
        obj.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(obj / "data.npz", points=pts.astype(np.float16), occ=occ,
                 surf_points=pc["points"][sel].astype(np.float16),
                 surf_normals=pc["normals"][sel].astype(np.float16),
                 # ShapeNet has no reliable colours: grey placeholder, texture loss is switched off for it
                 surf_colors=np.full((len(sel), 3), 128, np.uint8))
        json.dump({"source": "shapenet_onet", "loc": p["loc"].tolist(), "scale": float(p["scale"]),
                   "flag": ""}, open(obj / "norm.json", "w"))

        loc, scale = np.asarray(p["loc"], np.float64), float(p["scale"])
        k = np.asarray(cams["camera_mat_0"], np.float64) * (OUT_RES / 137.0)
        k[2, 2] = 1.0
        frames = []
        for i in range(24):
            w = np.asarray(cams[f"world_mat_{i}"], np.float64)
            r = CV_TO_GL @ w[:, :3]
            t = CV_TO_GL @ ((w[:, :3] @ loc + w[:, 3]) / scale)
            w2c = np.eye(4)
            w2c[:3, :3], w2c[:3, 3] = r, t
            frames.append({"file": f"{i:03d}.jpg", "c2w": np.linalg.inv(w2c).tolist()})
        with zipfile.ZipFile(Path(out) / "views" / f"{uid}.zip", "w", zipfile.ZIP_STORED) as z:
            for i in range(24):
                z.write(src / "img_choy2016" / f"{i:03d}.jpg", f"{i:03d}.jpg")
            z.writestr("cameras.json", json.dumps({"convention": "opengl", "frame": "shapenet_onet_normalised",
                                                   "res": OUT_RES, "K": k.tolist(), "frames": frames}))
        return uid, "ok"
    except Exception as e:  # noqa: BLE001
        return uid, f"error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder that contains ShapeNet/<synset>/")
    ap.add_argument("--synset", default="03001627")
    ap.add_argument("--category", default="chair")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.src) / "ShapeNet" / args.synset
    out = Path(args.out)
    for d in ("objects", "views", "meta"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "val", "test"):
        for line in open(root / f"{split}.lst"):
            if line.strip():
                rows.append({"shapenet_id": line.strip(), "split": split})
    df = pd.DataFrame(rows)
    if args.limit:
        df = df.groupby("split", group_keys=False).head(args.limit)
    df["uid"] = "sn_" + df.shapenet_id
    jobs = [(str(root / s), u, str(out)) for s, u in zip(df.shapenet_id, df.uid)]
    with ProcessPoolExecutor(args.workers) as ex:
        status = dict(ex.map(convert, jobs, chunksize=16))
    df["status"] = df.uid.map(status)

    cand = pd.DataFrame({"uid": df.uid, "group": "demo", "category": args.category,
                         "name": f"shapenet {args.category} " + df.shapenet_id, "split": df.split,
                         "license": "ShapeNet terms (research)", "selected": True, "reject": ""})
    pre = pd.DataFrame({"uid": df.uid, "status": df.status.where(df.status == "ok", df.status),
                        "seconds": 0.0, "uniform_inside": "", "flag": ""})
    for name, new in (("candidates.csv", cand), ("preprocessed.csv", pre)):
        f = out / "meta" / name
        old = pd.read_csv(f, keep_default_na=False) if f.exists() else new.iloc[:0]
        pd.concat([old[~old.uid.isin(new.uid)], new]).to_csv(f, index=False)
    print(df.groupby(["split", "status"]).size().to_string())


if __name__ == "__main__":
    main()

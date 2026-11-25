"""Step 6: turn every PASS mesh into generator training data.

Per object -> data/processed/objects/<uid>/
    data.npz   points (N,3) f16, occ (N,) bool       -> "is this point inside?"  (decoder target)
               surf_points (M,3) f16, surf_normals (M,3) f16, surf_colors (M,3) u8  (texture-field target)
    norm.json  center, scale, frame  -> SAME normalisation is used by the Blender renderer
    watertight.obj (only with --save-mesh, for checking by eye)

Usage:  python scripts/preprocess.py [--limit 5] [--save-mesh] [--workers 4]
"""
import argparse
import json
import zlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.data import mesh_io, occupancy  # noqa: E402
from gen3d.utils.common import load_config, project_path  # noqa: E402


def process_one(job):
    uid, path, out_dir, oc, save_mesh = job
    out = Path(out_dir) / uid
    if (out / "data.npz").exists() and (out / "norm.json").exists():
        n = json.load(open(out / "norm.json"))
        return uid, "ok", 0.0, n.get("uniform_inside"), n.get("flag", "")
    t0 = time.time()
    try:
        rng = np.random.default_rng(zlib.crc32(uid.encode()))
        parts = mesh_io.load_parts(path)
        center, scale = mesh_io.normalisation(parts)
        parts = mesh_io.apply_norm(parts, center, scale)

        # dense uncoloured samples for voxelisation (cheap), coloured samples for texture field
        dense, _, _ = mesh_io.sample_surface(parts, oc["voxel_surface_samples"], with_colour=False)
        occ_fn = occupancy.Occupancy(dense, res=oc["grid_res"], level_voxels=oc["level_voxels"])
        sp, sn, sc = mesh_io.sample_surface(parts, oc["surface_color_points"], with_colour=True)

        # query points: near-surface at several distances (sharp edges + context) + uniform in the box
        n = oc["points_per_object"]
        chunks = []
        for sigma, ratio in zip(oc["near_sigmas"], oc["near_ratios"]):
            k = int(n * ratio)
            chunks.append(sp[rng.integers(0, len(sp), k)] + rng.normal(0, sigma, (k, 3)))
        n_uni = n - sum(len(c) for c in chunks)
        uni = rng.uniform(-occupancy.BOUND, occupancy.BOUND, (n_uni, 3))
        pts = np.concatenate(chunks + [uni])
        occ = occ_fn.query(pts)

        # sanity flags (object is still saved; training skips flagged objects unless reviewed)
        uni_in = float(occ[-n_uni:].mean())
        flag = ""
        ext = np.ptp(np.concatenate([p.vertices for p in parts]), axis=0)
        bulky = ext.max() / max(ext.min(), 1e-6) < oc["bulky_max_aspect"]
        if uni_in < oc["uniform_inside_min"] and bulky:
            flag = "inside_too_low"        # bulky object but empty inside: a gap let the flood fill in
        elif uni_in > oc["uniform_inside_max"]:
            flag = "inside_too_high"       # enclosing dome / box swallowed everything
        elif sc.mean() > 0.94 and sc.std() < 0.02:
            flag = "all_white"             # no real colour -> geometry only, not texture training

        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "data.npz",
            points=pts.astype(np.float16), occ=occ,
            surf_points=sp.astype(np.float16), surf_normals=sn.astype(np.float16),
            surf_colors=(sc * 255).round().astype(np.uint8))
        json.dump({"center": center.tolist(), "scale": scale, "frame": "gltf_y_up",
                   "bound": occupancy.BOUND, "grid_res": oc["grid_res"], "level": occ_fn.level,
                   "uniform_inside": uni_in, "flag": flag},
                  open(out / "norm.json", "w"), indent=1)
        if save_mesh:
            occ_fn.to_mesh().export(out / "watertight.obj")
        return uid, "ok", time.time() - t0, uni_in, flag
    except Exception as e:  # noqa: BLE001
        return uid, f"error:{type(e).__name__}:{e}"[:200], time.time() - t0, None, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-mesh", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long (Kaggle 12 h limit)")
    args = ap.parse_args()

    cfg = load_config()
    oc = cfg["occupancy"]
    meta = project_path(cfg["data"]["meta_dir"])
    out_dir = project_path(cfg["data"]["processed_dir"]) / "objects"
    df = pd.read_csv(meta / "checked.csv", keep_default_na=False)
    df = df[df.status == "PASS"]
    if args.limit:
        df = df.head(args.limit)

    jobs = [(r.uid, r.path, str(out_dir), oc, args.save_mesh) for r in df.itertuples()]
    cols = ["uid", "status", "seconds", "uniform_inside", "flag"]
    log = meta / "preprocessed.csv"
    old = pd.read_csv(log, keep_default_na=False) if log.exists() else pd.DataFrame(columns=cols)

    def save(results):  # called every 100 objects -> a Colab disconnect loses almost nothing
        res = pd.DataFrame(results, columns=cols)
        res = pd.concat([old[~old.uid.isin(res.uid)], res])
        res.to_csv(log, index=False)
        return res

    results = []
    ex = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    it = ex.map(process_one, jobs, chunksize=4) if ex else map(process_one, jobs)
    t_start = time.time()
    for k, r in enumerate(tqdm(it, total=len(jobs), desc="preprocess"), 1):
        results.append(r)
        if k % 100 == 0:
            save(results)
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print(f"[time limit] stopping after {k} objects; re-run to continue")
            break
    if ex:
        ex.shutdown(wait=False, cancel_futures=True)
    res = save(results)

    print(res.status.str.split(":").str[0].value_counts().to_string())
    ok = res[res.status == "ok"]
    print("flags:\n" + ok.flag.replace("", "clean").value_counts().to_string())
    new = ok[ok.seconds.astype(float) > 0]
    if len(new):
        print(f"avg {new.seconds.astype(float).mean():.1f}s/object")
    print(f"Saved -> {out_dir}  |  log -> {log}")


if __name__ == "__main__":
    main()

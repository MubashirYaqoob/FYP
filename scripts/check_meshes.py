"""Step 3: open every downloaded GLB and run geometry/texture checks.

Checks: loads at all, real UVs + texture image, face count, bbox aspect (flat planes / sticks),
ground-plane / scene suspicion (flagged for manual review, not auto-rejected).

Output: data/meta/checked.csv  (status = PASS / REJECT / REVIEW, with reason)
Usage:  python scripts/check_meshes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path  # noqa: E402


def has_texture(geom):
    v = getattr(geom, "visual", None)
    if not isinstance(v, trimesh.visual.TextureVisuals) or v.uv is None or len(v.uv) == 0:
        return False
    mat = v.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    return img is not None


def check(path, f):
    """Return (status, reason, info dict)."""
    try:
        scene = trimesh.load(path, force="scene", process=False)
    except Exception as e:  # noqa: BLE001
        return "REJECT", f"load_error:{type(e).__name__}", {}

    geoms = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geoms:
        return "REJECT", "no_triangles", {}

    faces = sum(len(g.faces) for g in geoms)
    tex_faces = sum(len(g.faces) for g in geoms if has_texture(g))
    ext = np.asarray(scene.extents, dtype=float)
    info = {
        "n_parts": len(geoms),
        "faces_loaded": faces,
        "tex_face_ratio": round(tex_faces / faces, 3),
        "aspect": round(float(ext.max() / max(ext.min(), 1e-9)), 1),
        "highpoly": faces >= f["highpoly_faces"],
        # pool B (texture-preservation work) needs real UV + image texture on most faces
        "poolB": tex_faces / faces >= 0.5,
    }

    if faces < f["min_faces"] or faces > f["max_faces"]:
        return "REJECT", "face_count", info
    if info["aspect"] > f["max_aspect"]:
        return "REJECT", "degenerate_bbox", info

    # Scene / ground-plane suspicion: one very flat part much wider than everything else.
    for g in geoms:
        e = g.extents
        if e.min() < 0.01 * ext.max() and np.sort(e)[-2] > 0.9 * np.sort(ext)[-2] and len(geoms) > 1:
            return "REVIEW", "possible_ground_plane", info
    return "PASS", "", info


def _check_row(args):
    r, f = args
    status, reason, info = check(r["path"], f)
    return {**r, "status": status, "reason": reason, **info}


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    f = cfg["data"]["filter"]
    meta = project_path(cfg["data"]["meta_dir"])
    df = pd.read_csv(meta / "downloaded.csv", keep_default_na=False)
    df = df[df.path != ""]

    # incremental: keep earlier results, only check new files that are actually on disk
    out = meta / "checked.csv"
    old = pd.read_csv(out, keep_default_na=False) if out.exists() else pd.DataFrame(columns=["uid"])
    todo = df[~df.uid.isin(old.uid) & df.path.map(lambda p: Path(p).exists())]
    jobs = [(r, f) for r in todo.to_dict("records")]
    it = ProcessPoolExecutor(args.workers).map(_check_row, jobs, chunksize=8) if args.workers > 1 \
        else map(_check_row, jobs)
    out_rows = list(tqdm(it, total=len(jobs), desc="checking"))
    res = pd.concat([old, pd.DataFrame(out_rows)], ignore_index=True)
    # refresh paths (a new Colab session re-downloads to the same place)
    res["path"] = res.uid.map(dict(zip(df.uid, df.path))).fillna(res.path)
    res.to_csv(out, index=False)
    print("\n=== Stage-2 summary ===")
    key = res["group"] if "group" in res else res.category
    print(pd.crosstab(key, res.status).to_string())
    rej = res[res.status != "PASS"]
    if len(rej):
        print("\nReasons:\n" + rej.reason.value_counts().to_string())
    ok = res[res.status == "PASS"]
    print("\nPASS pools (A = generator, B = image-textured, highpoly = simplifier source):")
    print(ok.groupby("group" if "group" in ok else "category").agg(poolA=("uid", "size"), poolB=("poolB", "sum"),
                                     highpoly=("highpoly", "sum")).to_string())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

"""Step 2: download GLBs for objects that passed stage 1.

Output: data/raw/objaverse/hf-objaverse-v1/glbs/...  and  data/meta/downloaded.csv
Usage:  python scripts/download_objaverse.py [--limit 50] [--category chair] [--workers 8]
Re-running is safe: files already on disk are skipped, half-finished (.tmp) files are redone.

Note: we use our own thread-based downloader instead of objaverse.load_objects(), because on
Windows its worker processes ignore our download folder and write to ~/.objaverse.
"""
import argparse
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path, use_project_objaverse_dir  # noqa: E402

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


def fetch(uid, rel, root):
    local = os.path.join(root, rel)
    if os.path.exists(local):
        return uid, local, "cached"
    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".tmp"
    try:
        urllib.request.urlretrieve(HF + rel, tmp)
        os.replace(tmp, local)
        return uid, local, "downloaded"
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        return uid, "", f"error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max objects per GROUP (demo/related/other), 0 = all")
    ap.add_argument("--group", default=None, help="demo | related | other")
    ap.add_argument("--category", default=None, help="e.g. chair")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=0, help="download at most N objects not yet done (chunks)")
    args = ap.parse_args()

    cfg = load_config()
    objaverse = use_project_objaverse_dir(cfg)
    root = objaverse._VERSIONED_PATH
    meta = project_path(cfg["data"]["meta_dir"])
    sel = pd.read_csv(meta / "candidates.csv", keep_default_na=False)
    sel = sel[(sel.reject == "") & (sel.selected.astype(str) == "True")]
    df = sel
    if args.group:
        df = df[df.group == args.group]
    if args.category:
        df = df[df.category == args.category]
    if args.limit:
        df = df.groupby("group", group_keys=False).head(args.limit)
    # resume on a fresh Colab session: never re-download what is already checked or preprocessed
    done = set()
    objs = project_path(cfg["data"]["processed_dir"]) / "objects"
    if objs.exists():
        done |= {p.parent.name for p in objs.glob("*/data.npz")}
    if (meta / "checked.csv").exists():
        chk = pd.read_csv(meta / "checked.csv", keep_default_na=False)
        done |= set(chk[chk.status != "PASS"].uid)   # REJECT / REVIEW are final decisions
    df = df[~df.uid.isin(done)]
    if args.max_new:  # chunked mode: download -> check -> preprocess -> delete raw -> next chunk
        df = df.head(args.max_new)

    rel_paths = objaverse._load_object_paths()
    print(f"{len(df)} objects ({df.size_mb.astype(float).sum() / 1024:.2f} GB max) -> {root}")
    paths, stats = {}, {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, u, rel_paths[u], root) for u in df.uid]
        for f in tqdm(as_completed(futs), total=len(futs), desc="download"):
            uid, local, status = f.result()
            paths[uid] = local
            stats[status.split(":")[0]] = stats.get(status.split(":")[0], 0) + 1

    # merge with earlier runs; metadata columns always come fresh from candidates.csv
    out = meta / "downloaded.csv"
    if out.exists():
        old = pd.read_csv(out, keep_default_na=False)
        paths = {**dict(zip(old.uid, old.path)), **paths}
    paths = {u: p for u, p in paths.items() if p}
    df = sel[sel.uid.isin(paths)].assign(path=lambda x: x.uid.map(paths))
    df.to_csv(out, index=False)
    print(stats)
    print(f"On disk: {(df.path != '').sum()} objects  -> {out}")


if __name__ == "__main__":
    main()

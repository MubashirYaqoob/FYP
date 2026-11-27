"""Second dataset: pick ABO (Amazon Berkeley Objects) models of the product types in configs -> abo.

Reads ABO's public metadata (3D model list + product listings), keeps the product types we want,
gives each object a uid (`abo_<id>`), a category, a split (same 80/10/10 rule as Objaverse) and the
download URL of its GLB.

Output: <meta-dir>/candidates.csv  (same columns the rest of the pipeline uses, plus `url`)
Usage:  python scripts/build_abo_candidates.py [--meta-dir data/meta_abo] [--limit 20]
        python scripts/build_abo_candidates.py --name-include dining --name-exclude "arm|bar |swivel"
"""
import argparse
import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path  # noqa: E402
from scripts.build_candidates import split_of  # noqa: E402

LISTING_PARTS = "0123456789abcdef"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def product_types(base):
    """3dmodel_id -> (ABO product_type, English product name), from the 16 listing files."""
    types = {}
    for part in LISTING_PARTS:
        raw = gzip.decompress(get(f"{base}/listings/metadata/listings_{part}.json.gz"))
        for line in io.StringIO(raw.decode("utf-8")):
            j = json.loads(line)
            if "3dmodel_id" in j:
                names = [n["value"] for n in j.get("item_name", []) if n.get("language_tag", "").startswith("en")]
                types[j["3dmodel_id"]] = ((j.get("product_type") or [{}])[0].get("value", ""),
                                          names[0] if names else "")
    return types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", default="data/meta_abo")
    ap.add_argument("--limit", type=int, default=0, help="keep only N objects (quick test)")
    ap.add_argument("--name-include", default="", help="regex: keep only products whose name matches")
    ap.add_argument("--name-exclude", default="", help="regex: drop products whose name matches")
    ap.add_argument("--types", default="", help="comma list of ABO product types (default: all in config)")
    args = ap.parse_args()

    cfg = load_config()
    a = cfg["abo"]
    models = pd.read_csv(io.BytesIO(get(f"{a['base_url']}/3dmodels/metadata/3dmodels.csv.gz")),
                         compression="gzip")
    info = product_types(a["base_url"])
    models["ptype"] = models["3dmodel_id"].map(lambda m: info.get(m, ("", ""))[0])
    models["pname"] = models["3dmodel_id"].map(lambda m: info.get(m, ("", ""))[1])
    wanted = args.types.split(",") if args.types else list(a["product_types"])
    keep = models[models.ptype.isin(wanted)].copy()
    if args.name_include:
        keep = keep[keep.pname.str.contains(args.name_include, case=False, regex=True)]
    if args.name_exclude:
        keep = keep[~keep.pname.str.contains(args.name_exclude, case=False, regex=True)]
    if args.limit:
        keep = keep.groupby("ptype", group_keys=False).head(max(1, args.limit // len(a["product_types"])))

    out = pd.DataFrame({
        "uid": a["uid_prefix"] + keep["3dmodel_id"],
        "group": a["group"],
        "category": keep.ptype.map(a["product_types"]),
        "lvis_name": keep.ptype.str.lower(),
        "name": keep.pname,
        "faces": keep.faces,
        "textures": keep.textures,
        "tex_res": keep.image_width_max.fillna(0).astype(int),
        "size_mb": 0,
        "likes": 0,
        "license": "CC BY-NC 4.0 (ABO)",
        "image_texture": keep.textures > 0,
        "reject": "",
        "selected": True,
        "url": a["base_url"] + "/3dmodels/original/" + keep.path,
    })
    out["split"] = out.uid.map(split_of)
    meta = project_path(args.meta_dir)
    meta.mkdir(parents=True, exist_ok=True)
    out.to_csv(meta / "candidates.csv", index=False)
    print(out.groupby(["category", "split"]).size().unstack(fill_value=0).to_string())
    print(f"{len(out)} ABO objects -> {meta / 'candidates.csv'}")


if __name__ == "__main__":
    main()

"""Step 1: choose ~8k objects from Objaverse-LVIS (class-agnostic dataset), no 3D download yet.

Selection (all from ONE dataset):
  demo    : our 4 demo categories (chair, table, cup, cupboard) -> every object that passes the filter
  related : household / furniture / toys / vehicles              -> boosted (related_weight per round)
  other   : every other LVIS category (minus blacklist)          -> round-robin, 1 per category per round,
            so we get the widest possible variety of structures until target_total is reached
Inside each category the best objects come first (image texture, then popularity).

Splits: demo/related/other -> train/val/test (80/10/10 per category);
        `unseen_holdout_categories` whole "other" categories -> test_unseen (never seen in training).

Output: data/meta/candidates.csv   (every filtered object; selected=True for the chosen ones)
Usage:  python scripts/build_candidates.py
"""
import random
import re
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path, use_project_objaverse_dir  # noqa: E402


def stage1_reason(a, f, blacklist):
    """Return '' if the object passes, otherwise the first rejection reason."""
    glb = (a.get("archives") or {}).get("glb") or {}
    faces = glb.get("faceCount") or a.get("faceCount") or 0
    if not glb:
        return "no_glb"
    if a.get("isAgeRestricted"):
        return "age_restricted"
    if not f["allow_animated"] and (a.get("animationCount") or 0) > 0:
        return "animated"
    if faces < f["min_faces"]:
        return "too_few_faces"
    if faces > f["max_faces"]:
        return "too_many_faces"
    if (glb.get("size") or 0) > f["max_glb_mb"] * 1024 * 1024:
        return "file_too_big"
    if blacklist.search(a.get("name") or ""):
        return "name_looks_like_scene"
    return ""


def split_of(uid, train=0.8, val=0.1):
    x = (zlib.crc32(uid.encode()) % 1000) / 1000
    return "train" if x < train else ("val" if x < train + val else "test")


def main():
    cfg = load_config()
    d = cfg["data"]
    f = d["filter"]
    objaverse = use_project_objaverse_dir(cfg)
    name_bl = re.compile(r"\b(" + "|".join(map(re.escape, f["name_blacklist"])) + r")\b", re.I)
    cat_bl = re.compile("|".join(map(re.escape, d["category_blacklist"])), re.I)

    lvis = objaverse.load_lvis_annotations()
    demo_of = {n: g for g, names in d["demo_categories"].items() for n in names}
    related = set(d["related_categories"])
    for n in list(demo_of) + sorted(related):
        if n not in lvis:
            print(f"[warn] LVIS category not found: {n}")

    rows, seen = [], set()
    for lname, uids in lvis.items():
        if lname in demo_of:
            group, cat = "demo", demo_of[lname]
        elif lname in related:
            group, cat = "related", lname
        elif cat_bl.search(lname):
            continue
        else:
            group, cat = "other", lname
        for uid in uids:
            if uid not in seen:
                seen.add(uid)
                rows.append({"uid": uid, "group": group, "category": cat, "lvis_name": lname})

    print(f"Fetching metadata for {len(rows)} objects ...")
    ann = objaverse.load_annotations([r["uid"] for r in rows])
    for r in rows:
        a = ann.get(r["uid"], {})
        glb = (a.get("archives") or {}).get("glb") or {}
        r.update(
            name=a.get("name"),
            faces=glb.get("faceCount") or a.get("faceCount"),
            textures=glb.get("textureCount"),
            tex_res=glb.get("textureMaxResolution"),
            size_mb=round((glb.get("size") or 0) / 1e6, 2),
            likes=a.get("likeCount") or 0,
            license=a.get("license"),
            image_texture=(glb.get("textureCount") or 0) >= f["min_textures"],
            reject=stage1_reason(a, f, name_bl) if a else "no_metadata",
        )
    df = pd.DataFrame(rows)
    small = (df.group == "demo") | (df.size_mb <= d["max_glb_mb_nondemo"])
    df.loc[(df.reject == "") & ~small, "reject"] = "file_too_big_nondemo"
    ok = df[df.reject == ""].sort_values(["image_texture", "likes"], ascending=False)

    # ---------------- selection
    chosen = set(ok[ok.group == "demo"].uid)
    queues = {c: list(g.uid) for c, g in ok[ok.group != "demo"].groupby("category")}
    weight = {c: (d["related_weight"] if c in related else 1) for c in queues}
    order = sorted(queues)
    random.Random(cfg["seed"]).shuffle(order)
    while len(chosen) < d["target_total"] and any(queues.values()):
        for c in order:
            for _ in range(weight[c]):
                if queues[c] and len(chosen) < d["target_total"]:
                    chosen.add(queues[c].pop(0))

    # ---------------- splits
    other_cats = sorted(ok[(ok.group == "other") & ok.uid.isin(chosen)].category.unique())
    unseen = set(random.Random(cfg["seed"] + 1).sample(other_cats, min(d["unseen_holdout_categories"], len(other_cats))))
    sp = d["split"]
    df["selected"] = df.uid.isin(chosen)
    df["split"] = [("test_unseen" if c in unseen else split_of(u, sp["train"], sp["val"])) if s else ""
                   for u, c, s in zip(df.uid, df.category, df.selected)]

    out = project_path(d["meta_dir"]) / "candidates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    sel = df[df.selected]
    print("\n=== Stage-1 summary ===")
    print(f"LVIS objects considered: {len(df)} | pass filter: {len(ok)} | selected: {len(sel)}")
    print(sel.groupby("group").agg(objects=("uid", "size"), categories=("category", "nunique"),
                                   image_textured=("image_texture", "sum"),
                                   GB=("size_mb", lambda s: round(s.sum() / 1024, 2))).to_string())
    print("\nDemo categories:\n" + sel[sel.group == "demo"].category.value_counts().to_string())
    print("\nSplits:\n" + sel.split.value_counts().to_string())
    print(f"\nDownload size: {sel.size_mb.sum() / 1024:.2f} GB  ->  {out}")


if __name__ == "__main__":
    main()

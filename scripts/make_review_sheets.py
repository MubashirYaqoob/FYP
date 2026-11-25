"""Step 5: manual review. Builds numbered thumbnail grids per category + an editable CSV.

Input : data/processed/thumbs/<uid>.png   (from render_blender.py --mode thumb)
Output: outputs/review/<category>_<k>.png  (48 numbered tiles per sheet)
        data/meta/review.csv               (uid, category, number, keep=1, note)

How to review:
  1. open the sheets, note the numbers of bad objects (wrong object, scene, lying sideways, broken)
  2. in review.csv set keep=0 for those numbers (optional note: 'sideways', 'scene', 'not a chair' ...)
  3. training uses only keep=1
Re-running never overwrites decisions already in review.csv; new objects are appended.

Usage:  python scripts/make_review_sheets.py
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.utils.common import load_config, project_path  # noqa: E402

COLS, ROWS, TILE = 8, 6, 160


def main():
    cfg = load_config()
    meta = project_path(cfg["data"]["meta_dir"])
    thumbs = project_path(cfg["data"]["processed_dir"]) / "thumbs"
    out_dir = project_path("outputs/review")
    out_dir.mkdir(parents=True, exist_ok=True)

    checked = pd.read_csv(meta / "checked.csv", keep_default_na=False)
    have = {p.stem for p in thumbs.glob("*.png")}
    df = checked[checked.uid.isin(have)][["uid", "group", "category", "name", "reason"]]

    review_csv = meta / "review.csv"
    old = pd.read_csv(review_csv, keep_default_na=False) if review_csv.exists() else \
        pd.DataFrame(columns=["uid", "group", "category", "number", "keep", "note", "name"])
    new = df[~df.uid.isin(old.uid)].copy()
    new["keep"], new["note"] = 1, new.reason.replace("", None).fillna("")
    rev = pd.concat([old, new.drop(columns="reason")], ignore_index=True)
    rev["number"] = rev.groupby("group").cumcount() + 1
    rev[["uid", "group", "category", "number", "keep", "note", "name"]].to_csv(review_csv, index=False)

    per = COLS * ROWS
    for cat, g in rev.groupby("group"):   # one sheet series per group (demo / related / other)
        g = g.reset_index(drop=True)
        for s in range(0, len(g), per):
            sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), "white")
            draw = ImageDraw.Draw(sheet)
            for k, r in enumerate(g.iloc[s:s + per].itertuples()):
                x, y = (k % COLS) * TILE, (k // COLS) * TILE
                im = Image.open(thumbs / f"{r.uid}.png").convert("RGBA").resize((TILE - 8, TILE - 8))
                bg = Image.new("RGBA", im.size, "white")
                sheet.paste(Image.alpha_composite(bg, im).convert("RGB"), (x + 4, y + 4))
                colour = "red" if str(r.keep) == "0" else ("orange" if r.note else "black")
                draw.rectangle([x + 1, y + 1, x + TILE - 2, y + TILE - 2], outline="#dddddd")
                draw.text((x + 6, y + 4), str(r.number), fill=colour)
            path = out_dir / f"{cat}_{s // per + 1}.png"
            sheet.save(path)
            print(f"{path.name}: objects {s + 1}-{min(s + per, len(g))}")
    print(f"Edit keep=0 for bad objects in -> {review_csv}")


if __name__ == "__main__":
    main()

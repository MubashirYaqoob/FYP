"""PyTorch dataset for generator training: (photo, camera) -> (points, inside/outside, surface colours).

One item = one object, one random view (train) or a fixed view (val/test).
Points are returned in the VIEW frame (object rotated into the camera's orientation, see model.py),
so the object's original orientation in Objaverse does not matter.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_object_table(meta_dir, objects_dir, split=None):
    """Objects that are clean (preprocess flag empty), rendered, and not rejected in manual review."""
    meta_dir, objects_dir = Path(meta_dir), Path(objects_dir)
    pre = pd.read_csv(meta_dir / "preprocessed.csv", keep_default_na=False)
    cand = pd.read_csv(meta_dir / "candidates.csv", keep_default_na=False)
    df = pre[(pre.status == "ok") & (pre.flag == "")][["uid"]].merge(
        cand[["uid", "group", "category", "split"]], on="uid")
    review = meta_dir / "review.csv"
    if review.exists():
        rev = pd.read_csv(review, keep_default_na=False)
        df = df[~df.uid.isin(rev[rev.keep.astype(str) == "0"].uid)]
    df = df[df.uid.map(lambda u: (objects_dir / u / "views" / "cameras.json").exists())]
    if split:
        splits = [split] if isinstance(split, str) else list(split)
        df = df[df.split.isin(splits)]
    return df.reset_index(drop=True)


def image_to_tensor(rgba, background):
    """RGBA uint8 (H,W,4) -> normalised float tensor (3,H,W), composited on `background` (RGB 0..1)."""
    img = rgba.astype(np.float32) / 255.0
    rgb = img[..., :3] * img[..., 3:4] + np.asarray(background, np.float32) * (1 - img[..., 3:4])
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


class GEN3DDataset(Dataset):
    def __init__(self, table, objects_dir, cfg, train=True):
        self.table, self.dir, self.train = table, Path(objects_dir), train
        t = cfg["train"]
        self.n_points = t["points_per_sample"]
        self.n_surface = t["surface_points_per_sample"]
        self.random_bg = t.get("random_background", True) and train
        self.fixed_view = cfg["train"].get("val_view", 0)
        self._cams = {}

    def __len__(self):
        return len(self.table)

    def cameras(self, uid):
        if uid not in self._cams:
            self._cams[uid] = json.load(open(self.dir / uid / "views" / "cameras.json"))
        return self._cams[uid]

    def __getitem__(self, i):
        uid = self.table.uid[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        cams = self.cameras(uid)
        frame = cams["frames"][rng.integers(len(cams["frames"])) if self.train else self.fixed_view]

        # photo: white background, or a random plain colour while training (robust to real photos)
        bg = rng.uniform(0.6, 1.0, 3) if (self.random_bg and rng.random() < 0.5) else np.ones(3)
        rgba = np.asarray(Image.open(self.dir / uid / "views" / frame["file"]).convert("RGBA"))
        image = image_to_tensor(rgba, bg)

        # camera: world(normalised object) -> view rotation R and translation t
        w2c = np.linalg.inv(np.asarray(frame["c2w"], dtype=np.float64))
        R, t = w2c[:3, :3], w2c[:3, 3]

        z = np.load(self.dir / uid / "data.npz")
        pi = rng.integers(0, len(z["points"]), self.n_points)
        si = rng.integers(0, len(z["surf_points"]), self.n_surface)
        pts = z["points"][pi].astype(np.float32) @ R.T.astype(np.float32)
        surf = z["surf_points"][si].astype(np.float32) @ R.T.astype(np.float32)

        return {
            "image": image,
            "points": torch.from_numpy(pts),
            "occ": torch.from_numpy(z["occ"][pi].astype(np.float32)),
            "surf_points": torch.from_numpy(surf),
            "surf_rgb": torch.from_numpy(z["surf_colors"][si].astype(np.float32) / 255.0),
            "t": torch.tensor(t, dtype=torch.float32),
            "K": torch.tensor(cams["K"], dtype=torch.float32),
        }


def category_weights(table):
    """Sampling weight per object = 1 / sqrt(size of its category): keeps the demo categories
    boosted but stops chairs (largest category) from dominating every batch."""
    counts = table.category.map(table.category.value_counts())
    w = 1.0 / np.sqrt(counts.to_numpy(dtype=np.float64))
    return torch.from_numpy(w / w.sum())

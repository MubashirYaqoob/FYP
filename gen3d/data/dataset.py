"""PyTorch dataset for generator training: (photo, camera) -> (points, inside/outside, surface colours).

One item = one object, one random view (train) or a fixed view (val/test).
Views are read from  objects/<uid>/views/  (local renders) or  <views_zip_dir>/<uid>.zip  (Kaggle renders).
Points are returned in the VIEW frame (object rotated into the camera's orientation, see model.py),
so the object's original orientation in Objaverse does not matter.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def has_views(objects_dir, uid, views_zip_dir=None):
    if views_zip_dir and (Path(views_zip_dir) / f"{uid}.zip").exists():
        return True
    return (Path(objects_dir) / uid / "views" / "cameras.json").exists()


def load_object_table(meta_dir, objects_dir, split=None, views_zip_dir=None):
    """Objects that are clean (preprocess flag empty), rendered, and not rejected in manual review."""
    meta_dir, objects_dir = Path(meta_dir), Path(objects_dir)
    pre = pd.read_csv(meta_dir / "preprocessed.csv", keep_default_na=False)
    cand = pd.read_csv(meta_dir / "candidates.csv", keep_default_na=False)
    df = pre[(pre.status == "ok") & (pre.flag == "")][["uid"]].merge(
        cand[["uid", "group", "category", "split", "name"]], on="uid")
    review = meta_dir / "review.csv"
    if review.exists():
        rev = pd.read_csv(review, keep_default_na=False)
        df = df[~df.uid.isin(rev[rev.keep.astype(str) == "0"].uid)]
    df = df[df.uid.map(lambda u: has_views(objects_dir, u, views_zip_dir))]
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


def jitter_colours(rgba, strength, rng):
    """Randomly scale brightness, contrast and saturation by up to +-strength (alpha unchanged)."""
    rgb = rgba[..., :3].astype(np.float32)
    b, c, s = 1 + rng.uniform(-strength, strength, 3)
    rgb = rgb * b
    rgb = (rgb - rgb.mean()) * c + rgb.mean()
    grey = rgb.mean(-1, keepdims=True)
    rgb = (rgb - grey) * s + grey
    out = rgba.copy()
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


class GEN3DDataset(Dataset):
    def __init__(self, table, objects_dir, cfg, train=True, views_zip_dir=None):
        self.table, self.dir, self.train = table, Path(objects_dir), train
        self.zip_dir = Path(views_zip_dir) if views_zip_dir else None
        t = cfg["train"]
        self.n_points = t["points_per_sample"]
        self.n_surface = t["surface_points_per_sample"]
        self.random_bg = t.get("random_background", True) and train
        self.jitter = t.get("color_jitter", 0.0) if train else 0.0
        self.fixed_view = cfg["train"].get("val_view", 0)
        self.img_size = cfg["generator"]["encoder"]["img_size"]
        self._cams = {}

    def __len__(self):
        return len(self.table)

    def _zip(self, uid):
        if self.zip_dir and (self.zip_dir / f"{uid}.zip").exists():
            return self.zip_dir / f"{uid}.zip"
        return None

    def read_view_file(self, uid, name):
        """Bytes of views/<name> for this object, from the folder or from the object's zip."""
        zp = self._zip(uid)
        if zp:
            with zipfile.ZipFile(zp) as z:
                return z.read(name)
        return (self.dir / uid / "views" / name).read_bytes()

    def cameras(self, uid):
        if uid not in self._cams:
            self._cams[uid] = json.loads(self.read_view_file(uid, "cameras.json"))
        return self._cams[uid]

    def __getitem__(self, i):
        uid = self.table.uid[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        cams = self.cameras(uid)
        frame = cams["frames"][rng.integers(len(cams["frames"])) if self.train else self.fixed_view]

        # photo: white background, or a random plain colour while training (robust to real photos)
        bg = rng.uniform(0.6, 1.0, 3) if (self.random_bg and rng.random() < 0.5) else np.ones(3)
        pil = Image.open(io.BytesIO(self.read_view_file(uid, frame["file"]))).convert("RGBA")
        if pil.size != (self.img_size, self.img_size):   # e.g. ShapeNet/ONet renders are 137 x 137 JPEGs
            pil = pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        rgba = np.asarray(pil)
        if rgba[..., 3].min() == 255:   # JPEG without alpha: white background -> transparent (for random bg)
            from gen3d.generator.extract import background_mask
            rgba = rgba.copy()
            rgba[..., 3] = np.where(background_mask(rgba[..., :3], light=245, grey_tol=10), 0, 255)
        if self.jitter:  # brightness / contrast / saturation change, like different lighting in real photos
            rgba = jitter_colours(rgba, self.jitter, rng)
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

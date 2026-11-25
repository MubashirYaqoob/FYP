"""GEN3D generator: one photo -> occupancy (inside/outside) + colour for any 3D point.

    Generator = Encoder + pixel-aligned sampling + Decoder (occupancy MLP) + Texture field (colour MLP)

Coordinate convention (view-centric, must match gen3d/data/dataset.py):
  * points are given in the VIEW frame: the normalised object rotated so the camera looks along -Z
    (object centre stays at the origin, object size ~1)
  * the camera sits at  p_cam = p_view + t   (t = camera translation, about (0, 0, -2))
  * pixel:  u = fx * x / (-z) + cx ,  v = cy - fy * y / (-z)      (OpenGL camera, image rows go down)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import build_encoder


# ------------------------------------------------------------------ building blocks
class PositionalEncoding(nn.Module):
    """(x, y, z) -> [p, sin(2^k pi p), cos(2^k pi p)]  so the MLP can learn fine detail."""

    def __init__(self, n_freqs):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * math.pi, persistent=False)
        self.out_dim = 3 + 3 * 2 * n_freqs

    def forward(self, p):                                   # (B, N, 3)
        x = p[..., None] * self.freqs                       # (B, N, 3, F)
        enc = torch.cat([torch.sin(x), torch.cos(x)], -1).flatten(-2)
        return torch.cat([p, enc], -1)                      # (B, N, out_dim)


class ResBlock(nn.Module):
    """Fully connected residual block: x + fc(relu(fc(relu(x))))."""

    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(F.relu(x))))


class PointMLP(nn.Module):
    """Per-point MLP used for both the occupancy decoder and the texture field."""

    def __init__(self, in_dim, hidden, n_blocks, out_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.out(F.relu(self.blocks(self.inp(x))))


def project(points_view, t, K):
    """View-frame points -> pixel coordinates (u, v) and depth.  points (B,N,3), t (B,3), K (B,3,3)."""
    p = points_view + t[:, None, :]
    depth = (-p[..., 2]).clamp(min=1e-4)
    u = K[:, None, 0, 0] * p[..., 0] / depth + K[:, None, 0, 2]
    v = K[:, None, 1, 2] - K[:, None, 1, 1] * p[..., 1] / depth
    return u, v, depth


def sample_features(fmap, u, v, img_size):
    """Bilinear lookup of the feature map at pixel (u, v).  fmap (B,C,h,w) -> (B,N,C)."""
    grid = torch.stack([u / img_size * 2 - 1, v / img_size * 2 - 1], -1)[:, :, None, :]  # (B,N,1,2)
    feat = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return feat[..., 0].transpose(1, 2)


# ------------------------------------------------------------------ the generator
class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g = cfg["generator"]
        self.img_size = g["encoder"]["img_size"]
        self.pixel_aligned = g.get("pixel_aligned", True)
        self.encoder = build_encoder(g["encoder"])
        self.posenc = PositionalEncoding(g["decoder"]["pos_enc_freqs"])

        feat = g["encoder"]["out_dim"]
        in_dim = self.posenc.out_dim + feat + (feat if self.pixel_aligned else 0)
        self.decoder = PointMLP(in_dim, g["decoder"]["hidden"], g["decoder"]["layers"], 1)
        self.texture = PointMLP(in_dim, g["texture_field"]["hidden"], g["texture_field"]["layers"], 3)

    def encode(self, image):
        """Run the encoder once per image; reuse the result for many point queries."""
        return self.encoder(image)                          # global (B,C), fmap (B,C,h,w)

    def point_features(self, enc, points, t, K):
        g, fmap = enc
        parts = [self.posenc(points), g[:, None, :].expand(-1, points.shape[1], -1)]
        if self.pixel_aligned:
            u, v, _ = project(points, t, K)
            parts.append(sample_features(fmap, u, v, self.img_size))
        return torch.cat(parts, -1)

    def occupancy_logits(self, enc, points, t, K):
        return self.decoder(self.point_features(enc, points, t, K))[..., 0]      # (B,N)

    def colour(self, enc, points, t, K):
        return torch.sigmoid(self.texture(self.point_features(enc, points, t, K)))  # (B,N,3) in [0,1]

    def forward(self, image, points, surf_points, t, K):
        enc = self.encode(image)
        return self.occupancy_logits(enc, points, t, K), self.colour(enc, surf_points, t, K)


def generator_loss(occ_logits, occ_gt, rgb_pred, rgb_gt, lambda_tex):
    """BCE on inside/outside + L1 on surface colour."""
    l_occ = F.binary_cross_entropy_with_logits(occ_logits, occ_gt.float())
    l_tex = (rgb_pred - rgb_gt).abs().mean()
    return l_occ + lambda_tex * l_tex, {"occ": l_occ.item(), "tex": l_tex.item()}


@torch.no_grad()
def iou(occ_logits, occ_gt):
    """Intersection-over-union of predicted (p > 0.5) vs true inside points, per batch."""
    pred, gt = occ_logits > 0, occ_gt.bool()
    inter = (pred & gt).sum(1).float()
    union = (pred | gt).sum(1).float().clamp(min=1)
    return (inter / union).mean().item()

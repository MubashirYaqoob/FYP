"""GEN3D image encoder.

Final choice: DINOv2 ViT-S/14 (pretrained, fine-tuned by us).
Ablation baselines: ResNet-18 (ImageNet-pretrained) and ResNet-18 (from scratch).

Every encoder returns the same two things, so the decoder never cares which one is used:
    global_feat : (B, out_dim)          -> whole-image understanding (shape type, style, colour)
    feat_map    : (B, out_dim, H', W')  -> spatial features for pixel-aligned sampling (v2, PIFu/DISN style)
"""
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT-S/14 (21M params). 224x224 input -> 16x16 patch grid, 384-d features."""

    def __init__(self, out_dim=256, pretrained=True, freeze_blocks=8, img_size=224):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        self.grid = img_size // 14
        c = self.backbone.embed_dim  # 384

        # Low-data fine-tuning: freeze patch embedding + first `freeze_blocks` of 12 blocks.
        if pretrained and freeze_blocks > 0:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
            for blk in self.backbone.blocks[:freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False

        self.global_proj = nn.Linear(c, out_dim)
        self.map_proj = nn.Conv2d(c, out_dim, kernel_size=1)

    def forward(self, img):
        tokens = self.backbone.forward_features(img)          # (B, prefix + N, C)
        cls = tokens[:, 0]
        patches = tokens[:, self.n_prefix:]                    # (B, N, C)
        b, n, c = patches.shape
        fmap = patches.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return self.global_proj(cls), self.map_proj(fmap)


class ResNet18Encoder(nn.Module):
    """Ablation baseline. pretrained=False gives the 'encoder trained from scratch' experiment."""

    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                  net.layer1, net.layer2, net.layer3, net.layer4)
        self.global_proj = nn.Linear(512, out_dim)
        self.map_proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, img):
        fmap = self.stem(img)                                  # (B, 512, 7, 7)
        return self.global_proj(fmap.mean((2, 3))), self.map_proj(fmap)


def build_encoder(cfg):
    name = cfg["name"]
    if name == "dinov2_s14":
        return DINOv2Encoder(cfg["out_dim"], cfg.get("pretrained", True),
                             cfg.get("freeze_blocks", 8), cfg.get("img_size", 224))
    if name == "resnet18":
        return ResNet18Encoder(cfg["out_dim"], cfg.get("pretrained", True))
    raise ValueError(f"Unknown encoder: {name}")


if __name__ == "__main__":
    enc = DINOv2Encoder()
    g, m = enc(torch.randn(2, 3, 224, 224))
    trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    total = sum(p.numel() for p in enc.parameters())
    print("global:", tuple(g.shape), "map:", tuple(m.shape), f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M")

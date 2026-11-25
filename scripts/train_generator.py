"""Train the GEN3D generator (encoder + decoder + texture field), end to end.

Usage:
  python scripts/train_generator.py                         # full training (Kaggle GPU)
  python scripts/train_generator.py --overfit 4 --epochs 300  # sanity check: memorise 4 objects (laptop)
  python scripts/train_generator.py --resume                 # continue from outputs/checkpoints/last.pt

Every epoch: train loss; every `val_every` epochs: val loss + IoU, `last.pt` and `best.pt` (by val IoU).
Log: outputs/checkpoints/log.csv  (epoch, losses, IoU, time) -> loss curves for the report.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.data.dataset import GEN3DDataset, category_weights, load_object_table  # noqa: E402
from gen3d.generator.model import Generator, generator_loss, iou  # noqa: E402
from gen3d.utils.common import load_config, project_path  # noqa: E402


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates."""
    train = optimizer is not None
    model.train(train)
    totals, n = {"loss": 0.0, "occ": 0.0, "tex": 0.0, "iou": 0.0}, 0
    amp = device.type == "cuda" and cfg["train"].get("amp", True)
    for batch in loader:
        b = to_device(batch, device)
        with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=amp):
            occ_logits, rgb = model(b["image"], b["points"], b["surf_points"], b["t"], b["K"])
            loss, parts = generator_loss(occ_logits.float(), b["occ"], rgb.float(), b["surf_rgb"],
                                         cfg["train"]["lambda_tex"])
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        totals["loss"] += loss.item()
        totals["occ"] += parts["occ"]
        totals["tex"] += parts["tex"]
        totals["iou"] += iou(occ_logits, b["occ"])
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit", type=int, default=0, help="train AND validate on N objects (code check)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lr", type=float, default=None, help="override train.lr (e.g. higher for --overfit)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t = cfg["train"]
    if args.lr:
        t["lr"] = args.lr
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = project_path(cfg["data"]["meta_dir"])
    objects = project_path(cfg["data"]["processed_dir"]) / "objects"
    ckpt_dir = project_path(t["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data
    if args.overfit:
        train_tab = load_object_table(meta, objects).head(args.overfit)
        val_tab = train_tab
    else:
        train_tab = load_object_table(meta, objects, "train")
        val_tab = load_object_table(meta, objects, "val")
    train_ds = GEN3DDataset(train_tab, objects, cfg, train=True)
    val_ds = GEN3DDataset(val_tab, objects, cfg, train=False)
    sampler = WeightedRandomSampler(category_weights(train_tab), num_samples=len(train_tab), replacement=True)
    workers = t["num_workers"] if device.type == "cuda" else 0
    train_dl = DataLoader(train_ds, batch_size=min(t["batch_size"], len(train_ds)), sampler=sampler,
                          num_workers=workers, pin_memory=device.type == "cuda", drop_last=not args.overfit)
    val_dl = DataLoader(val_ds, batch_size=min(t["batch_size"], len(val_ds)), num_workers=workers)
    print(f"device={device} | train objects={len(train_ds)} | val objects={len(val_ds)}")

    # ---------------- model + optimiser (encoder learns 10x slower: it is already pretrained)
    model = Generator(cfg).to(device)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    optimizer = torch.optim.AdamW([{"params": other, "lr": t["lr"]},
                                   {"params": enc_params, "lr": t["encoder_lr"]}],
                                  weight_decay=t["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and t.get("amp", True))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train / 1e6:.1f} M")

    start, best = 1, -1.0
    if args.resume and (ckpt_dir / "last.pt").exists():
        ck = torch.load(ckpt_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start, best = ck["epoch"] + 1, ck.get("best_iou", -1.0)
        print(f"resumed from epoch {ck['epoch']} (best val IoU {best:.3f})")

    epochs = args.epochs or t["epochs"]
    val_every = 1 if args.overfit else t["val_every"]
    log_path = ckpt_dir / "log.csv"
    new_log = not log_path.exists() or not args.resume
    with open(log_path, "w" if new_log else "a", newline="") as f:
        log = csv.writer(f)
        if new_log:
            log.writerow(["epoch", "train_loss", "train_occ", "train_tex", "train_iou",
                          "val_loss", "val_occ", "val_tex", "val_iou", "seconds"])
        for epoch in range(start, epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_dl, device, cfg, optimizer, scaler)
            va = run_epoch(model, val_dl, device, cfg) if epoch % val_every == 0 else None
            row = [epoch, tr["loss"], tr["occ"], tr["tex"], tr["iou"]]
            row += [va["loss"], va["occ"], va["tex"], va["iou"]] if va else ["", "", "", ""]
            log.writerow(row + [round(time.time() - t0, 1)])
            f.flush()
            msg = f"epoch {epoch:4d} | train loss {tr['loss']:.4f} IoU {tr['iou']:.3f}"
            if va:
                msg += f" | val loss {va['loss']:.4f} IoU {va['iou']:.3f}"
            print(msg + f" | {time.time() - t0:.1f}s", flush=True)

            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_iou": best, "config": cfg}
            if va and va["iou"] > best:
                best = va["iou"]
                state["best_iou"] = best
                torch.save(state, ckpt_dir / "best.pt")
            if epoch % val_every == 0 or epoch == epochs:
                torch.save(state, ckpt_dir / "last.pt")
    print(f"done. best val IoU {best:.3f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

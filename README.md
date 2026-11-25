# GEN3D

2D image → our own trained 3D generator → textured mesh → learned simplification (D-QEM: QEM + Transformer) → **low-poly mesh with texture preserved**.

See [PLAN.md](PLAN.md) for the roadmap and FYP-1 deliverables.

## Structure
```
configs/          default.yaml (all settings)
scripts/          dataset download, filtering, rendering, preprocessing
gen3d/generator/  encoder (DINOv2 ViT-S/14), occupancy decoder, texture field, mesh extraction
gen3d/simplify/   QEM, edge features, labels, tokenizer, transformer, D-QEM
gen3d/eval/       metrics (IoU, Chamfer, F-score, PSNR/SSIM/LPIPS)
app/              Gradio demo
notebooks/        Colab/Kaggle training notebooks
data/, outputs/   git-ignored
```

## Setup (local, Windows)
Needs Python 3.11 (PyTorch does not support 3.15 yet).
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m gen3d.generator.encoder      # smoke test
```

## Setup (Colab / Kaggle, for training)
```
!git clone <repo-url> GEN && cd GEN && pip install -r requirements.txt
```

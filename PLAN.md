# GEN3D — Final Project Plan (FYP-1)

**Goal:** 2D image → apna trained 3D generator → textured 3D mesh → apna learned simplifier (QEM + Transformer = D-QEM) → **low-poly mesh with high texture preserved**.
















**Rule:** Generator aur Simplifier dono hum khud train karenge. Sirf 2D image encoder (ResNet-18 / DINOv2-small) ImageNet-pretrained se start hoga aur fine-tune hoga.

---

## 1. Final Decisions

| Item | Decision |
|---|---|
| Dataset | **Objaverse-LVIS** subset (4 categories) |
| Categories | chair, table, cup/mug, cupboard/cabinet/wardrobe (exact LVIS names script se verify karna) |
| **Training style (FINAL)** | **Class-agnostic**: one dataset, many categories, no category input → model learns 3D structure, not class |
| Data (selected 2026-11-25) | **8,000 objects / 28 GB from Objaverse-LVIS only** (no other dataset needed). demo 1,283 (chair 573, table 249, cupboard 248, cup 213) + related 913 (furniture, household, toys, vehicles; 32 cat.) + other 5,804 (993 cat., round-robin for variety). Expect ~7k after check + preprocess + review |
| Splits | train 6,257 / val 759 / test 752 / **test_unseen 232** (40 whole categories never seen in training → generalisation result) |
| Data pipeline | `build_candidates.py` (filter + selection + splits) → `download_objaverse.py` → `check_meshes.py` → `preprocess.py` → `render_blender.py` (thumb) → `make_review_sheets.py` → `render_blender.py` (views) |
| Generator change | class-agnostic + **pixel-aligned features in v1** (global feature alone generalises badly across ~1000 categories) |
| Generator | Image encoder → Occupancy decoder (MLP) → Marching Cubes → xatlas UV → Texture Field + image projection → baked texture |
| **Encoder (FINAL)** | **DINOv2 ViT-S/14** (21M, pretrained, first 8/12 blocks frozen, last 4 fine-tuned). Ablations: ResNet-18 pretrained, ResNet-18 from scratch. Code: `gen3d/generator/encoder.py` |
| Coordinate frame | **View-centric** (shape ko camera frame mein express karna) → Objaverse ki random orientation ka masla khatam |
| Simplifier | Apna attribute-aware QEM (position + UV, seam lock) + Edge-MLP tokenizer + Transformer encoder → residual cost |
| D-QEM cost | `QEM(e) + λ · Transformer(e)`  (λ = 0 → plain QEM, ablation ke liye) |
| Transformer label | `TrueError(collapse e) − QEM(e)`, TrueError = geometric + UV/texture distortion + seam penalty |
| Compute | Colab / Kaggle (T4/P100) |
| Demo | Gradio web app |

### Objaverse ke liye extra steps (zaroori)
1. **Filter:** texture/UV maujood ho, single object ho (scene nahi), faces 2k–500k, file load ho.
2. **Normalize:** center + unit cube scale.
3. **Upright check:** most GLBs Y-up hote hain; har object ka 1 thumbnail render → manual quick review (keep / reject / rotate).
4. **View-centric training:** har render ke liye target points ko camera frame mein transform karo — canonical front-facing ki zaroorat nahi.
5. **Watertight:** ManifoldPlus / mesh2sdf → occupancy points (100k per object, 50% near surface).
6. **Surface colors:** 50k surface points + RGB (texture field ke liye).
7. **Renders:** Blender (bpy), 12–24 views per object, white background, camera params save.

---

## 2. Roadmap (14 weeks)

| Week | Phase | Kaam | Output | Owner |
|---|---|---|---|---|
| 1 | Setup | Repo, env, Colab/Kaggle, lit review start (ONet, DISN, Texture Fields, QEM 97/98) | env ready | All |
| 1–3 | Dataset | Objaverse-LVIS download (4 cat), filter, normalize, upright review | clean meshes | A |
| 2–4 | Data prep | Watertight, occupancy points, surface colors, Blender renders (pehle **chair**) | `.npz` + renders | A |
| 2–5 | QEM | Apna QEM (priority queue, quadrics) → UV-aware → seam lock; PyMeshLab se compare | `qem.py` | B |
| 3–5 | Evaluation | Chamfer, Hausdorff, IoU, F-score, multi-view PSNR/SSIM/LPIPS, face count, time | `evaluate.py` | B |
| 4–7 | Generator v1 | Geometry (occupancy) — chair → phir table | checkpoint + meshes | A + C |
| 7–9 | Texture | Texture field + xatlas UV + visible-side projection + bake; cup, cupboard add | textured `.glb` | A |
| 6–9 | Features + Labels | Edge features (length, sharpness, dihedral, boundary, QEM error, UV seam, UV stretch, color variance, valence, curvature) + labels | edge dataset | B + C |
| 9–11 | Transformer v0 | Edge-MLP tokenizer → Transformer (4 layers, 128 dim, patch 256–512 edges) | checkpoint + loss curves | C |
| 11–12 | Integration | image → generator → D-QEM → low-poly textured | `pipeline.py` | All |
| 12–14 | Demo + Docs | Gradio app, report, slides, updated architecture diagram | presentation | All |

---

## 2b. 🎯 FYP-1 FINAL EXAM RUBRIC (50 marks) → what we must show
Source: "Evaluation Rubrics FYP I & II Combine.pdf", page 3. **All FYP-1 work is planned against this table.**

| CLO | Marks | Rubric asks for | GEN3D deliverable | Owner |
|---|---|---|---|---|
| **CLO3** Demo | **15** | Prototype demo, efficiency/efficacy, goals achieved, **technical difficulties resolved** | Gradio demo: photo → our generator → our QEM low-poly (side-by-side + metrics); numbers table; `docs/technical_issues_log.md` | All |
| **CLO4** PM + PPT | **14** | PPT + answers (6); **report for 40% modules**, formal language/dress, **updated timeline + execution plan** (8) | Slides, 40% report, updated Gantt/timeline, risk table | All |
| **CLO5** Teamwork | **12** | Equal work distribution in every phase; **≥1 supervisor meeting per week** | Task table with names per phase; weekly meeting log (`docs/meetings.md`) | All |
| **CLO2** Design | **9** | see breakdown below | | |

**CLO2 breakdown (9 marks, diagrams must be in BOTH ppt and report):**
| Item | Scale | GEN3D artefact |
|---|---|---|
| System analysis | 0–4 | Use-case diagram (User: upload image, generate 3D, set target poly %, compare, download), SRS (functional + non-functional), **test plan** (unit/integration/system; metrics), wireframes of Gradio UI |
| System design | 0–3 | **Layer/component diagram** (updated architecture) + **activity/DFD** of pipeline + sequence diagram (demo request) |
| DB representation | 0–1 | Data schema: `candidates/checked/preprocessed.csv` columns + per-object files (`data.npz`, `norm.json`, `cameras.json`) as an ER-style diagram |
| Data structure design | 0–1 | Mesh (V,F,UV), 256³ voxel grid, KD-tree, quadric 4×4 matrices, edge priority queue (min-heap), point/occupancy tensors |
| Algorithm design | 0–1 | Pseudocode: occupancy flood-fill, QEM edge collapse, training loop, marching cubes extraction |
| **Coding conventions** | **0–5** | Clear names, modular package (`gen3d/`), **no hard-coding (all in `configs/default.yaml`)**, docstrings/comments, suitable data structures, **few black-box built-ins: core algorithms (occupancy, QEM, metrics) are our own code**; libraries (PyMeshLab etc.) only as baselines |

**Working rules from this rubric**
1. Every algorithm we write → keep pseudocode + one diagram for the report.
2. Every problem we solve → add a line to `docs/technical_issues_log.md` (CLO3 "difficulties resolved").
3. Every supervisor meeting → add to `docs/meetings.md` (CLO5).
4. Split each phase (requirements, design, code, testing) across all members, and write the names down.

---

## 3. ⭐ FINAL DELIVERABLE — FYP-1 (40%) — Ye end mein dikhana hai

### Deliverables
1. **Apna trained image-to-3D generator** — chair + table pe achhi geometry, basic texture (cup/cupboard bonus).
2. **Generator evaluation** — IoU, Chamfer, F-score (SF3D sirf comparison baseline).
3. **Apna UV-aware QEM** jo texture preserve karta hai (seam lock).
4. **Feature extraction + label generation** complete; **Transformer v0** trained (loss curves).
5. **Evaluation table:** Original vs PyMeshLab QEM vs Our QEM vs D-QEM v0 @ 50% / 25% / 10% faces.
6. **End-to-end Gradio demo.**
7. **Docs:** SRS, lit review, updated architecture diagram, FYP-1 report + slides.

### Demo flow (presentation day)
```
[Upload chair.jpg] → rembg background removal
   → [Generate 3D]  → GEN3D generator → dense textured mesh (~100k faces) in 3D viewer
   → [Slider: target 10%] → [Simplify]
   → Side-by-side:  Original | QEM | D-QEM
   → Table: faces, Chamfer, SSIM/LPIPS, time
```

### FYP-2 (remaining 60%)
Pixel-aligned features (PIFu/DISN style), more data/categories, full Transformer training, texture re-baking (high→low poly), ablations (λ, features, layers), user study, UI polish, thesis/paper.

---

## 4. Risks & Backups

| Risk | Backup |
|---|---|
| Real photos pe kharab results | rembg + white-bg photos demo mein; render augmentation (lighting, blur) |
| Blurry texture | visible side pe input image projection |
| Cup data kam | augmentation, ya FYP-2 |
| GPU time kam | 64³ resolution + 1 category se start |
| Objaverse noisy | strict filtering + manual thumbnail review |

---

## 5. Architecture Diagram Fixes
- 3D Generation Layer: "Trained encoder/decoder" → **"GEN3D Generator: Image Encoder → Occupancy Decoder → Marching Cubes → UV + Texture Field (Trained)"**
- Feature list mein texture features add (UV seam, UV stretch, color variance).
- Stopping condition box → Mesh Simplification block ke andar.
- D-QEM ka full form likho: **Deep-QEM**.
- FYP-2: Output se pehle **Texture Re-bake** box.

---

## 6. Folder Structure
```
GEN/
  data/            # raw + processed (git-ignored)
  scripts/         # download, filter, render (Blender), preprocess
  gen3d/
    generator/     # encoder, occupancy decoder, texture field, mesh extraction
    simplify/      # qem.py, features.py, labels.py, tokenizer, transformer, dqem.py
    eval/          # metrics
  app/             # Gradio demo
  notebooks/       # Colab training notebooks
  PLAN.md
```

"""Train GEN3D-Gen v1 on Kaggle (GPU T4, internet ON for the DINOv2 weights).

Inputs : dataset `gen3d-code` (our code)
         notebook output `gen3d-data-pipeline`  (meta CSVs + objects/<uid>/data.npz)
         notebook output `gen3d-render`         (views/<uid>.zip, 16 photos + cameras per object)
Output : /kaggle/working/checkpoints/{best.pt,last.pt,log.csv}, train_summary.txt
"""
import glob
import os
import shutil
import subprocess
import time
import zipfile

EPOCHS = 300
TIME_BUDGET_H = 10.5   # stop training in time to save the output before Kaggle's 12 h limit
EXTRA_ARGS = ""        # e.g. "--overfit 8" for a quick check
INIT_FROM = ""         # "best.pt" -> start from the checkpoint found in the inputs (fine-tuning)
RUN_NAME = "GEN3D-Gen v1"
MIN_OBJECTS = 0        # safety: stop before training if fewer objects than this were found in the inputs
EVAL_ARGS = ""         # e.g. "--name-regex shapenet --limit 500": evaluate best.pt on the test split after training

T0 = time.time()
W = "/kaggle/working"


def sh(cmd):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    return subprocess.run(cmd, shell=True).returncode


# ---------------------------------------------------------------- code + inputs
GEN = "/tmp/GEN"
shutil.rmtree(GEN, ignore_errors=True)
zips = glob.glob("/kaggle/input/**/gen3d_code.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(GEN)
else:
    cfg = glob.glob("/kaggle/input/**/configs/default.yaml", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(cfg)), GEN)

# every attached dataset (Objaverse pipeline, ABO, ...) has meta/*.csv + processed/objects/<uid>/
# -> merge them into one meta folder and one folder of symlinks, so training sees one dataset
import pandas as pd  # noqa: E402

METAS = sorted(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/meta/preprocessed.csv", recursive=True))
META, OBJECTS = "/tmp/meta_all", "/tmp/objects_all"
os.makedirs(META, exist_ok=True)
os.makedirs(OBJECTS, exist_ok=True)
for name in ("candidates.csv", "preprocessed.csv", "review.csv"):
    parts = [pd.read_csv(f"{m}/{name}", keep_default_na=False) for m in METAS if os.path.exists(f"{m}/{name}")]
    if parts:
        pd.concat(parts, ignore_index=True).drop_duplicates("uid", keep="last").to_csv(f"{META}/{name}", index=False)
for m in METAS:   # object folders live in processed/objects (Objaverse, ABO) or objects/ (ShapeNet import)
    for src in (os.path.join(os.path.dirname(m), "processed", "objects"), os.path.join(os.path.dirname(m), "objects")):
        for d in os.listdir(src) if os.path.isdir(src) else []:
            if not os.path.exists(f"{OBJECTS}/{d}"):
                os.symlink(f"{src}/{d}", f"{OBJECTS}/{d}")
VIEWS = sorted({os.path.dirname(z) for z in glob.glob("/kaggle/input/**/views/*.zip", recursive=True)})
print("datasets:", METAS, "| objects:", len(os.listdir(OBJECTS)), "| views folders:", VIEWS)
assert len(os.listdir(OBJECTS)) >= MIN_OBJECTS, f"only {len(os.listdir(OBJECTS))} objects found, expected {MIN_OBJECTS}"

# all render parts -> one folder of symlinks, so the dataset sees every rendered object
VZ = "/tmp/views_all"
os.makedirs(VZ, exist_ok=True)
for d in VIEWS:
    for z in glob.glob(f"{d}/*.zip"):
        dst = f"{VZ}/{os.path.basename(z)}"
        if not os.path.exists(dst):
            os.symlink(z, dst)
print("rendered objects available:", len(os.listdir(VZ)))

sh("pip install -q timm trimesh scikit-image")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2")

# ---------------------------------------------------------------- train
os.chdir(GEN)
ck = f"{W}/checkpoints"
init = ""
if INIT_FROM:
    found = glob.glob(f"/kaggle/input/**/checkpoints/{INIT_FROM}", recursive=True)
    print("init checkpoint:", found)
    init = f"--init {found[0]}"
rc = sh(f"python scripts/train_generator.py {init} --epochs {EPOCHS} --max-hours {TIME_BUDGET_H} "
        f"--meta-dir {META} --objects-dir {OBJECTS} --views-zip-dir {VZ} --ckpt-dir {ck} {EXTRA_ARGS}")

# ---------------------------------------------------------------- evaluation (ONet protocol)
if EVAL_ARGS and os.path.exists(f"{ck}/best.pt"):
    sh(f"python scripts/evaluate.py --ckpt {ck}/best.pt --meta-dir {META} --objects-dir {OBJECTS} "
       f"--views-zip-dir {VZ} --out {W}/eval_test.csv {EVAL_ARGS}")

# ---------------------------------------------------------------- summary

lines = [RUN_NAME, f"exit code: {rc}", f"runtime: {(time.time() - T0) / 3600:.2f} h"]
if os.path.exists(f"{ck}/log.csv"):
    log = pd.read_csv(f"{ck}/log.csv")
    val = log.dropna(subset=["val_iou"])
    lines.append(f"epochs trained: {int(log.epoch.max())}")
    lines.append(f"last train loss {log.train_loss.iloc[-1]:.4f}  IoU {log.train_iou.iloc[-1]:.3f}")
    if len(val):
        best = val.loc[val.val_iou.idxmax()]
        lines.append(f"best val IoU {best.val_iou:.3f} at epoch {int(best.epoch)} (val loss {best.val_loss:.4f})")
print(f"\n=== {RUN_NAME} TRAINING SUMMARY ===\n" + "\n".join(lines))
open(f"{W}/train_summary.txt", "w").write("\n".join(lines) + "\n")

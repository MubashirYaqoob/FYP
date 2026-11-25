"""GEN3D data pipeline on Kaggle (CPU, internet ON).

download (Objaverse) -> check_meshes -> preprocess  ->  /kaggle/working/{meta,processed}
The output of this notebook is the training dataset for the generator.
Raw GLBs live in /tmp (not saved). Code comes from the private dataset `gen3d-code`.
"""
import glob
import os
import shutil
import subprocess
import time
import zipfile

TEST_LIMIT = 0        # >0: only this many objects per group (quick test run). 0 = full run
TIME_BUDGET_H = 11.0  # Kaggle stops sessions at 12 h; leave time to save the output

T0 = time.time()
os.environ["TQDM_MININTERVAL"] = "120"   # progress line every 2 min instead of flooding the log


def sh(cmd):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    print(f"[exit {rc}]", flush=True)
    return rc


# ---------------------------------------------------------------- code
GEN = "/tmp/GEN"
shutil.rmtree(GEN, ignore_errors=True)
zips = glob.glob("/kaggle/input/**/gen3d_code.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(GEN)
else:  # Kaggle may have auto-extracted the zip
    cfg = glob.glob("/kaggle/input/**/configs/default.yaml", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(cfg)), GEN)
print("code:", sorted(os.listdir(GEN)))

# ---------------------------------------------------------------- data layout
W = "/kaggle/working"
os.makedirs(f"{GEN}/data", exist_ok=True)
for name in ("meta", "processed"):
    os.makedirs(f"{W}/{name}", exist_ok=True)
    link = f"{GEN}/data/{name}"
    if os.path.lexists(link):
        shutil.rmtree(link)
    os.symlink(f"{W}/{name}", link)
if not os.path.exists(f"{W}/meta/candidates.csv"):
    shutil.copy(f"{GEN}/data_seed/candidates.csv", f"{W}/meta/")

sh("pip install -q objaverse trimesh pyyaml scikit-image")
sh("nproc; free -g | head -2; df -h /tmp /kaggle/working")
os.chdir(GEN)
N = os.cpu_count()

# ---------------------------------------------------------------- pipeline (chunked)
# Each chunk: download CHUNK new objects -> check -> preprocess -> delete the raw GLBs.
# Keeps /tmp small (~5 GB) and every finished chunk is already safe in /kaggle/working.
CHUNK = 1500
RAW_GLBS = f"{GEN}/data/raw/objaverse/hf-objaverse-v1/glbs"
limit = f"--limit {TEST_LIMIT}" if TEST_LIMIT else ""
for chunk in range(1, 100):
    left = TIME_BUDGET_H - (time.time() - T0) / 3600
    if left < 0.5:
        print("[time budget used] stopping; re-run this notebook to continue")
        break
    print(f"\n######## chunk {chunk}  ({left:.1f} h left) ########", flush=True)
    sh(f"python scripts/download_objaverse.py --workers 32 --max-new {CHUNK} {limit} 2>&1 | tail -3")
    if not glob.glob(f"{RAW_GLBS}/*/*.glb"):
        print("nothing left to download -> done")
        break
    sh(f"python scripts/check_meshes.py --workers {N} 2>&1 | tail -12")
    left = TIME_BUDGET_H - (time.time() - T0) / 3600
    sh(f"python scripts/preprocess.py --workers {N} --max-hours {max(0.25, left - 0.3):.2f} 2>&1 | tail -8")
    shutil.rmtree(RAW_GLBS, ignore_errors=True)
    sh("df -h / /tmp | tail -2")
    if TEST_LIMIT:
        break

# ---------------------------------------------------------------- summary
import pandas as pd  # noqa: E402

m = f"{W}/meta/"
cand = pd.read_csv(m + "candidates.csv", keep_default_na=False)
chk = pd.read_csv(m + "checked.csv", keep_default_na=False)
pre = pd.read_csv(m + "preprocessed.csv", keep_default_na=False)
sel = cand[cand.selected.astype(str) == "True"]
ok = pre[(pre.status == "ok") & (pre.flag == "")].merge(sel[["uid", "group", "category", "split"]], on="uid")
summary = [
    f"selected {len(sel)} | downloaded+checked {len(chk)} | PASS {(chk.status == 'PASS').sum()} "
    f"| preprocessed ok {(pre.status == 'ok').sum()} | CLEAN {len(ok)}",
    "flags: " + str(pre[pre.flag != ""].flag.value_counts().to_dict()),
    "errors: " + str(pre[pre.status.str.startswith("error")].status.str.split(":").str[1].value_counts().to_dict()),
    ok.groupby("group").agg(objects=("uid", "size"), categories=("category", "nunique")).to_string(),
    ok.split.value_counts().to_string(),
    f"total categories (clean): {ok.category.nunique()}",
    f"runtime: {(time.time() - T0) / 3600:.2f} h",
]
print("\n=== GEN3D DATA SUMMARY ===\n" + "\n".join(summary))
open(f"{W}/summary.txt", "w").write("\n".join(summary) + "\n")

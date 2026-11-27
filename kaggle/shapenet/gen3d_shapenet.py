"""GEN3D: ShapeNet as prepared by Occupancy Networks (CVPR 2019) -> our format.  Kaggle CPU, internet ON.

Downloads only the wanted categories from the ONet zip (byte range, no 73 GB download), converts them
(scripts/import_shapenet_onet.py) and saves  objects/, views/, meta/  as this notebook's output.
Same layout as the other data notebooks, so training reads it directly.
~1.15 MB per object (chairs: 6,778 objects -> ~7.8 GB). Kaggle keeps up to ~20 GB of output.
"""
import glob
import os
import shutil
import subprocess
import time
import zipfile

SYNSETS = [("03001627", "chair")]   # add ("04256520", "sofa"), ("04379243", "table") ... if space allows
TEST_LIMIT = 0                      # >0: convert only N objects per split (quick check)

T0 = time.time()
W = "/kaggle/working"


def sh(cmd):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    return subprocess.run(cmd, shell=True).returncode


GEN = "/tmp/GEN"
shutil.rmtree(GEN, ignore_errors=True)
zips = glob.glob("/kaggle/input/**/gen3d_code.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(GEN)
else:
    cfg = glob.glob("/kaggle/input/**/configs/default.yaml", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(cfg)), GEN)
os.chdir(GEN)
sh("pip install -q remotezip pandas")
sh("nproc; free -g | head -2; df -h /tmp /kaggle/working | tail -2")

for synset, cat in SYNSETS:
    src = f"/tmp/onet_{synset}"
    test = "--test 3000" if TEST_LIMIT else ""
    if sh(f"python scripts/fetch_onet_category.py --synset {synset} --dest {src} {test}") != 0:
        print(f"download failed for {synset}")
        continue
    lim = f"--limit {TEST_LIMIT}" if TEST_LIMIT else ""
    sh(f"python scripts/import_shapenet_onet.py --src {src} --synset {synset} --category {cat} "
       f"--out {W} --workers {os.cpu_count()} {lim}")
    shutil.rmtree(src, ignore_errors=True)
    sh(f"du -sh {W}/objects {W}/views; df -h /tmp | tail -1")

import pandas as pd  # noqa: E402

pre = pd.read_csv(f"{W}/meta/preprocessed.csv", keep_default_na=False)
cand = pd.read_csv(f"{W}/meta/candidates.csv", keep_default_na=False)
ok = pre[pre.status == "ok"].merge(cand[["uid", "category", "split"]], on="uid")
summary = ["ShapeNet (ONet) import", ok.groupby(["category", "split"]).size().unstack(fill_value=0).to_string(),
           "errors: " + str(pre[pre.status != "ok"].status.value_counts().to_dict()),
           f"runtime: {(time.time() - T0) / 3600:.2f} h"]
print("\n=== SUMMARY ===\n" + "\n".join(summary))
open(f"{W}/shapenet_summary.txt", "w").write("\n".join(summary) + "\n")

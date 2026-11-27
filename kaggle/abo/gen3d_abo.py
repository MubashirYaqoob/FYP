"""GEN3D second dataset on Kaggle: ABO (Amazon Berkeley Objects) furniture (5 categories).

One GPU notebook does the whole pipeline for ABO, chunk by chunk (each GLB is downloaded once):
  download GLB -> check_meshes -> preprocess (our occupancy labels) -> Blender render -> zip views
Output (/kaggle/working): meta/{candidates,downloaded,checked,preprocessed}.csv,
                          processed/objects/<uid>/{data.npz,norm.json}, views/<uid>.zip, abo_summary.txt
Same output layout as gen3d-data-pipeline + gen3d-render, so training can read both datasets.
"""
import glob
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

TEST_LIMIT = 0         # >0: only this many objects (quick test). 0 = everything
VIEWS = 12
NAME_INCLUDE = ""     # e.g. "dining" for dining chairs only
NAME_EXCLUDE = ""
TYPES = ""            # ABO product types for this notebook, e.g. "CHAIR,STOOL_SEATING" ("" = all)
SAMPLES = 16
CHUNK = 150            # ABO GLBs are big (~40 MB, 4K textures): ~6 GB of /tmp per chunk
TIME_BUDGET_H = 11.0
BLENDER_URLS = ["https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz",
                "https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz"]

T0 = time.time()
W = "/kaggle/working"


def sh(cmd):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def hours():
    return (time.time() - T0) / 3600


# ---------------------------------------------------------------- code + data layout
GEN = "/tmp/GEN"
shutil.rmtree(GEN, ignore_errors=True)
zips = glob.glob("/kaggle/input/**/gen3d_code.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(GEN)
else:
    cfg = glob.glob("/kaggle/input/**/configs/default.yaml", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(cfg)), GEN)
os.makedirs(f"{GEN}/data", exist_ok=True)
for name in ("meta", "processed"):
    os.makedirs(f"{W}/{name}", exist_ok=True)
    link = f"{GEN}/data/{name}"
    if os.path.lexists(link):
        shutil.rmtree(link)
    os.symlink(f"{W}/{name}", link)
OUT = f"{W}/views"
os.makedirs(OUT, exist_ok=True)
RAW = "/tmp/abo_glbs"
RENDER = "/tmp/abo_render"

sh("pip install -q trimesh pyyaml scikit-image")
sh("apt-get -qq update > /dev/null && apt-get -qq install -y libxrender1 libxi6 libxkbcommon0 libsm6 "
   "libgl1 libxxf86vm1 libxfixes3 libegl1 > /dev/null")
BLENDER = None
for url in BLENDER_URLS:
    if sh(f"mkdir -p /tmp/blender && curl -sL {url} | tar -xJ -C /tmp/blender --strip-components=1") == 0 \
            and os.path.exists("/tmp/blender/blender"):
        BLENDER = "/tmp/blender/blender"
        break
assert BLENDER, "Blender download failed"
sh(f"{BLENDER} --version | head -1; nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /tmp")
N_GPU = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout or 1))
PROCS = 2 * N_GPU
N_CPU = os.cpu_count()
os.chdir(GEN)

# ---------------------------------------------------------------- candidates
sh(f'python scripts/build_abo_candidates.py --meta-dir {W}/meta --name-include "{NAME_INCLUDE}" '
   f'--name-exclude "{NAME_EXCLUDE}" --types "{TYPES}"' + (f" --limit {TEST_LIMIT}" if TEST_LIMIT else ""))
import pandas as pd  # noqa: E402

cand = pd.read_csv(f"{W}/meta/candidates.csv", keep_default_na=False)
cand["order"] = cand.category.map({"chair": 0, "table": 1, "sofa": 2, "stool": 3}).fillna(4)
cand = cand.sort_values(["order", "uid"]).reset_index(drop=True)
# skip objects an earlier ABO notebook already finished (attached as input)
done_before = {os.path.basename(z)[:-4] for z in glob.glob("/kaggle/input/**/views/abo_*.zip", recursive=True)}
cand = cand[~cand.uid.isin(done_before)].reset_index(drop=True)
print(f"already done in earlier ABO runs: {len(done_before)}")
print(cand.category.value_counts().to_string())


def fetch(row):
    local = f"{RAW}/{row.uid}.glb"
    try:
        urllib.request.urlretrieve(row.url, local + ".tmp")
        os.replace(local + ".tmp", local)
        return row.uid, local
    except Exception:  # noqa: BLE001
        return row.uid, ""


def run_shard(p, shard):
    """One Blender worker; restart after a crash, skipping only the object it died on."""
    crashed = []
    todo = shard
    if not len(todo):
        return crashed
    for _ in range(25):
        todo = todo[~todo.uid.map(lambda u: os.path.exists(f"{RENDER}/{u}/views/cameras.json"))]
        todo = todo[~todo.uid.isin(crashed)]
        if not len(todo):
            break
        lst = f"/tmp/abo_list_{p}.csv"
        todo[["uid", "path", "category"]].to_csv(lst, index=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p % N_GPU))
        cmd = [BLENDER, "-b", "-P", f"{GEN}/scripts/render_blender.py", "--", "--mode", "views",
               "--engine", "cycles", "--samples", str(SAMPLES), "--views", str(VIEWS),
               "--list", lst, "--norm-dir", f"{W}/processed/objects", "--out-dir", RENDER]
        rc = subprocess.run(cmd, env=env, stdout=open(f"/tmp/abo_blender_{p}.log", "a"),
                            stderr=subprocess.STDOUT).returncode
        left = todo[~todo.uid.map(lambda u: os.path.exists(f"{RENDER}/{u}/views/cameras.json"))]
        if rc == 0 or not len(left):
            break
        crashed.append(left.uid.iloc[0])
        print(f"  worker {p}: Blender exit {rc}, skipping {left.uid.iloc[0]}", flush=True)
    return crashed


# ---------------------------------------------------------------- chunks
log_rows = []
for start in range(0, len(cand), CHUNK):
    if hours() > TIME_BUDGET_H - 0.7:
        print("[time budget used] stopping; finished objects are already in the output")
        break
    chunk = cand.iloc[start:start + CHUNK]
    chunk = chunk[~chunk.uid.map(lambda u: os.path.exists(f"{OUT}/{u}.zip"))]
    if not len(chunk):
        continue
    t_chunk = time.time()
    shutil.rmtree(RAW, ignore_errors=True)
    shutil.rmtree(RENDER, ignore_errors=True)
    os.makedirs(RAW, exist_ok=True)

    # 1. download
    with ThreadPoolExecutor(16) as ex:
        paths = dict(ex.map(fetch, chunk.itertuples()))
    dl = chunk.drop(columns=["order"]).assign(path=chunk.uid.map(paths))
    dl = dl[dl.path != ""]
    old = f"{W}/meta/downloaded.csv"
    prev = pd.read_csv(old, keep_default_na=False) if os.path.exists(old) else dl.iloc[:0]
    pd.concat([prev[~prev.uid.isin(dl.uid)], dl]).to_csv(old, index=False)

    # 2. check + 3. our occupancy labels (both incremental: only the new objects)
    # a very big model can run out of memory and kill a worker (run 2 lost whole chunks this way):
    # retry with fewer workers; objects that still fail are simply left out
    for w in (N_CPU, 2, 1):
        if sh(f"python scripts/check_meshes.py --workers {w} > /tmp/step.log 2>&1; rc=$?; tail -4 /tmp/step.log; exit $rc") == 0:
            break
    for w in (N_CPU, 2, 1):
        if sh(f"python scripts/preprocess.py --workers {w} > /tmp/step.log 2>&1; rc=$?; tail -4 /tmp/step.log; exit $rc") == 0:
            break

    # 4. render the clean ones
    pre_f = f"{W}/meta/preprocessed.csv"
    pre = pd.read_csv(pre_f, keep_default_na=False) if os.path.exists(pre_f) else pd.DataFrame(columns=["uid", "status", "flag"])
    clean = set(pre[(pre.status == "ok") & (pre.flag == "")].uid)
    ready = dl[dl.uid.isin(clean)][["uid", "path", "category"]].reset_index(drop=True)
    crashed = set()
    if len(ready):
      with ThreadPoolExecutor(PROCS) as ex:
        crashed = {u for lst in ex.map(lambda p: run_shard(p, ready.iloc[p::PROCS]), range(PROCS)) for u in lst}

    # 5. zip views into the output
    for uid in chunk.uid:
        vdir = f"{RENDER}/{uid}/views"
        ok = os.path.exists(f"{vdir}/cameras.json")
        if ok:
            with zipfile.ZipFile(f"{OUT}/{uid}.zip", "w", zipfile.ZIP_STORED) as z:
                for fn in sorted(os.listdir(vdir)):
                    z.write(f"{vdir}/{fn}", fn)
        status = ("ok" if ok else "no_download" if not paths.get(uid) else "not_clean" if uid not in clean
                  else "crash" if uid in crashed else "failed")
        log_rows.append({"uid": uid, "status": status})
    pd.DataFrame(log_rows).to_csv(f"{W}/abo_log.csv", index=False)
    n_ok = sum(r["status"] == "ok" for r in log_rows)
    dt = time.time() - t_chunk
    print(f"chunk {start // CHUNK + 1}: {len(chunk)} objects in {dt / 60:.1f} min "
          f"({dt / len(chunk):.1f} s/object) | total rendered {n_ok} | {hours():.2f} h", flush=True)

shutil.rmtree(RAW, ignore_errors=True)
shutil.rmtree(RENDER, ignore_errors=True)

# ---------------------------------------------------------------- summary
log = pd.DataFrame(log_rows)
ok = log[log.status == "ok"].merge(cand[["uid", "category", "split"]], on="uid") if len(log) else log
summary = [f"ABO objects: {len(cand)} | processed this run: {len(log)}",
           "status: " + (str(log.status.value_counts().to_dict()) if len(log) else "{}"),
           "rendered by category/split:\n" + (ok.groupby(["category", "split"]).size().unstack(fill_value=0).to_string()
                                              if len(ok) else "-"),
           f"views {VIEWS} | samples {SAMPLES} | runtime {hours():.2f} h"]
print("\n=== GEN3D ABO SUMMARY ===\n" + "\n".join(summary))
open(f"{W}/abo_summary.txt", "w").write("\n".join(summary) + "\n")

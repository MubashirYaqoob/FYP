"""GEN3D rendering on Kaggle (GPU, internet ON).

Inputs : dataset `gen3d-code` (our code) + output of notebook `gen3d-data-pipeline` (meta CSVs, norm.json)
Output : /kaggle/working/views/<uid>.zip   (16 PNG views + cameras.json per object, stored uncompressed)
         /kaggle/working/render_log.csv, render_summary.txt

Per chunk: download the GLBs again -> render with Blender (Cycles, GPU, several processes in
parallel) -> zip each object's views into the output -> delete the GLBs. Demo objects go first.
"""
import glob
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

TEST_LIMIT = 0         # >0: render only this many objects (speed test). 0 = everything
VIEWS = 12             # run 1 used 16; 12 random views are plenty (training picks one per sample)
SAMPLES = 16           # Cycles samples per view (denoised); 224 px needs few
CHUNK = 400            # objects downloaded + rendered per round
TIME_BUDGET_H = 11.3   # Kaggle stops sessions at 12 h; leave time to save the output
BLENDER_URLS = ["https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz",
                "https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz"]
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"

T0 = time.time()
W = "/kaggle/working"
OUT = f"{W}/views"
TMP = "/tmp/gen3d"


def sh(cmd):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def hours():
    return (time.time() - T0) / 3600


# ---------------------------------------------------------------- code + inputs
GEN = "/tmp/GEN"
shutil.rmtree(GEN, ignore_errors=True)
zips = glob.glob("/kaggle/input/**/gen3d_code.zip", recursive=True)
if zips:
    zipfile.ZipFile(zips[0]).extractall(GEN)
else:
    cfg = glob.glob("/kaggle/input/**/configs/default.yaml", recursive=True)[0]
    shutil.copytree(os.path.dirname(os.path.dirname(cfg)), GEN)
PIPE = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/meta/preprocessed.csv", recursive=True)[0]))
NORM_DIR = f"{PIPE}/processed/objects"
print("code:", GEN, "| pipeline output:", PIPE)

# ---------------------------------------------------------------- Blender + GPU
sh("apt-get -qq update > /dev/null && apt-get -qq install -y libxrender1 libxi6 libxkbcommon0 libsm6 "
   "libgl1 libxxf86vm1 libxfixes3 libegl1 > /dev/null")
BLENDER = None
for url in BLENDER_URLS:
    if sh(f"mkdir -p /tmp/blender && curl -sL {url} | tar -xJ -C /tmp/blender --strip-components=1") == 0 \
            and os.path.exists("/tmp/blender/blender"):
        BLENDER = "/tmp/blender/blender"
        break
assert BLENDER, "Blender download failed"
sh(f"{BLENDER} --version | head -1; nvidia-smi --query-gpu=name,memory.total --format=csv; nproc")
N_GPU = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout or 1))
PROCS = 2 * N_GPU      # two Blender processes per GPU: one imports while the other renders

# ---------------------------------------------------------------- object list (demo first)
pre = pd.read_csv(f"{PIPE}/meta/preprocessed.csv", keep_default_na=False)
cand = pd.read_csv(f"{PIPE}/meta/candidates.csv", keep_default_na=False)
objs = pre[(pre.status == "ok") & (pre.flag == "")][["uid"]].merge(cand[["uid", "group", "category"]], on="uid")
objs["order"] = objs.group.map({"demo": 0, "related": 1, "other": 2})
objs = objs.sort_values(["order", "uid"]).reset_index(drop=True)
if TEST_LIMIT:
    objs = objs.head(TEST_LIMIT)
os.makedirs(OUT, exist_ok=True)
# skip objects already rendered here OR in an attached earlier render run (/kaggle/input/**/views/*.zip)
done = {os.path.basename(z)[:-4] for z in glob.glob("/kaggle/input/**/views/*.zip", recursive=True)}
done |= {os.path.basename(z)[:-4] for z in glob.glob(f"{OUT}/*.zip")}
print(f"already rendered in earlier runs: {len(done)}")
objs = objs[~objs.uid.isin(done)]
print(f"objects to render: {len(objs)}  | GPUs {N_GPU} | Blender processes {PROCS}")

sh("pip install -q objaverse")
import objaverse  # noqa: E402

rel_paths = objaverse._load_object_paths()


def fetch(uid):
    local = f"{TMP}/glbs/{uid}.glb"
    try:
        urllib.request.urlretrieve(HF + rel_paths[uid], local)
        return uid, local
    except Exception:  # noqa: BLE001
        return uid, ""


log_rows = []
for start in range(0, len(objs), CHUNK):
    if hours() > TIME_BUDGET_H - 0.6:
        print("[time budget used] stopping; rendered objects are already in the output")
        break
    chunk = objs.iloc[start:start + CHUNK]
    t_chunk = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(f"{TMP}/glbs", exist_ok=True)
    with ThreadPoolExecutor(32) as ex:
        paths = dict(ex.map(fetch, chunk.uid))
    chunk = chunk.assign(path=chunk.uid.map(paths))
    ready = chunk[chunk.path != ""]

    # split into shards, one Blender worker each. If Blender crashes (run 1 lost 80 objects this way:
    # one heavy model killed the process and the rest of its shard), restart it on the objects that
    # are left, skipping only the object it was working on.
    def run_shard(p, shard):
        crashed = []
        todo = shard
        for attempt in range(25):
            todo = todo[~todo.uid.map(lambda u: os.path.exists(f"{TMP}/render/{u}/views/cameras.json"))]
            todo = todo[~todo.uid.isin(crashed)]
            if not len(todo):
                break
            lst = f"{TMP}/list_{p}.csv"
            todo[["uid", "path", "category"]].to_csv(lst, index=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p % N_GPU))
            cmd = [BLENDER, "-b", "-P", f"{GEN}/scripts/render_blender.py", "--", "--mode", "views",
                   "--engine", "cycles", "--samples", str(SAMPLES), "--views", str(VIEWS),
                   "--list", lst, "--norm-dir", NORM_DIR, "--out-dir", f"{TMP}/render"]
            rc = subprocess.run(cmd, env=env, stdout=open(f"{TMP}/blender_{p}.log", "a"),
                                stderr=subprocess.STDOUT).returncode
            left = todo[~todo.uid.map(lambda u: os.path.exists(f"{TMP}/render/{u}/views/cameras.json"))]
            if rc == 0 or not len(left):
                break
            crashed.append(left.uid.iloc[0])       # the object Blender died on
            print(f"  worker {p}: Blender exit {rc}, skipping {left.uid.iloc[0]} and restarting", flush=True)
        return crashed

    with ThreadPoolExecutor(PROCS) as ex:
        crashed_lists = list(ex.map(lambda p: run_shard(p, ready.iloc[p::PROCS]), range(PROCS)))
    crashed = {u for lst in crashed_lists for u in lst}

    # zip each finished object into the output, record status
    for uid in chunk.uid:
        vdir = f"{TMP}/render/{uid}/views"
        ok = os.path.exists(f"{vdir}/cameras.json")
        if ok:
            with zipfile.ZipFile(f"{OUT}/{uid}.zip", "w", zipfile.ZIP_STORED) as z:
                for fn in sorted(os.listdir(vdir)):
                    z.write(f"{vdir}/{fn}", fn)
        status = "ok" if ok else ("no_download" if not paths.get(uid) else ("crash" if uid in crashed else "failed"))
        log_rows.append({"uid": uid, "status": status})
    pd.DataFrame(log_rows).to_csv(f"{W}/render_log.csv", index=False)
    n_ok = sum(r["status"] == "ok" for r in log_rows)
    dt = time.time() - t_chunk
    print(f"chunk {start // CHUNK + 1}: {len(chunk)} objects in {dt / 60:.1f} min "
          f"({dt / max(len(chunk), 1):.1f} s/object) | total ok {n_ok} | {hours():.2f} h", flush=True)
    for lg in sorted(glob.glob(f"{TMP}/blender_*.log"))[:1]:
        sh(f"grep -E 'FAILED|Error|error' {lg} | head -5")

# ---------------------------------------------------------------- summary
log = pd.DataFrame(log_rows)
summary = [f"rendered ok: {(log.status == 'ok').sum() if len(log) else 0} / {len(log)}",
           "status: " + (str(log.status.value_counts().to_dict()) if len(log) else "{}"),
           f"views per object: {VIEWS} | samples: {SAMPLES} | GPUs: {N_GPU} | processes: {PROCS}",
           f"runtime: {hours():.2f} h"]
print("\n=== GEN3D RENDER SUMMARY ===\n" + "\n".join(summary))
open(f"{W}/render_summary.txt", "w").write("\n".join(summary) + "\n")
shutil.rmtree(TMP, ignore_errors=True)

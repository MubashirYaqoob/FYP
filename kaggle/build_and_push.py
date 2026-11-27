"""Bundle the GEN3D code and push it + the data-pipeline notebook to Kaggle.

Usage (from GEN folder, venv active):
  python kaggle/build_and_push.py code              # create / update private dataset <user>/gen3d-code
  python kaggle/build_and_push.py pipeline [--test N]   # push + run notebook <user>/gen3d-data-pipeline
  python kaggle/build_and_push.py train --name gen3d-train-chair --init-from gen3d-train --epochs 200 \
         --run-name "GEN3D-Gen v1-chair" --extra "--categories chair,stool,bench --val-categories chair"
"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "_build"
KAGGLE = str(ROOT / ".venv" / "Scripts" / "kaggle.exe")


def username():
    out = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout
    return re.search(r"username:\s*(\S+)", out).group(1)


def run(*args):
    print("$ kaggle", " ".join(args))
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return r


def build_code():
    d = BUILD / "code"
    d.mkdir(parents=True, exist_ok=True)
    z = d / "gen3d_code.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for top in ["configs", "gen3d", "scripts"]:
            for p in (ROOT / top).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    f.write(p, p.relative_to(ROOT).as_posix())
        f.write(ROOT / "requirements.txt", "requirements.txt")
        f.write(ROOT / "data/meta/candidates.csv", "data_seed/candidates.csv")
    return d


def push_code():
    user = username()
    d = build_code()
    json.dump({"title": "gen3d-code", "id": f"{user}/gen3d-code", "licenses": [{"name": "other"}]},
              open(d / "dataset-metadata.json", "w"), indent=1)
    exists = run("datasets", "status", f"{user}/gen3d-code").returncode == 0
    if exists:
        run("datasets", "version", "-p", str(d), "-m", "code update")
    else:
        run("datasets", "create", "-p", str(d))  # private by default


def push_pipeline(test_limit):
    user = username()
    d = BUILD / "data_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle/data_pipeline/gen3d_data_pipeline.py").read_text(encoding="utf-8")
    src = re.sub(r"^TEST_LIMIT = \d+", f"TEST_LIMIT = {test_limit}", src, flags=re.M)
    (d / "gen3d_data_pipeline.py").write_text(src, encoding="utf-8")
    json.dump({
        "id": f"{user}/gen3d-data-pipeline", "title": "gen3d-data-pipeline",
        "code_file": "gen3d_data_pipeline.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/gen3d-code"], "competition_sources": [], "kernel_sources": [],
    }, open(d / "kernel-metadata.json", "w"), indent=1)
    run("kernels", "push", "-p", str(d))


def push_render(test_limit, part=1):
    """GPU notebook: re-download GLBs and render views per clean object (reads the pipeline output).
    part 2+ is a separate notebook that also reads the earlier render outputs and skips what is done,
    so every part's output stays available as training input."""
    user = username()
    name = "gen3d-render" if part == 1 else f"gen3d-render-{part}"
    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle/render/gen3d_render.py").read_text(encoding="utf-8")
    src = re.sub(r"^TEST_LIMIT = \d+", f"TEST_LIMIT = {test_limit}", src, flags=re.M)
    (d / "gen3d_render.py").write_text(src, encoding="utf-8")
    json.dump({
        "id": f"{user}/{name}", "title": name,
        "code_file": "gen3d_render.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/gen3d-code"], "competition_sources": [],
        "kernel_sources": [f"{user}/gen3d-data-pipeline"]
                          + [f"{user}/gen3d-render"] + [f"{user}/gen3d-render-{k}" for k in range(2, part)],
    }, open(d / "kernel-metadata.json", "w"), indent=1)
    run("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")  # explicit GPU (T4 x2)


def push_train(extra="", name="gen3d-train", init_from="", epochs=0, run_name="", skip="", eval_args="", min_objects=0):
    """GPU notebook: train GEN3D-Gen on every rendered object (pipeline output + all render parts)."""
    user = username()
    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle/train/gen3d_train.py").read_text(encoding="utf-8")
    if extra:
        src = re.sub(r'^EXTRA_ARGS = "[^"]*"', f'EXTRA_ARGS = "{extra}"', src, flags=re.M)
    if init_from:  # fine-tune: start from <init_from> notebook's checkpoints/best.pt
        src = re.sub(r'^INIT_FROM = "[^"]*"', 'INIT_FROM = "best.pt"', src, flags=re.M)
    if epochs:
        src = re.sub(r"^EPOCHS = \d+", f"EPOCHS = {epochs}", src, flags=re.M)
    if run_name:
        src = re.sub(r'^RUN_NAME = "[^"]*"', f'RUN_NAME = "{run_name}"', src, flags=re.M)
    if min_objects:
        src = re.sub(r"^MIN_OBJECTS = \d+", f"MIN_OBJECTS = {min_objects}", src, flags=re.M)
    if eval_args:
        src = re.sub(r'^EVAL_ARGS = "[^"]*"', f'EVAL_ARGS = "{eval_args}"', src, flags=re.M)
    (d / "gen3d_train.py").write_text(src, encoding="utf-8")
    renders = [k["ref"] for k in kernel_refs(user)
               if k["ref"].split("/")[1].startswith(("gen3d-render", "gen3d-abo", "gen3d-shapenet"))
               and k["ref"].split("/")[1] not in skip.split(",")]  # skip = runs still in progress
    json.dump({
        "id": f"{user}/{name}", "title": name,
        "code_file": "gen3d_train.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/gen3d-code"], "competition_sources": [],
        "kernel_sources": [f"{user}/gen3d-data-pipeline"] + renders
                          + ([f"{user}/{init_from}"] if init_from else []),
    }, open(d / "kernel-metadata.json", "w"), indent=1)
    print("render inputs:", renders)
    run("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")


def push_abo(test_limit, part=1, types=""):
    """GPU notebook: ABO chairs/stools/benches -> download, check, labels, render (second dataset)."""
    user = username()
    name = "gen3d-abo" if part == 1 else f"gen3d-abo-{part}"
    d = BUILD / name
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle/abo/gen3d_abo.py").read_text(encoding="utf-8")
    src = re.sub(r"^TEST_LIMIT = \d+", f"TEST_LIMIT = {test_limit}", src, flags=re.M)
    src = re.sub(r'^TYPES = "[^"]*"', f'TYPES = "{types}"', src, flags=re.M)
    (d / "gen3d_abo.py").write_text(src, encoding="utf-8")
    json.dump({
        "id": f"{user}/{name}", "title": name,
        "code_file": "gen3d_abo.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/gen3d-code"], "competition_sources": [],
        # earlier ABO parts as input, so objects they already rendered are skipped
        "kernel_sources": ["%s/gen3d-abo" % user] if part > 1 else [],
    }, open(d / "kernel-metadata.json", "w"), indent=1)
    run("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4")


def push_shapenet(test_limit):
    """CPU notebook: ShapeNet categories from the Occupancy Networks data -> our format."""
    user = username()
    d = BUILD / "gen3d-shapenet"
    d.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "kaggle/shapenet/gen3d_shapenet.py").read_text(encoding="utf-8")
    src = re.sub(r"^TEST_LIMIT = \d+", f"TEST_LIMIT = {test_limit}", src, flags=re.M)
    (d / "gen3d_shapenet.py").write_text(src, encoding="utf-8")
    json.dump({
        "id": f"{user}/gen3d-shapenet", "title": "gen3d-shapenet",
        "code_file": "gen3d_shapenet.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{user}/gen3d-code"], "competition_sources": [], "kernel_sources": [],
    }, open(d / "kernel-metadata.json", "w"), indent=1)
    run("kernels", "push", "-p", str(d))


def kernel_refs(user):
    out = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--csv"], capture_output=True, text=True).stdout
    rows = [ln.split(",") for ln in out.strip().splitlines()[1:]]
    return [{"ref": r[0]} for r in rows if r and r[0].startswith(user + "/")]


if __name__ == "__main__":
    what = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0
    if what == "code":
        push_code()
    elif what == "pipeline":
        push_pipeline(n)
    elif what == "render":
        part = int(sys.argv[sys.argv.index("--part") + 1]) if "--part" in sys.argv else 1
        push_render(n, part)
    elif what == "abo":
        part = int(sys.argv[sys.argv.index("--part") + 1]) if "--part" in sys.argv else 1
        types = sys.argv[sys.argv.index("--types") + 1] if "--types" in sys.argv else ""
        push_abo(n, part, types)
    elif what == "shapenet":
        push_shapenet(n)
    elif what == "train":
        def arg(k, default=""):
            return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default
        push_train(arg("--extra"), arg("--name", "gen3d-train"), arg("--init-from"),
                   int(arg("--epochs", "0")), arg("--run-name"), arg("--skip"), arg("--eval"), int(arg("--min-objects", "0")))

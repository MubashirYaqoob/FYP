"""Bundle the GEN3D code and push it + the data-pipeline notebook to Kaggle.

Usage (from GEN folder, venv active):
  python kaggle/build_and_push.py code              # create / update private dataset <user>/gen3d-code
  python kaggle/build_and_push.py pipeline [--test N]   # push + run notebook <user>/gen3d-data-pipeline
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


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "code":
        push_code()
    elif what == "pipeline":
        n = int(sys.argv[sys.argv.index("--test") + 1]) if "--test" in sys.argv else 0
        push_pipeline(n)

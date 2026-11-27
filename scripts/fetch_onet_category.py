"""Download ONE ShapeNet category from the Occupancy Networks dataset zip, without the other 60 GB.

The ONet zip (dataset_small_v1.1.zip, 73 GB) stores each category's files next to each other.
We read the zip's table of contents over HTTP, download only the byte range of the wanted category
(one HTTP range request, e.g. 11.4 GB for chairs) and unpack the files from it.

Usage: python scripts/fetch_onet_category.py --synset 03001627 --dest /tmp/onet
Result: /tmp/onet/ShapeNet/03001627/<id>/{points.npz, pointcloud.npz, img_choy2016/...}, *.lst
"""
import argparse
import os
import struct
import subprocess
import zlib
from pathlib import Path

ONET_ZIP = "https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip"
SKIP = (".binvox",)          # voxel files are not used by us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synset", default="03001627")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--url", default=ONET_ZIP)
    ap.add_argument("--test", type=int, default=0, help="only the first N files (quick check)")
    args = ap.parse_args()

    from remotezip import RemoteZip   # reads the central directory with range requests

    prefix = f"ShapeNet/{args.synset}/"
    with RemoteZip(args.url) as z:
        entries = [i for i in z.infolist() if i.filename.startswith(prefix)]
    if args.test:
        entries = sorted(entries, key=lambda i: i.header_offset)[:args.test]
    start = min(i.header_offset for i in entries)
    end = max(i.header_offset + 30 + len(i.filename.encode()) + 1024 + i.compress_size for i in entries)
    blob = Path(args.dest) / f"{args.synset}.part"
    blob.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(entries)} files, downloading bytes {start}..{end} ({(end - start) / 1e9:.2f} GB)", flush=True)
    subprocess.run(["curl", "-s", "-S", "--retry", "5", "-r", f"{start}-{end}", "-o", str(blob), args.url], check=True)

    n = 0
    with open(blob, "rb") as f:
        for i in entries:
            if i.filename.endswith("/") or i.filename.endswith(SKIP):
                continue
            f.seek(i.header_offset - start)
            head = f.read(30)
            name_len, extra_len = struct.unpack("<HH", head[26:30])
            f.seek(name_len + extra_len, os.SEEK_CUR)
            data = f.read(i.compress_size)
            if i.compress_type == 8:
                data = zlib.decompress(data, -15)
            out = Path(args.dest) / i.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            n += 1
    blob.unlink()
    print(f"unpacked {n} files -> {Path(args.dest) / prefix}")


if __name__ == "__main__":
    main()

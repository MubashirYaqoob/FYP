"""Step 4: render objects with Blender (runs INSIDE Blender, headless).

Modes
  thumb : 1 picture per object (fast Workbench engine) for manual review
          -> data/processed/thumbs/<uid>.png
  views : N random views per object for training (Eevee locally, Cycles on GPU machines)
          -> data/processed/objects/<uid>/views/000.png ... + cameras.json

Run (Windows, from the GEN folder):
  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" -b -P scripts/render_blender.py -- --mode thumb
  "...blender.exe" -b -P scripts/render_blender.py -- --mode views --limit 5
Options after "--":  --mode thumb|views  --limit N  --views 16  --res 224  --engine eevee|cycles|workbench

Frames / conventions (must match training code)
  * Object normalisation is read from objects/<uid>/norm.json written by preprocess.py
    (so run preprocess.py first). Everything is stored in the glTF frame (Y-up), normalised.
  * cameras.json: c2w = 4x4 camera-to-world in that frame, OpenGL convention
    (camera looks along -Z, +Y is up). K = 3x3 intrinsics in pixels.
"""
import csv
import json
import math
import os
import random
import sys
import zlib

import bpy
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "meta")
OBJ_DIR = os.path.join(ROOT, "data", "processed", "objects")
THUMB_DIR = os.path.join(ROOT, "data", "processed", "thumbs")

# glTF (Y-up) -> Blender (Z-up): x_b = x_g, y_b = -z_g, z_b = y_g
GLTF_TO_BL = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
BL_TO_GLTF = GLTF_TO_BL.inverted()


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = {"mode": "thumb", "limit": 0, "views": 16, "res": 224, "engine": None}
    it = iter(argv)
    for k in it:
        a[k.lstrip("-")] = next(it)
    a["limit"], a["views"], a["res"] = int(a["limit"]), int(a["views"]), int(a["res"])
    if a["engine"] is None:
        a["engine"] = "workbench" if a["mode"] == "thumb" else "eevee"
    return a


def object_list(limit):
    with open(os.path.join(META, "checked.csv"), newline="", encoding="utf-8") as f:
        paths = {r["uid"]: (r["path"], r["category"]) for r in csv.DictReader(f) if r["status"] == "PASS"}
    with open(os.path.join(META, "preprocessed.csv"), newline="", encoding="utf-8") as f:
        ok = [r["uid"] for r in csv.DictReader(f) if r["status"] == "ok"]
    objs = [(u, *paths[u]) for u in ok if u in paths]
    return objs[:limit] if limit else objs


# ------------------------------------------------------------------ scene setup
def setup_scene(engine, res):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    if engine == "eevee":
        names = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in names else "BLENDER_EEVEE"
    elif engine == "cycles":
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 64
        sc.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = True
                sc.cycles.device = "GPU"
                break
            except Exception:  # noqa: BLE001
                continue
    else:
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.light = "STUDIO"
        sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"   # true texture colours

    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0  # soft white ambient

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens, cam.data.sensor_width = 35, 32  # ~49.1 deg FOV
    sc.collection.objects.link(cam)
    sc.camera = cam

    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sc.collection.objects.link(sun)
    return sc, cam, sun


def import_normalised(path, norm):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    pivot = bpy.data.objects.new("pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    for o in new:
        if o.parent is None:
            o.parent = pivot
    s = norm["scale"]
    c = GLTF_TO_BL @ Vector(norm["center"] + [1.0])
    pivot.scale = (s, s, s)
    pivot.location = -s * Vector(c[:3])
    bpy.context.view_layer.update()

    # sanity check: normalised object should span ~1 on its longest side
    # (use real vertices: bound_box corners of rotated parts over-estimate the extent)
    lo, hi = Vector((1e9,) * 3), Vector((-1e9,) * 3)
    for o in new:
        if o.type == "MESH":
            mw = o.matrix_world
            for v in o.data.vertices:
                w = mw @ v.co
                lo, hi = Vector(map(min, lo, w)), Vector(map(max, hi, w))
    ext = max(hi - lo)
    if abs(ext - 1.0) > 0.05:
        print(f"[warn] normalised extent {ext:.3f} (expected 1.0) for {path}")


def look_at(cam, az_deg, el_deg, dist):
    az, el = math.radians(az_deg), math.radians(el_deg)
    # spherical coordinates in the glTF frame (Y-up), converted to Blender
    pos_g = Vector((dist * math.cos(el) * math.sin(az), dist * math.sin(el), dist * math.cos(el) * math.cos(az)))
    pos = (GLTF_TO_BL @ pos_g.to_4d()).to_3d()
    cam.location = pos
    cam.rotation_euler = (-pos).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def intrinsics(cam, res):
    f = cam.data.lens / cam.data.sensor_width * res
    return [[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]]


# ------------------------------------------------------------------ modes
def render_thumb(sc, cam, sun, uid, path, norm, res):
    out = os.path.join(THUMB_DIR, uid + ".png")
    if os.path.exists(out):
        return
    import_normalised(path, norm)
    look_at(cam, 35, 20, 2.0)
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)


def render_views(sc, cam, sun, uid, path, norm, n_views, res):
    vdir = os.path.join(OBJ_DIR, uid, "views")
    if os.path.exists(os.path.join(vdir, "cameras.json")):
        return
    os.makedirs(vdir, exist_ok=True)
    import_normalised(path, norm)
    rng = random.Random(zlib.crc32(uid.encode()))
    frames = []
    for i in range(n_views):
        az = rng.uniform(0, 360)
        el = rng.uniform(-10, 40)        # mostly from above, like real photos
        dist = rng.uniform(1.8, 2.2)
        look_at(cam, az, el, dist)
        sun.rotation_euler = (math.radians(rng.uniform(20, 60)), 0, math.radians(rng.uniform(0, 360)))
        sun.data.energy = rng.uniform(2.0, 4.0)
        sc.render.filepath = os.path.join(vdir, f"{i:03d}.png")
        bpy.ops.render.render(write_still=True)
        c2w = BL_TO_GLTF @ cam.matrix_world
        frames.append({"file": f"{i:03d}.png", "azimuth": az, "elevation": el, "distance": dist,
                       "c2w": [list(r) for r in c2w]})
    with open(os.path.join(vdir, "cameras.json"), "w") as f:
        json.dump({"convention": "opengl", "frame": "gltf_y_up_normalised", "res": res,
                   "K": intrinsics(cam, res), "frames": frames}, f, indent=1)


def main():
    a = parse_args()
    objs = object_list(a["limit"])
    os.makedirs(THUMB_DIR, exist_ok=True)
    print(f"[GEN3D] mode={a['mode']} engine={a['engine']} objects={len(objs)}")
    for k, (uid, path, cat) in enumerate(objs):
        with open(os.path.join(OBJ_DIR, uid, "norm.json")) as f:
            norm = json.load(f)
        try:
            sc, cam, sun = setup_scene(a["engine"], 256 if a["mode"] == "thumb" else a["res"])
            if a["mode"] == "thumb":
                render_thumb(sc, cam, sun, uid, path, norm, 256)
            else:
                render_views(sc, cam, sun, uid, path, norm, a["views"], a["res"])
            print(f"[GEN3D] {k + 1}/{len(objs)} {cat} {uid} ok")
        except Exception as e:  # noqa: BLE001
            print(f"[GEN3D] {k + 1}/{len(objs)} {cat} {uid} FAILED: {e}")


if __name__ == "__main__":
    main()

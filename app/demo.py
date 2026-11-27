"""GEN3D demo: photo -> our generator (GEN3D-Gen) -> textured 3D mesh -> simplifier slider.

Run (from the project folder, venv active):
    python app/demo.py
then open http://127.0.0.1:7860 in the browser.

Stage 1 uses our trained models (configs/default.yaml -> demo.models); pick one in the UI.
Stage 2 currently uses PyMeshLab quadric edge collapse as the QEM baseline; our own QEM and
D-QEM will be added to the same panel later.
"""
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen3d.generator.extract import (clean_mesh, default_camera, extract_mesh, photo_colours, photo_to_tensor,
                                     prepare_photo)  # noqa: E402
from gen3d.generator.model import Generator  # noqa: E402
from gen3d.utils.common import load_config, project_path  # noqa: E402
from scripts.visualize_pipeline import pipeline_figure  # noqa: E402

CFG = load_config()
D = CFG["demo"]
OUT = project_path(D["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ model
_MODELS = {}


def get_model(name):
    """Load a checkpoint once and keep it in memory: (model, its config)."""
    if name not in _MODELS:
        ck = torch.load(project_path(D["models"][name]), map_location=DEVICE, weights_only=False)
        model = Generator(ck["config"]).to(DEVICE).eval()
        model.load_state_dict(ck["model"])
        _MODELS[name] = (model, ck["config"])
    return _MODELS[name]


MODEL_NAMES = list(D["models"])
get_model(MODEL_NAMES[0])  # load the default model at start-up
IMG_SIZE = _MODELS[MODEL_NAMES[0]][1]["generator"]["encoder"]["img_size"]


def mesh_stats(mesh, seconds):
    return f"**{len(mesh.faces):,}** triangles · **{len(mesh.vertices):,}** vertices · {seconds:.1f} s"


def export_all(mesh, stem):
    """Save the mesh as .glb (viewer, Blender, Unity), .obj and .ply (MeshLab, Blender).
    All three keep the per-vertex colours. Returns [glb, obj, ply] paths."""
    paths = [OUT / f"{stem}.{ext}" for ext in ("glb", "obj", "ply")]
    for p in paths:
        mesh.export(p)
    return [str(p) for p in paths]


# ------------------------------------------------------------------ stage 1: photo -> mesh
def generate(photo_path, model_name, resolution, level, cleanup, use_photo):
    if not photo_path:
        raise gr.Error("Upload a photo first (white or transparent background works best).")
    t0 = time.time()
    try:
        rgba = prepare_photo(photo_path, IMG_SIZE)
    except ValueError:
        raise gr.Error("No object found: the whole photo looks like background.")
    model, mcfg = get_model(model_name)
    t, K = default_camera(mcfg, DEVICE)
    mesh = extract_mesh(model, photo_to_tensor(rgba), t, K, res=int(resolution), level=float(level))
    if mesh is None:
        raise gr.Error("The model found no object in this photo. Try a lower surface threshold.")
    if cleanup:
        mesh = clean_mesh(mesh, D["min_part"], D["smooth_iters"])
    if use_photo:   # real photo colours on the visible side, texture field on the hidden side
        hr = D["photo_res"]
        k_hr = K.cpu().numpy() * (hr / IMG_SIZE)
        k_hr[2, 2] = 1.0
        mesh = photo_colours(mesh, prepare_photo(photo_path, hr), t.cpu().numpy(), k_hr)
    files = export_all(mesh, f"generated_{time.strftime('%H%M%S')}")
    path = files[0]
    seen = Image.fromarray(rgba).convert("RGBA")
    white = Image.new("RGBA", seen.size, (255, 255, 255, 255))
    seen = Image.alpha_composite(white, seen).convert("RGB")
    return seen, path, mesh_stats(mesh, time.time() - t0), path, files


# ------------------------------------------------------------------ stage 2: simplify
def simplify(mesh_path, keep_percent):
    if not mesh_path:
        raise gr.Error("Generate a 3D mesh first (step 1).")
    import pymeshlab

    t0 = time.time()
    src = trimesh.load(mesh_path, force="mesh", process=False)
    target = max(4, int(len(src.faces) * float(keep_percent) / 100))
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=src.vertices.astype(np.float64),
                               face_matrix=src.faces.astype(np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservenormal=True,
                                                optimalplacement=True, planarquadric=True)
    m = ms.current_mesh()
    verts, faces = m.vertex_matrix(), m.face_matrix()
    # colour of each new vertex = colour of the nearest original vertex
    _, idx = cKDTree(src.vertices).query(verts)
    colours = np.asarray(src.visual.vertex_colors)[idx]
    low = trimesh.Trimesh(verts, faces, vertex_colors=colours, process=False)
    files = export_all(low, f"lowpoly_{int(keep_percent)}pct_{time.strftime('%H%M%S')}")
    ratio = len(src.faces) / max(1, len(low.faces))
    return files[0], mesh_stats(low, time.time() - t0) + f" · **{ratio:.0f}×** fewer triangles", files


# ------------------------------------------------------------------ inside the model
def inside(photo_path, model_name):
    if not photo_path:
        raise gr.Error("Upload a photo first.")
    import matplotlib.pyplot as plt

    model, mcfg = get_model(model_name)
    fig = pipeline_figure(model, mcfg, photo_path, res=64)
    path = OUT / f"inside_{time.strftime('%H%M%S')}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------ UI
CSS = """
#title h1 {margin-bottom: 0}
.stage {border-radius: 10px}
.note {font-size: 0.9em; opacity: 0.8}
"""

examples = sorted(str(p) for p in project_path(D["examples_dir"]).glob("*.png"))

with gr.Blocks(title="GEN3D Demo", css=CSS, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        "# GEN3D\n"
        "**One photo → our trained 3D generator → textured mesh → low-poly mesh.** "
        f"Our own trained models, running on {DEVICE.type.upper()}.",
        elem_id="title",
    )
    mesh_state = gr.State(None)

    with gr.Group(elem_classes="stage"):
        gr.Markdown("### Step 1 · Photo → 3D (GEN3D-Gen)")
        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Image(type="filepath", label="Photo (white or transparent background)",
                                 image_mode=None, format="png", height=260)
                model_pick = gr.Dropdown(MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
                with gr.Row():
                    res = gr.Radio(D["resolutions"], value=D["default_resolution"],
                                   label="Detail (grid size)", info="64 fast · 128 detailed")
                    level = gr.Slider(0.2, 0.8, value=D["default_level"], step=0.05,
                                      label="Surface threshold",
                                      info="lower keeps thin parts, may add blobs")
                with gr.Row():
                    cleanup = gr.Checkbox(True, label="Clean up mesh (remove floating bits + light smoothing)")
                    use_photo = gr.Checkbox(True, label="Photo colours (realistic look on the visible side)")
                go = gr.Button("Generate 3D", variant="primary")
                gr.Examples(examples, inputs=photo, label="Example photos (our renders)",
                            examples_per_page=9)
            with gr.Column(scale=1):
                seen = gr.Image(label="What the model sees (224 × 224)", height=200,
                                interactive=False)
                stats1 = gr.Markdown()
                dl1 = gr.File(label="Download 3D mesh (.glb · .obj · .ply)", file_count="multiple",
                              interactive=False)
            with gr.Column(scale=2):
                view1 = gr.Model3D(label="Generated textured mesh (drag to rotate)", height=460,
                                   clear_color=(0.96, 0.97, 0.97, 1.0))

    with gr.Group(elem_classes="stage"):
        gr.Markdown("### Step 2 · Make it low-poly")
        with gr.Row():
            keep = gr.Slider(1, 100, value=D["default_keep"], step=1,
                             label="Keep % of triangles", scale=3)
            go2 = gr.Button("Simplify", variant="primary", scale=1)
        with gr.Row():
            with gr.Column():
                view_orig = gr.Model3D(label="Original (from step 1)", height=380,
                                       clear_color=(0.96, 0.97, 0.97, 1.0))
            with gr.Column():
                view_low = gr.Model3D(label="Low-poly · QEM baseline (PyMeshLab)", height=380,
                                      clear_color=(0.96, 0.97, 0.97, 1.0))
                stats2 = gr.Markdown()
                dl2 = gr.File(label="Download low-poly mesh (.glb · .obj · .ply)",
                              file_count="multiple", interactive=False)
        gr.Markdown("Our own UV-aware QEM and **D-QEM** (QEM + Transformer) will be added here "
                    "as extra columns for a side-by-side comparison.", elem_classes="note")

    with gr.Group(elem_classes="stage"):
        gr.Markdown("### Inside the model · what happens to your photo, step by step\n"
                    "Uses the photo and model chosen in step 1: encoder features, 3D points projected "
                    "onto the photo, inside-probability slices (front and side), coloured inside points, "
                    "final mesh. Takes about 20 s on CPU.")
        go3 = gr.Button("Show inside the model", variant="secondary")
        inside_img = gr.Image(label="Internal pipeline", interactive=False)

    go.click(generate, [photo, model_pick, res, level, cleanup, use_photo], [seen, view1, stats1, mesh_state, dl1]).then(
        lambda p: p, mesh_state, view_orig)
    go2.click(simplify, [mesh_state, keep], [view_low, stats2, dl2])
    go3.click(inside, [photo, model_pick], inside_img)

if __name__ == "__main__":
    demo.launch(server_port=D["port"], inbrowser=True)

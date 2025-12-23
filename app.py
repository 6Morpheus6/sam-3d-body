import sys
import shutil
import tempfile
import subprocess
import uuid
import re
from pathlib import Path

import gradio as gr
import trimesh

# =========================================================
# KONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PREVIEW_DIR = OUTPUT_DIR / "_preview"
PREVIEW_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt"
MHR_PATH = BASE_DIR / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"

# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def resolve_input_images(files):
    image_paths = []
    for f in files:
        p = Path(f)
        if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            image_paths.append(str(p.resolve()))

    if not image_paths:
        raise RuntimeError("Keine gültigen Bilder gefunden.")

    return image_paths


def get_next_run_id(output_dir: Path, width: int = 5) -> str:
    pattern = re.compile(r"run_(\d+)")
    max_id = 0

    for p in output_dir.iterdir():
        if p.is_dir():
            m = pattern.fullmatch(p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))

    return f"{max_id + 1:0{width}d}"


def cleanup_mesh_filenames(folder: Path):
    for p in folder.iterdir():
        if p.is_file() and "_combined" in p.stem:
            new_path = p.with_name(p.name.replace("_combined", ""))
            if not new_path.exists():
                p.rename(new_path)


# =========================================================
# DECIMATION (PREVIEW)
# =========================================================

def decimate_mesh(mesh_path: str, ratio: float):
    mesh = trimesh.load(mesh_path, force="mesh", process=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Mesh konnte nicht geladen werden")

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    original_faces = len(mesh.faces)

    ratio = float(ratio)
    ratio = max(0.01, min(1.0, ratio))

    target_faces = int(original_faces * ratio)
    print(target_faces)

    print(
        f"[DECIMATE] {Path(mesh_path).name} | "
        f"{original_faces} → {target_faces} faces"
    )

    # 100 % → Original zurückgeben
    if target_faces >= original_faces:
        return mesh_path, original_faces, target_faces

    try:
        decimated = mesh.simplify_quadric_decimation(
            face_count=target_faces,
            aggression=5
        )
    except Exception as e:
        print("[ERROR] Decimation fehlgeschlagen:", e)
        return mesh_path, original_faces, target_faces

    preview_path = PREVIEW_DIR / (
        Path(mesh_path).stem
        + f"_decimated_{target_faces}.glb"
    )

    decimated.export(preview_path)
    return str(preview_path), original_faces, target_faces


# =========================================================
# PIPELINE CALLBACK
# =========================================================

def process_images(files):
    image_paths = resolve_input_images(files)

    run_id = get_next_run_id(OUTPUT_DIR)
    run_output_dir = OUTPUT_DIR / f"run_{run_id}"
    run_output_dir.mkdir()

    with tempfile.TemporaryDirectory(prefix="sam3d_run_") as tmpdir:
        tmpdir = Path(tmpdir)

        for i, src in enumerate(image_paths):
            dst = tmpdir / f"image_{i:03d}{Path(src).suffix.lower()}"
            shutil.copyfile(src, dst)

        cmd = [
            sys.executable,
            "demo.py",
            "--image_folder", str(tmpdir),
            "--output_folder", str(run_output_dir),
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--mhr_path", str(MHR_PATH),
        ]

        subprocess.run(cmd, check=True)

    cleanup_mesh_filenames(run_output_dir)

    meshes = sorted(
        str(p) for p in run_output_dir.iterdir()
        if p.suffix.lower() in [".obj", ".ply", ".glb"]
    )

    dropdown_update = gr.update(
        choices=meshes,
        value=meshes[0] if meshes else None
    )

    status = f"Fertig. {len(meshes)} Mesh(es) erzeugt."

    return meshes, dropdown_update, status

# =========================================================
# UI CALLBACKS
# =========================================================

def on_mesh_selected(selected_mesh):
    if not selected_mesh:
        return gr.update(value=1.0), None
    return gr.update(value=1.0), selected_mesh


def update_decimation(selected_mesh, ratio):
    if not selected_mesh:
        return None, "Faces: –"

    preview, original, target = decimate_mesh(selected_mesh, ratio)

    info = f"**Faces:** {original:,} → {target:,} ({int(ratio*100)}%)"
    return preview, info

# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(title="SAM-3D-Body (Windows, Gradio 5.x)") as demo:

    gr.Markdown("## SAM-3D-Body – 3D Mesh aus Bildern")

    with gr.Row():
        with gr.Column(scale=1):

            image_input = gr.File(
                file_types=["image"],
                file_count="multiple",
                label="Bilder hochladen",
            )

            run_button = gr.Button("Run SAM-3D-Body", variant="primary")

            status_text = gr.Textbox(label="Status", interactive=False)

            mesh_selector = gr.Dropdown(
                label="Mesh auswählen (Preview)",
                choices=[],
                interactive=True,
            )

            decimation_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Mesh-Detail (Anteil der Polygone)",
            )

            mesh_output = gr.File(
                label="Erzeugte Meshes (Download)",
                file_count="multiple",
                interactive=False,
            )

        with gr.Column(scale=1):

            mesh_viewer = gr.Model3D(
                label="3D Vorschau",
                clear_color=[0.9, 0.9, 0.9, 1.0],
                height="85vh",
            )

            face_info = gr.Markdown("Faces: –")

    run_button.click(
        fn=process_images,
        inputs=image_input,
        outputs=[mesh_output, mesh_selector, status_text],
    )

    mesh_selector.change(
        fn=on_mesh_selected,
        inputs=mesh_selector,
        outputs=[decimation_slider, mesh_viewer],
    )

    decimation_slider.change(
        fn=update_decimation,
        inputs=[mesh_selector, decimation_slider],
        outputs=[mesh_viewer, face_info],
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    demo.launch()

import re
import sys
import shutil
import trimesh
import zipfile
import tempfile
import subprocess
import gradio as gr
from pathlib import Path


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

CURRENT_PREVIEW_MESH = None

# =========================================================
# Help Functions
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


def list_runs(output_dir: Path):
    runs = []
    for p in sorted(output_dir.iterdir()):
        if p.is_dir() and p.name.startswith("run_"):
            runs.append(p.name)
    return runs

# =========================================================
# DECIMATION (PREVIEW)
# =========================================================

def decimate_mesh(mesh_path: str, ratio: float):
    mesh = trimesh.load(mesh_path, force="mesh", process=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Mesh could not be loaded for decimation.")

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

    # 100 % → Return original
    if target_faces >= original_faces:
        return mesh_path, original_faces, target_faces

    try:
        decimated = mesh.simplify_quadric_decimation(
            face_count=target_faces,
            aggression=5
        )
    except Exception as e:
        print("[ERROR] Decimation failed:", e)
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

    status = f"Done. Created {len(meshes)} Mesh(es)."

    mesh_output_update = gr.update(
        value=meshes,
        visible=bool(meshes)
    )

    run_update = gr.update(
        choices=list_runs(OUTPUT_DIR),
        value=f"run_{run_id}",
    )


    return mesh_output_update, dropdown_update, status, run_update

# =========================================================
# UI CALLBACKS
# =========================================================

def on_mesh_selected(selected_mesh):
    global CURRENT_PREVIEW_MESH
    if not selected_mesh:
        CURRENT_PREVIEW_MESH = None
        return gr.update(value=1.0), None, "Faces: –"

    mesh = trimesh.load(selected_mesh, force="mesh", process=False)
    face_count = len(mesh.faces)

    CURRENT_PREVIEW_MESH = selected_mesh
    info = f"**Faces:** {face_count:,} (100 %)"
    return gr.update(value=1.0), selected_mesh, info



def update_decimation(selected_mesh, ratio):
    global CURRENT_PREVIEW_MESH
    if not selected_mesh:
        CURRENT_PREVIEW_MESH = None
        return None, "Faces: –"

    preview, original, target = decimate_mesh(selected_mesh, ratio)
    CURRENT_PREVIEW_MESH = preview

    info = f"**Faces:** {original:,} → {target:,} ({int(ratio*100)}%)"
    return preview, info


def export_current_mesh(export_format):
    if not CURRENT_PREVIEW_MESH:
        return gr.update(visible=False)

    src = Path(CURRENT_PREVIEW_MESH)
    export_path = src.with_suffix(f".{export_format}")

    mesh = trimesh.load(src, force="mesh", process=False)
    mesh.export(export_path)

    return gr.update(value=str(export_path), visible=True)

def download_all_meshes(mesh_files):
    if not mesh_files:
        return gr.update(visible=False)

    zip_path = OUTPUT_DIR / "sam3d_meshes.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in mesh_files:
            f = Path(f)
            z.write(f, arcname=f.name)

    return gr.update(value=str(zip_path), visible=True)


def load_meshes_for_run(run_name: str):
    run_dir = OUTPUT_DIR / run_name
    if not run_dir.exists():
        return gr.update(choices=[], value=None), None, "Faces: –"

    meshes = sorted(
        str(p) for p in run_dir.iterdir()
        if p.suffix.lower() in [".glb", ".obj", ".ply"]
    )

    if not meshes:
        return gr.update(choices=[], value=None), None, "Faces: –"

    # Faces from the first Mesh
    mesh = trimesh.load(meshes[0], force="mesh", process=False)
    face_info = f"**Faces:** {len(mesh.faces):,} (100 %)"

    return gr.update(choices=meshes, value=meshes[0]), meshes[0], face_info

# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(title="SAM3D Body") as demo:

    gr.Markdown("## SAM3D Body – Create 3D Meshes from Images")

    with gr.Row():
        with gr.Column(scale=1):

            image_input = gr.File(
                file_types=["image"],
                file_count="multiple",
                label="Upload Images",
            )

            run_button = gr.Button("Run SAM3D Body", variant="primary")

            status_text = gr.Textbox(label="Status", interactive=False)

            run_selector = gr.Dropdown(
                label="Select Previous Run",
                choices=list_runs(OUTPUT_DIR),
                interactive=True,
            )

            mesh_selector = gr.Dropdown(
                label="Select Mesh (Preview)",
                choices=[],
                interactive=True,
            )

            decimation_slider = gr.Slider(
                minimum=0.05,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Mesh-Detail (Amount of Polygons)",
            )

            export_format = gr.Dropdown(
                choices=["glb", "obj", "ply"],
                value="glb",
                label="Export-Format",
            )

            mesh_output = gr.File(
                label="Created Meshes (Download)",
                file_count="multiple",
                interactive=False,
                visible=False,
            )

            export_button = gr.Button("Export current Mesh")

            export_file = gr.File(
                label="Exported File",
                visible=False,
            )

            batch_download_btn = gr.Button("Download all Meshes as ZIP")

            batch_zip = gr.File(
                label="ZIP Download",
                visible=False,
            )

        with gr.Column(scale=1):

            mesh_viewer = gr.Model3D(
                label="3D Preview",
                clear_color=[0.09, 0.09, 0.11, 1.0],
                height="85vh",
            )

            face_info = gr.Markdown("Faces: –")

    run_button.click(
        fn=process_images,
        inputs=image_input,
        outputs=[mesh_output, mesh_selector, status_text, run_selector],
    )

    run_selector.change(
        fn=load_meshes_for_run,
        inputs=run_selector,
        outputs=[mesh_selector, mesh_viewer, face_info],
    )

    batch_download_btn.click(
        fn=download_all_meshes,
        inputs=mesh_output,
        outputs=batch_zip,
    )

    mesh_selector.change(
        fn=on_mesh_selected,
        inputs=mesh_selector,
        outputs=[decimation_slider, mesh_viewer, face_info],
    )

    decimation_slider.change(
        fn=update_decimation,
        inputs=[mesh_selector, decimation_slider],
        outputs=[mesh_viewer, face_info],
    )

    export_button.click(
        fn=export_current_mesh,
        inputs=export_format,
        outputs=export_file,
    )

# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    demo.launch()

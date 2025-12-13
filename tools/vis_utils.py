# Copyright (c) Meta Platforms

import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
import os

# Farbe egal – keine Renderbilder mehr
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces, outdir="./output", image_name="sample"):
    """
    Statt Bilder zu rendern erzeugen wir nun GLB-Dateien.
    Für jede Person wird eine GLB erzeugt:
        <outdir>/<image_name>_person_<id>.glb

    Rückgabe: Liste der erzeugten GLB-Pfade
    """
    os.makedirs(outdir, exist_ok=True)
    glb_paths = []

    for pid, person_output in enumerate(outputs):
        renderer = Renderer(
            focal_length=person_output["focal_length"],
            faces=faces
        )

        glb_path = os.path.join(outdir, f"{image_name}_person_{pid}.glb")

        renderer.save_glb(
            vertices=person_output["pred_vertices"],
            cam_t=person_output["pred_cam_t"],
            output_path=glb_path,
        )

        glb_paths.append(glb_path)

    return glb_paths



def visualize_sample_together(img_cv2, outputs, faces, outdir="./output", image_name="sample"):
    """
    Statt ein großes 2D-Bild zu rendern:
    -> alles wird zu EINEM GLB kombiniert
       <image_name>_combined.glb
    """
    os.makedirs(outdir, exist_ok=True)

    # Combine all vertices into one global mesh
    all_pred_vertices = []
    all_faces = []

    for pid, person_output in enumerate(outputs):
        verts = person_output["pred_vertices"] + person_output["pred_cam_t"]
        all_pred_vertices.append(verts)
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)

    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    combined_path = os.path.join(outdir, f"{image_name}_combined.glb")

    renderer = Renderer(
        focal_length=outputs[0]["focal_length"],
        faces=all_faces
    )

    renderer.save_glb(
        vertices=all_pred_vertices,
        cam_t=[0, 0, 0],
        output_path=combined_path
    )

    return combined_path

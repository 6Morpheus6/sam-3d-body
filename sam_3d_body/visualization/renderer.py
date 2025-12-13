# sam_3d_body/visualization/renderer.py
# Simplified renderer: exports meshes to GLB using trimesh (Windows-friendly).
import os
from typing import List, Optional
import numpy as np
import trimesh
import io

class Renderer:
    """
    Lightweight renderer replacement:
    - no pyrender / no OpenGL
    - exports meshes to GLB files (binary glTF)
    - can return GLB bytes for direct use in web UIs
    """

    def __init__(self, focal_length: float, faces: Optional[np.ndarray] = None):
        """
        focal_length: kept for API compatibility
        faces: (F,3) face indices numpy array
        """
        self.focal_length = focal_length
        self.faces = faces

    def _make_trimesh(self, vertices: np.ndarray, translation: Optional[np.ndarray] = None,
                      vertex_colors=None) -> trimesh.Trimesh:
        """
        Create a trimesh.Trimesh object from vertices and self.faces.
        Optionally apply translation (camera translation) to vertices.
        """
        verts = vertices.copy()
        if translation is not None:
            # translation is expected shape (3,)
            verts = verts + np.asarray(translation).reshape(1, 3)

        if self.faces is None:
            # fallback: try to infer a convex hull if faces not provided
            mesh = trimesh.Trimesh(verts, process=True)
        else:
            mesh = trimesh.Trimesh(verts, faces=self.faces.copy(), process=False)

        # apply vertex colors if provided
        if vertex_colors is not None:
            # vertex_colors shape (V,4) or (V,3)
            mesh.visual.vertex_colors = np.asarray(vertex_colors)

        return mesh

    def save_glb(self, vertices: np.ndarray, cam_t: Optional[np.ndarray],
                 output_path: str, mesh_base_color=(1.0, 1.0, 0.9)) -> str:
        """
        Export mesh to GLB file on disk and return the path.
        """
        # prepare vertex colors RGBA 0-255
        col = np.array([int(255 * c) for c in mesh_base_color] + [255], dtype=np.uint8)
        vertex_colors = np.tile(col.reshape(1, 4), (vertices.shape[0], 1))

        mesh = self._make_trimesh(vertices, translation=cam_t, vertex_colors=vertex_colors)

        # ensure directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # export GLB (binary glTF)
        mesh.export(output_path, file_type="glb")
        return output_path

    def get_glb_bytes(self, vertices: np.ndarray, cam_t: Optional[np.ndarray],
                      mesh_base_color=(1.0, 1.0, 0.9)) -> bytes:
        """
        Export mesh to GLB and return bytes (useful to send to web UI without writing disk).
        """
        col = np.array([int(255 * c) for c in mesh_base_color] + [255], dtype=np.uint8)
        vertex_colors = np.tile(col.reshape(1, 4), (vertices.shape[0], 1))
        mesh = self._make_trimesh(vertices, translation=cam_t, vertex_colors=vertex_colors)

        bio = io.BytesIO()
        mesh.export(bio, file_type="glb")
        bio.seek(0)
        return bio.read()

    # Compatibility helper: previous API returned RGBA images. We keep a no-op placeholder.
    def __call__(self, vertices, cam_t, image, *args, imgname=None, return_rgba=False, **kwargs):
        """
        Legacy-call compatibility: export glb next to output and return overlay if requested.
        - If return_rgba True, we return None (rendering disabled).
        """
        # default behavior: write a glb next to image name if provided
        if imgname is not None:
            base = os.path.splitext(os.path.basename(imgname))[0]
            outpath = os.path.join("output", f"{base}.glb")
        else:
            outpath = os.path.join("output", "mesh.glb")

        self.save_glb(vertices, cam_t, outpath)
        if return_rgba:
            # we no longer render images on backend; return None or a placeholder
            return None
        return os.path.abspath(outpath)

    # convenience: create trimesh for further processing
    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0):
        col = np.array([int(255 * c) for c in mesh_base_color] + [255], dtype=np.uint8)
        vertex_colors = np.tile(col.reshape(1, 4), (vertices.shape[0], 1))
        mesh = self._make_trimesh(vertices + camera_translation, vertex_colors=vertex_colors)
        if rot_angle != 0:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis))
        return mesh

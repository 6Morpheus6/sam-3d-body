import os
import torch
import numpy as np

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class HumanSegmentor:
    def __init__(self, name="sam2", device="cuda", **kwargs):
        self.device = device

        if name == "sam2":
            print("########### Using human segmentor: SAM2...")
            self.sam = load_sam2(device, **kwargs)
            self.sam_func = run_sam2
        else:
            raise NotImplementedError

    def run_sam(self, img, boxes, **kwargs):
        return self.sam_func(self.sam, img, boxes)


def load_sam2(device, path):
    """
    Erwartete Struktur:

    segmentor_path/
        checkpoints/
            sam2.1_hiera_large.pt
        sam2/
            configs/
                sam2.1/
                    sam2.1_hiera_l.yaml
    """

    sam2_root = os.path.join(path, "sam2")
    config_root = os.path.abspath(os.path.join(sam2_root, "configs"))

    checkpoint = os.path.abspath(
        os.path.join(path, "checkpoints", "sam2.1_hiera_large.pt")
    )

    if not os.path.exists(config_root):
        raise FileNotFoundError(f"SAM2 config dir not found: {config_root}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # ⚠️ Hydra darf nur einmal initialisiert sein
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # ✅ ABSOLUTER Pfad – zwingend für Hydra
    initialize_config_dir(
        config_dir=config_root,
        job_name="sam2_inference",
        version_base=None,   # unterdrückt die Warnung sauber
    )

    # ⚠️ NUR Config-NAME, kein Pfad
    model = build_sam2(
        "sam2.1/sam2.1_hiera_l",
        checkpoint,
        device=device
    )

    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()
    return predictor


def run_sam2(sam_predictor, img, boxes):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(img)

        all_masks, all_scores = [], []

        for i in range(boxes.shape[0]):
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes[[i]],
                multimask_output=True,
            )

            best = np.argmax(scores)
            all_masks.append(masks[best])
            all_scores.append(scores[best])

        return np.stack(all_masks), np.stack(all_scores)

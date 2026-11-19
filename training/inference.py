"""CUDA inference adapter shared by replay and the local-network bridge."""
from __future__ import annotations

import sys
import os
from pathlib import Path

from .workflow import NUSCENES_CLASSES, load_yaml, sha256, transfer_state


def load_predictor(pcdet_root, config_path, checkpoint_path, device="cuda:0"):
    """Return a car-only predictor taking finite float32 N×4 x/y/z/intensity.

    The fifth feature is zero seconds of time lag, never LiDAR ring number.
    Output boxes use ego metres, [length,width,height], and yaw radians.
    Only car detections whose XY centre is within the shared 50 m region return.
    """
    import numpy as np
    import torch
    from easydict import EasyDict

    root = Path(pcdet_root).resolve()
    if not (root / "pcdet/models/detectors/voxelnext.py").is_file() or not (root / "pcdet/datasets/dataset.py").is_file():
        raise ValueError("Expected official VoxelNeXt/OpenPCDet source with its VoxelNeXt detector and DatasetTemplate")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("VoxelNeXt inference needs NVIDIA CUDA; this adapter does not implement a Metal backend")
    sys.path.insert(0, str(root))
    from pcdet.datasets.dataset import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils.spconv_utils import find_all_spconv_keys

    config = EasyDict(load_yaml(Path(config_path), root / "tools"))
    if config.CLASS_NAMES != NUSCENES_CLASSES or config.MODEL.NAME != "VoxelNeXt":
        raise ValueError("Expected the retained ten-class nuScenes VoxelNeXt configuration")
    torch_device = torch.device(device)
    with torch.cuda.device(torch_device):
        dataset = DatasetTemplate(dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES, training=False)
        model = build_network(model_cfg=config.MODEL, num_class=len(config.CLASS_NAMES), dataset=dataset)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError("Checkpoint needs model_state or state_dict")
        transfer = transfer_state(model, state, find_all_spconv_keys(model))
        model.to(torch_device).eval()

    def predict(points):
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 4 or not np.isfinite(cloud).all():
            raise ValueError("Expected finite N×4 x,y,z,intensity; remove invalid sensor returns first")
        if not len(cloud):
            return []
        features = np.column_stack((cloud, np.zeros(len(cloud), dtype=np.float32)))
        item = dataset.prepare_data(data_dict={"points": features, "frame_id": "live"})
        batch = dataset.collate_batch([item])
        if not len(batch.get("voxel_coords", batch.get("points", []))):
            return []
        with torch.cuda.device(torch_device), torch.inference_mode():
            load_data_to_gpu(batch)
            predictions, _ = model(batch)
            row = predictions[0]
            boxes = row["pred_boxes"].detach().cpu().numpy()
            scores = row["pred_scores"].detach().cpu().numpy()
            labels = row["pred_labels"].detach().cpu().numpy()
        result = []
        car_label = config.CLASS_NAMES.index("car") + 1
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == car_label and np.linalg.norm(box[:2]) <= 50.0:
                result.append({"center": box[:3].tolist(), "size": box[3:6].tolist(),
                               "yaw": float(box[6]), "score": float(score)})
        return result

    predict.metadata = {"backend": "voxelnext", "device": str(torch_device),
                        "gpu_name": torch.cuda.get_device_name(torch_device),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "checkpoint_sha256": sha256(Path(checkpoint_path)),
                        "config_sha256": sha256(Path(config_path)), "transfer": transfer,
                        "scored_class": "car", "max_range_m": 50.0, "time_lag_seconds": 0.0}
    return predict

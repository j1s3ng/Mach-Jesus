"""Pure preparation utilities; importing this module does not import CUDA."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

NUSCENES_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path, tools_root: Path, active=()) -> dict:
    """Resolve OpenPCDet base configs without importing its compiled extensions."""
    import yaml

    path = Path(path).resolve()
    if path in active:
        raise ValueError(f"Recursive base config: {path}")
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")

    def resolve(node):
        if isinstance(node, dict):
            result = {}
            reference = node.get("_BASE_CONFIG_")
            if reference:
                base = tools_root / reference
                if not base.is_file():
                    base = path.parent / reference
                result = load_yaml(base, tools_root, (*active, path))
            for key, child in node.items():
                if key == "_BASE_CONFIG_":
                    continue
                child = resolve(child)
                if isinstance(child, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], child)
                else:
                    result[key] = child
            return result
        return [resolve(x) for x in node] if isinstance(node, list) else node

    return resolve(value)


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def split_scenes(train_infos, val_infos, token_to_scene, development_fraction=0.2, seed=42):
    """Never split neighboring frames independently or borrow released val scenes."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one")

    def scenes(infos):
        tokens = [str(info["token"]) for info in infos]
        if len(tokens) != len(set(tokens)):
            raise ValueError("Duplicate sample token in an info collection")
        missing = set(tokens) - token_to_scene.keys()
        if missing:
            raise ValueError(f"{len(missing)} info tokens are missing from sample.json")
        return {token_to_scene[token] for token in tokens}

    train_scenes, val_scenes = scenes(train_infos), scenes(val_infos)
    if train_scenes & val_scenes:
        raise ValueError("Released train and validation scenes overlap")
    if len(train_scenes) < 2 or not val_scenes:
        raise ValueError("Need at least two training scenes and one held-out validation scene")
    if {str(x["token"]) for x in train_infos} & {str(x["token"]) for x in val_infos}:
        raise ValueError("Released train and validation samples overlap")
    ordered = sorted(train_scenes)
    random.Random(seed).shuffle(ordered)
    count = max(1, min(len(ordered) - 1, round(len(ordered) * development_fraction)))
    dev_scenes = set(ordered[:count])
    fit = [x for x in train_infos if token_to_scene[str(x["token"])] not in dev_scenes]
    dev = [x for x in train_infos if token_to_scene[str(x["token"])] in dev_scenes]
    manifest = {
        "seed": seed, "development_fraction": development_fraction,
        "fit_scenes": sorted(train_scenes - dev_scenes), "development_scenes": sorted(dev_scenes),
        "heldout_scenes": sorted(val_scenes),
        "fit_tokens": sorted(str(x["token"]) for x in fit),
        "development_tokens": sorted(str(x["token"]) for x in dev),
        "heldout_tokens": sorted(str(x["token"]) for x in val_infos),
    }
    return fit, dev, manifest


def configure_source_model(source_cfg, vmax_dataset_cfg, data_root, info_paths):
    """Preserve source weights' geometry/head; adapt only the dataset protocol."""
    cfg = copy.deepcopy(source_cfg)
    if cfg.get("CLASS_NAMES") != NUSCENES_CLASSES:
        raise ValueError("Base config must preserve the original ten nuScenes classes in order")
    if cfg.get("MODEL", {}).get("NAME") != "VoxelNeXt":
        raise ValueError("Base model must be VoxelNeXt")
    features = cfg["DATA_CONFIG"]["POINT_FEATURE_ENCODING"]
    expected = ["x", "y", "z", "intensity", "timestamp"]
    if features["used_feature_list"] != expected or features["src_feature_list"] != expected:
        raise ValueError("Expected five source features x,y,z,intensity,time-lag")
    data = copy.deepcopy(vmax_dataset_cfg)
    for key in ("POINT_CLOUD_RANGE", "DATA_PROCESSOR", "POINT_FEATURE_ENCODING", "DATA_AUGMENTOR"):
        data[key] = copy.deepcopy(cfg["DATA_CONFIG"][key])
    data.update({
        "DATA_PATH": str(Path(data_root).resolve()), "VERSION": "v1.0-mini",
        "MAX_SWEEPS": 1, "BALANCED_RESAMPLING": False,
        "PRED_VELOCITY": True, "SET_NAN_VELOCITY_TO_ZEROS": True,
        "DATA_SPLIT": {"train": "train", "test": "val"},
        "INFO_PATH": {"train": [str(Path(info_paths["fit"]).resolve())], "test": [str(Path(info_paths["development"]).resolve())]},
    })
    # A database made from all released train scenes leaks development GT into fit.
    # All experiments share these geometric augmentations and omit GT pasting.
    augment = data["DATA_AUGMENTOR"]
    augment["AUG_CONFIG_LIST"] = [x for x in augment["AUG_CONFIG_LIST"] if x["NAME"] != "gt_sampling"]
    augment["DISABLE_AUG_LIST"] = ["gt_sampling"]
    for processor in data["DATA_PROCESSOR"]:
        if processor["NAME"] == "shuffle_points":
            processor["SHUFFLE_ENABLED"]["test"] = False
    cfg["DATA_CONFIG"] = data
    cfg["MODEL"]["DENSE_HEAD"]["POST_PROCESSING"]["SCORE_THRESH"] = 0.001
    # The model retains its velocity output tensors for full checkpoint reuse, but
    # Vmax's perception labels do not provide trustworthy velocity targets.
    weights = cfg["MODEL"]["DENSE_HEAD"]["LOSS_CONFIG"]["LOSS_WEIGHTS"]["code_weights"]
    if len(weights) != 10:
        raise ValueError("Expected the nuScenes ten-code regression head including velocity")
    weights[-2:] = [0.0, 0.0]
    return cfg


def experiment_specs(epochs=20, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    return [
        {"name": "pretrained", "mode": "baseline", "lr": None, "epochs": 0, "seed": seed},
        {"name": "head_1e-4", "mode": "head", "lr": 1e-4, "epochs": epochs, "seed": seed},
        {"name": "full_1e-4", "mode": "full", "lr": 1e-4, "epochs": epochs, "seed": seed},
        {"name": "full_1e-5", "mode": "full", "lr": 1e-5, "epochs": epochs, "seed": seed},
    ]


def transfer_state(model, checkpoint_state, spconv_keys=()):
    """Require every learned tensor and BN statistic, allowing known spconv layouts.

    Counter buffers may be absent in older checkpoints. The report records those
    cases explicitly; no random learned weight may silently survive loading.
    """
    import torch

    target = model.state_dict()
    state = {key.removeprefix("module."): value for key, value in checkpoint_state.items()}
    if len(state) != len(checkpoint_state):
        raise ValueError("Checkpoint has colliding DDP-prefixed keys")
    updated, entries, missing, mismatched = {}, [], [], []
    for key, value in target.items():
        disk = state.get(key)
        conversion = None
        if disk is None:
            if key.endswith("num_batches_tracked") or key == "global_step":
                updated[key] = torch.zeros_like(value)
                entries.append({"key": key, "status": "reset_counter"})
                continue
            missing.append(key)
            continue
        if tuple(disk.shape) != tuple(value.shape) and key in spconv_keys:
            if disk.ndim >= 2 and tuple(disk.transpose(-1, -2).shape) == tuple(value.shape):
                disk = disk.transpose(-1, -2).contiguous()
                conversion = "spconv_transpose_last_two"
            elif disk.ndim == 5 and tuple(disk.permute(4, 0, 1, 2, 3).shape) == tuple(value.shape):
                disk = disk.permute(4, 0, 1, 2, 3).contiguous()
                conversion = "spconv_output_first"
        if tuple(disk.shape) != tuple(value.shape):
            mismatched.append({"key": key, "checkpoint": list(disk.shape), "model": list(value.shape)})
            continue
        updated[key] = disk
        entries.append({"key": key, "status": "loaded", "shape": list(value.shape), "conversion": conversion})
    report = {"loaded_tensors": sum(x["status"] == "loaded" for x in entries), "model_tensors": len(target),
              "entries": entries, "missing": missing, "mismatched": mismatched,
              "unused_checkpoint_keys": sorted(state.keys() - target.keys())}
    if missing or mismatched:
        raise ValueError("Pretrained weights do not fully match the model: " + json.dumps(report))
    model.load_state_dict(updated, strict=True)
    return report


def set_training_mode(model, mode):
    """Freeze both backbone gradients and running statistics in head-only runs."""
    if mode not in {"baseline", "head", "full"}:
        raise ValueError(f"Unknown mode {mode}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == "full" or (mode == "head" and name.startswith("dense_head.")))
    model.train(mode != "baseline")
    if mode == "head":
        for name, module in model.named_children():
            if name != "dense_head":
                module.eval()
        model.dense_head.train()
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

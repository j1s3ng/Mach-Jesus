"""Generate scene-disjoint, matched-compute experiments from exported A2RL data."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

from .workflow import configure_source_model, experiment_specs, load_yaml, sha256, split_scenes


def prepare(args):
    import numpy as np
    import yaml

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a prepared study: {output}")
    pcdet = args.openpcdet_root.resolve()
    tools = pcdet / "tools"
    source = args.source_config.resolve() if args.source_config else tools / "cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
    vmax = tools / "cfgs/dataset_configs/vmax_dataset.yaml"
    if not vmax.is_file():
        raise FileNotFoundError("Use A2RL's bundled code/vmax_OpenPCDet fork; stock OpenPCDet is insufficient")
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Download the official nuScenes VoxelNeXt pretrained checkpoint first")
    metadata = args.data_root.resolve() / "v1.0-mini"
    samples_path = metadata / "sample.json"
    samples = json.loads(samples_path.read_text())
    token_to_scene = {str(row["token"]): str(row["scene_token"]) for row in samples}
    if len(token_to_scene) != len(samples):
        raise ValueError("Duplicate tokens in sample.json")
    # These are trusted local official export files; Python pickle is executable.
    with args.train_info.open("rb") as handle:
        train_infos = pickle.load(handle)
    with args.val_info.open("rb") as handle:
        val_infos = pickle.load(handle)
    fit, dev, split = split_scenes(train_infos, val_infos, token_to_scene, args.development_fraction, args.seed)
    # Validate the source layout rather than assuming all exporters share a shape.
    for info in [*fit, *dev, *val_infos]:
        boxes = np.asarray(info.get("gt_boxes"))
        names = np.asarray(info.get("gt_names"))
        if boxes.ndim != 2 or boxes.shape[1] not in (7, 9) or len(boxes) != len(names):
            raise ValueError(f"Unsupported GT box layout for token {info['token']}")
        if set(names.tolist()) - {"car"}:
            raise ValueError("Expected the A2RL car-only annotation taxonomy")
        if boxes.shape[1] == 7:
            boxes = np.concatenate([boxes, np.zeros((len(boxes), 2), dtype=boxes.dtype)], axis=1)
        info["gt_boxes"] = boxes
        info["gt_names"] = names
        if not np.isfinite(boxes[:, :7]).all() or (boxes[:, 3:6] <= 0).any():
            raise ValueError("Nonfinite or invalid boxes in the official export")
        point_file = metadata / info["lidar_path"]
        if not point_file.is_file() or point_file.stat().st_size % 20:
            raise ValueError(f"Expected an existing float32 five-column LiDAR file: {point_file}")
    output.mkdir(parents=True, exist_ok=True)
    info_paths = {}
    for name, values in (("fit", fit), ("development", dev), ("heldout", val_infos)):
        path = output / f"infos_{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(values, handle, protocol=4)
        info_paths[name] = path
    cfg = configure_source_model(load_yaml(source, tools), load_yaml(vmax, tools), args.data_root, info_paths)
    # Fail if the fork's loader stops discarding ring/replacing it with time lag.
    loader_file = pcdet / "pcdet/datasets/nuscenes/nuscenes_dataset.py"
    loader_source = loader_file.read_text()
    if "get_lidar_with_sweeps" not in loader_source or "[:, :4]" not in loader_source:
        raise ValueError("Inspect this fork's point loader: expected ring removal and one-sweep zero time-lag")
    cfg["OPTIMIZATION"]["NUM_EPOCHS"] = args.epochs
    cfg["OPTIMIZATION"]["BATCH_SIZE_PER_GPU"] = args.batch_size
    # Our runner uses constant-LR AdamW, identical across the three fits.
    cfg["OPTIMIZATION"]["OPTIMIZER"] = "adamw"
    specs = experiment_specs(args.epochs, args.seed)
    for spec in specs:
        run_cfg = copy.deepcopy(cfg)
        if spec["lr"] is not None:
            run_cfg["OPTIMIZATION"]["LR"] = spec["lr"]
        path = output / f"{spec['name']}.yaml"
        path.write_text(yaml.safe_dump(run_cfg, sort_keys=False))
        spec["config"] = str(path)
        spec["config_sha256"] = sha256(path)
    source_identity = {
        str(path.resolve()): sha256(path) for path in [source, vmax, loader_file, args.train_info, args.val_info, samples_path, args.checkpoint]
    }
    for name, path in info_paths.items():
        source_identity[str(path)] = sha256(path)
    split_path = output / "split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    first = np.fromfile(metadata / fit[0]["lidar_path"], dtype=np.float32).reshape(-1, 5)
    if not np.isfinite(first[:, :4]).all():
        raise ValueError("First input cloud has nonfinite coordinates or intensity")
    plan = {
        "status": "prepared_not_executed", "openpcdet_root": str(pcdet),
        "pretrained_checkpoint": str(args.checkpoint.resolve()), "source_identity": source_identity,
        "split_manifest": str(split_path), "split_sha256": sha256(split_path),
        "info_paths": {key: str(value) for key, value in info_paths.items()},
        "experiments": specs, "common_evaluation_radius_m": 50.0,
        "protocol": {
            "classes_retained": cfg["CLASS_NAMES"], "scored_class": "car", "sweeps": 1,
            "raw_point_layout": ["x", "y", "z", "raw_intensity", "ring"],
            "model_point_layout": ["x", "y", "z", "raw_intensity", "zero_time_lag"],
            "intensity_normalization": "unchanged, matching the source loader; sensor distributions still need inspection",
            "gt_sampling": "disabled in every fit to prevent development-scene database leakage",
            "velocity_loss": "zero-weight; velocity output tensors retained for checkpoint compatibility",
            "optimizer": "AdamW, constant LR, source weight decay and gradient clipping",
            "coordinate_frame": "official Vmax vehicle_rear_axle_middle_ground; verify axes in a plotted sample before fitting",
            "single_sweep_limitation": "Original checkpoint was trained with ten sweeps; all target comparisons use identical one-sweep input",
        },
        "first_fit_cloud": {
            "points": len(first), "xyz_min": first[:, :3].min(0).tolist() if len(first) else None,
            "xyz_max": first[:, :3].max(0).tolist() if len(first) else None,
            "intensity_percentiles": np.percentile(first[:, 3], [0, 1, 50, 99, 100]).tolist() if len(first) else None,
        },
        "unexecuted": ["CUDA forward/backward pilot", "pretrained transfer audit", "all training", "all real-data evaluations"],
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, help="Official nuScenes VoxelNeXt config matching the checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Export root containing v1.0-mini/")
    parser.add_argument("--train-info", type=Path, required=True)
    parser.add_argument("--val-info", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    args = parser.parse_args()
    try:
        plan = prepare(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps({"status": plan["status"], "plan": str(args.output / "plan.json")}, indent=2))


if __name__ == "__main__":
    main()

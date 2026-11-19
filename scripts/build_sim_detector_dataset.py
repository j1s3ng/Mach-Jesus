#!/usr/bin/env python3
"""Build a simulator-only detector dataset from measured clouds and offline GT.

All ego views of a physical run stay in the same split. The preferred capture
is cars/Rxx/points_manifest.jsonl plus a Supervisor-only gt_poses.jsonl. Legacy
portable recording/manifest.jsonl with ego-frame gt_boxes is also accepted.
This script is offline; its GT output must never be read by race controllers.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
from race_lidar.io import COORDINATE_FRAME, validate_box, validate_points
from scripts.training_sources import load_training_releases, validate_training_source
from scripts.cv_training_sources import archive_training_sources, validate_export_training_sources


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    with Path(path).open() as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f"Invalid JSON at {path}:{index}") from error


def rotation(car):
    if "rotation" in car or "rotation_matrix" in car:
        value = np.asarray(car.get("rotation", car.get("rotation_matrix")), float).reshape(3, 3)
    else:
        yaw = float(car.get("yaw_rad", car.get("yaw", float("nan"))))
        c, s = math.cos(yaw), math.sin(yaw)
        value = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    if not np.isfinite(value).all() or not np.allclose(value.T@value, np.eye(3), atol=1e-3) or np.linalg.det(value) < .99:
        raise ValueError("Ground-truth rotation must be finite, orthonormal and right-handed")
    return value


def local_spec(car, config):
    default = config.get("detector_gt_box", config.get("gt_box", {}))
    center = car.get("box_center_local_m", default.get("center_local_m"))
    size = car.get("size_m", car.get("box_size_m", default.get("size_m")))
    if center is None or size is None:
        raise ValueError("Pose GT needs explicit box_center_local_m and size_m (or config.detector_gt_box)")
    center, size = np.asarray(center, float), np.asarray(size, float)
    if center.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[center, size]).all() or np.any(size <= 0):
        raise ValueError("Invalid physical-asset box convention")
    return center, size


def world_pose_boxes(cars, ego_id, config, ground_z=0., include_ego=False):
    if isinstance(cars, dict):
        cars = [{"car_id": key, **value} for key, value in cars.items()]
    by_id = {str(car["car_id"]): car for car in cars}
    if ego_id not in by_id:
        raise ValueError(f"Missing ego {ego_id} in matching Supervisor pose row")
    ego = by_id[ego_id]
    ego_position = np.asarray(ego["position_m"], float)
    ego_rotation = rotation(ego)
    ego_yaw = math.atan2(ego_rotation[1, 0], ego_rotation[0, 0])
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    leveled_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    origin = np.array([ego_position[0], ego_position[1], float(ground_z)])
    boxes = []
    signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
    for car_id, car in by_id.items():
        if car_id == ego_id and not include_ego:
            continue
        center, size = local_spec(car, config)
        orientation = rotation(car)
        position = np.asarray(car["position_m"], float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("GT positions need three finite world coordinates")
        corners_world = (center+signs*size/2)@orientation.T+position
        corners = (corners_world-origin)@leveled_rotation
        heading_world = orientation[:, 0]
        heading = heading_world@leveled_rotation
        yaw = math.atan2(heading[1], heading[0])
        cy, sy = math.cos(yaw), math.sin(yaw)
        axes = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]])
        aligned = corners@axes
        low, high = aligned.min(0), aligned.max(0)
        box_center = ((low+high)/2)@axes.T
        boxes.append({"car_id": car_id, "center": box_center.tolist(), "size": (high-low).tolist(), "yaw": yaw,
                      "label_geometry": "ego-upright yaw box enclosing 8 transformed physical asset corners"})
    return boxes


def count_inside(points, box, tolerance=.03):
    validate_box(box)
    delta = np.asarray(points[:, :3], float)-np.asarray(box["center"], float)
    c, s = math.cos(box["yaw"]), math.sin(box["yaw"])
    local = delta@np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    return int(np.count_nonzero(np.all(np.abs(local) <= np.asarray(box["size"])/2+tolerance, axis=1)))


def capture_manifests(run):
    paths = set()
    for pattern in ("cars/*/points_manifest.jsonl", "cars/*/lidar_manifest.jsonl", "cars/*/detector_manifest.jsonl",
                    "cars/*/recording/manifest.jsonl", "recording/manifest.jsonl", "manifest.jsonl"):
        paths.update(run.glob(pattern))
    if not paths:
        raise ValueError(f"No measured point manifest found under {run}")
    return sorted(paths)


def resolve_points(run, manifest, relative):
    for candidate in (manifest.parent/relative, run/relative):
        path = candidate.resolve()
        if path.is_file() and path.is_relative_to(run.resolve()):
            return path
    raise ValueError(f"Point path is missing or outside its physical run: {relative}")


def build(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose a fresh dataset output; existing exports are immutable")
    reuse = getattr(args, "training_export", None)
    train_roots = [path.resolve() for path in (args.train_runs or [])]
    if bool(reuse) == bool(train_roots): raise ValueError("Choose exactly one of --train-runs or --training-export")
    release_path = getattr(args, "training_releases", None)
    releases = load_training_releases(release_path) if release_path else ()
    for run in train_roots:
        validate_training_source(run, require_config=True, releases=releases)
    if reuse: validate_export_training_sources(reuse, release_path)
    val_roots = [path.resolve() for path in args.val_runs] if args.val_runs else train_roots if args.diagnostic_only else []
    if not val_roots: raise ValueError("Independent validation runs are required unless --diagnostic-only is explicit")
    authority_path = getattr(args, "qualification_authority", None)
    authority = json.loads(authority_path.read_text()) if authority_path else None
    if authority:
        declared = authority["reserved_unopened_qualification_groups"] if isinstance(authority, dict) else [row["source_path"] for row in authority]
        permitted = {str(Path(path).resolve()) for path in declared}
        if {str(path) for path in val_roots} != permitted:
            raise ValueError("Qualification export roots differ from explicit reserved authority")
    else:
        for run in val_roots: validate_training_source(run, require_config=True)
    if set(train_roots)&set(val_roots) and not args.diagnostic_only:
        raise ValueError("The same entire physical run cannot appear in train and validation")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"points").mkdir()
    groups, identities, point_splits, source_files, report_splits = {}, set(), {}, {}, {}
    input_counts, excluded_duplicates = {"train": 0, "val": 0}, []
    input_counts_by_group = {"train": {}, "val": {}}
    for split, roots in (("train", train_roots), ("val", val_roots)):
        if split == "train" and reuse:
            source_metadata = json.loads((reuse/"dataset.json").read_text())
            for key, expected in (("min_measured_points_per_box", args.min_points), ("evaluation_radius_m", args.max_range)):
                if source_metadata[key] != expected: raise ValueError("Frozen training and qualification label contracts differ")
            values = list(jsonl(reuse/"train.jsonl"))
            for row in values:
                source = (reuse/row["points_file"]).resolve()
                if not source.is_relative_to(reuse.resolve()) or sha256(source) != row["points_sha256"]:
                    raise ValueError("Frozen training cloud changed")
                os.link(source, args.output/row["points_file"])
                point_splits[row["points_sha256"]] = "train";groups[row["scene_id"]] = "train"
                input_counts_by_group["train"][row["scene_id"]] = input_counts_by_group["train"].get(row["scene_id"], 0)+1
            (args.output/"train.jsonl").write_bytes((reuse/"train.jsonl").read_bytes())
            source_files.update(source_metadata["source_file_sha256"])
            input_counts["train"] = len(values);report_splits["train"] = source_metadata["splits"]["train"]
            continue
        output_rows = []
        for run in roots:
            config_path = run/"config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            group = str(config.get("scene_group_id", config.get("run_id", run.name)))
            if group in groups and groups[group] != split and not args.diagnostic_only:
                raise ValueError(f"Physical scenario/run group leaks across splits: {group}")
            groups[group] = split
            if config_path.exists(): source_files[str(config_path)] = sha256(config_path)
            gt_path = run/"gt_poses.jsonl"
            poses = {}
            if gt_path.exists():
                source_files[str(gt_path)] = sha256(gt_path)
                for row in jsonl(gt_path):
                    timestamp = int(row["timestamp_ns"])
                    if timestamp in poses: raise ValueError("Duplicate Supervisor GT timestamp")
                    poses[timestamp] = row
            exported = 0
            for manifest in capture_manifests(run):
                source_files[str(manifest)] = sha256(manifest)
                for index, entry in enumerate(jsonl(manifest)):
                    if index % args.stride: continue
                    if args.max_frames_per_run and exported >= args.max_frames_per_run: break
                    if entry.get("coordinate_frame", COORDINATE_FRAME) != COORDINATE_FRAME:
                        raise ValueError("Point capture must already use the leveled x-forward/y-left/z-up ego-ground frame")
                    car_id = str(entry.get("car_id", manifest.parent.name if manifest.parent.name.startswith("R") else "ego"))
                    timestamp = int(entry["timestamp_ns"])
                    source = resolve_points(run, manifest, entry["points_file"])
                    input_counts[split] += 1
                    input_counts_by_group[split][group] = input_counts_by_group[split].get(group, 0)+1
                    digest = sha256(source)
                    # Deterministic input-only filtering precedes every GT
                    # transform/visibility calculation. Labels and predictions
                    # cannot affect this qualification subset selection.
                    if split == "val" and point_splits.get(digest) == "train" and args.drop_duplicate_validation:
                        excluded_duplicates.append({"scene_id": group, "car_id": car_id,
                                                    "frame_id": str(entry.get("frame_id", index)), "timestamp_ns": timestamp,
                                                    "points_sha256": digest, "reason": "exact_training_cloud_duplicate"})
                        continue
                    points = validate_points(np.load(source, allow_pickle=False))
                    if not len(points): continue
                    if "gt_boxes" in entry:
                        boxes = entry["gt_boxes"]
                        label_source = "legacy_offline_ego_frame_gt"
                    else:
                        if timestamp not in poses:
                            raise ValueError(f"No exact Supervisor GT pose timestamp {timestamp} for {source}; no nearest-frame joining")
                        boxes = world_pose_boxes(poses[timestamp]["cars"], car_id, config,
                                                 entry.get("ego_ground_z_m", config.get("ego_ground_z_m", 0.)))
                        label_source = "offline_supervisor_pose_join"
                    visible, excluded = [], []
                    for box in boxes:
                        if str(box.get("class", "car")) != "car" or str(box.get("car_id", "")) == car_id: continue
                        validate_box(box)
                        radial = float(np.linalg.norm(np.asarray(box["center"])[:2]))
                        count = count_inside(points, box) if radial <= args.max_range else 0
                        value = {**box, "visible_points": count}
                        if radial <= args.max_range and count >= args.min_points:
                            visible.append(value)
                        else:
                            excluded.append({**value, "exclusion": "out_of_range" if radial > args.max_range else "insufficient_measured_points"})
                    if args.positive_only and not visible: continue
                    identity = (group, car_id, timestamp)
                    identity_scope = (split, *identity)
                    if identity_scope in identities: raise ValueError(f"Duplicate physical frame: {identity}")
                    identities.add(identity_scope)
                    if digest in point_splits and point_splits[digest] != split and not args.diagnostic_only:
                        raise ValueError("Identical measured point file leaks across train and validation")
                    point_splits[digest] = split
                    token = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
                    relative = f"points/{token}.npy"
                    shutil.copyfile(source, args.output/relative)
                    output_rows.append({"scene_id": group, "frame_id": token, "car_id": car_id, "timestamp_ns": timestamp,
                                        "points_file": relative, "points_sha256": digest, "coordinate_frame": COORDINATE_FRAME,
                                        "gt_boxes": visible, "excluded_gt_boxes": excluded, "label_source": label_source,
                                        "scenario_id": config.get("scenario_id", config.get("scenario", group)), "seed": config.get("seed"),
                                        "source_run": str(run), "source_frame_id": str(entry.get("frame_id", index))})
                    exported += 1
        if not output_rows: raise ValueError(f"No exported {split} frames; inspect labels, timestamp joins and visibility threshold")
        (args.output/(split+".jsonl")).write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in output_rows))
        report_splits[split] = {"frames": len(output_rows), "positive_frames": sum(bool(row["gt_boxes"]) for row in output_rows),
                                "visible_car_boxes": sum(len(row["gt_boxes"]) for row in output_rows),
                                "groups": sorted({row["scene_id"] for row in output_rows})}
    report = {"dataset": "measured_Webots_LiDAR_with_offline_supervisor_or_portable_GT", "simulated_only": True,
              "a2rl_data_used": False, "gt_for_control": False,
              "diagnostic_only": args.diagnostic_only, "validation_independent": not args.diagnostic_only,
              "split_unit": "same-run gradient smoke; no validation claim" if args.diagnostic_only else "entire physical run/scenario group including every ego view",
              "positive_only_selection": args.positive_only, "stride": args.stride, "frame_cap_per_run": args.max_frames_per_run,
              "input_frame_counts_before_dedup": input_counts,
              "input_frame_counts_by_group": input_counts_by_group,
              "validation_exact_cloud_deduplication": args.drop_duplicate_validation,
              "excluded_duplicate_validation_frames": excluded_duplicates,
              "qualification_subset_scope": "run-separated, exact-cloud-deduplicated moving subset" if args.drop_duplicate_validation else None,
              "min_measured_points_per_box": args.min_points, "evaluation_radius_m": args.max_range,
              "point_in_box_tolerance_m": .03, "timestamp_join": "exact integer nanoseconds only",
              "ground_origin": "ego rear axle XY and documented ground Z; yaw leveled; matches racer lidar_points transform",
              "visibility_limit": "Inside-box return evidence, not perfect instance attribution; low-return objects below threshold are excluded.",
              "validation_purpose": "qualification_only_no_fitting_or_development_selection" if authority else "development",
              "qualification_authority_sha256": sha256(authority_path) if authority_path else None,
              "qualification_authority": {"reserved_unopened_qualification_groups": declared} if authority else None,
              "reused_training_export": {"dataset_metadata_sha256": sha256(reuse/"dataset.json"), "manifest_sha256": sha256(reuse/"train.jsonl")} if reuse else None,
              "splits": report_splits, "source_file_sha256": source_files,
              "manifest_sha256": {name: sha256(args.output/(name+".jsonl")) for name in ("train", "val")}}
    (args.output/"dataset.json").write_text(json.dumps(report, indent=2))
    archive_training_sources(args.output, release_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-runs", nargs="+", type=Path)
    parser.add_argument("--training-export", type=Path, help="Reuse exact audited training manifest/point hard links")
    parser.add_argument("--qualification-authority", type=Path, help="Explicit exact reserved run list; exported validation is evaluation-only")
    parser.add_argument("--training-releases", type=Path, help="Exact immutable authorized source exceptions; no generic override")
    parser.add_argument("--val-runs", nargs="+", type=Path)
    parser.add_argument("--diagnostic-only", action="store_true", help="Allow same-run reuse only for a bounded gradient smoke; no independent validation claim")
    parser.add_argument("--drop-duplicate-validation", action="store_true", help="Input-only exclusion of validation clouds identical to frozen training clouds, recorded before scoring")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--max-range", type=float, default=50.)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-run", type=int, default=0)
    parser.add_argument("--positive-only", action="store_true", help="Useful for a tiny positive-label gradient smoke; record this selection")
    args = parser.parse_args()
    if args.min_points < 1 or args.max_range <= 0 or args.stride < 1 or args.max_frames_per_run < 0:
        parser.error("Need positive visibility/range/stride and nonnegative frame cap")
    print(json.dumps(build(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

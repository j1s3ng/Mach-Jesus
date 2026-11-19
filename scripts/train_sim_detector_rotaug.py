#!/usr/bin/env python3
"""Isolated simulator-only VoxelNeXt adaptation; live services stay unchanged.

Uses the official source/config/strict checkpoint transfer and a JSON/NPY
dataset from build_sim_detector_dataset.py. Ground truth enters training loss
and offline evaluation only; evaluation forward passes contain no GT fields.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from training.workflow import NUSCENES_CLASSES, load_yaml, set_training_mode, sha256, transfer_state
from race_lidar.metrics import evaluate_frames
from scripts.cv_training_sources import validate_export_training_sources
from training.quarter_rotation import QuarterTurnAugmentation


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def read_split(root, name):
    rows = [json.loads(line) for line in (root/(name+".jsonl")).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f"Empty {name} split")
    for row in rows:
        point_path = (root/row["points_file"]).resolve()
        if not point_path.is_relative_to(root.resolve()) or sha256(point_path) != row["points_sha256"]:
            raise ValueError("Measured point data changed or escapes dataset root")
    return rows


def validate_evaluation_authority(metadata, candidate_checkpoint, authority_path):
    if metadata.get("validation_purpose") == "qualification_only_no_fitting_or_development_selection":
        if not candidate_checkpoint or not authority_path or sha256(authority_path) != metadata["qualification_authority_sha256"]:
            raise ValueError("Reserved CV qualification inputs require evaluation-only mode and exact qualification authority")


def comparison_metrics(predictions, truth):
    """Custom held-out diagnostics at the runtime's useful confidence levels."""
    metrics = evaluate_frames(predictions, truth)
    truth_by_id = {(row["scene_id"], row["frame_id"]): row["gt_boxes"] for row in truth}
    empty_ids = {key for key, boxes in truth_by_id.items() if not boxes}
    metrics["empty_frames"] = len(empty_ids)
    operating = {}
    for threshold in (.1, .2, .25, .35, .5, .6, .75):
        filtered = [{**row, "boxes": [box for box in row["boxes"] if box["score"] >= threshold]} for row in predictions]
        measured = evaluate_frames(filtered, truth)
        empty_fp = sum(len(row["boxes"]) for row in filtered if (row["scene_id"], row["frame_id"]) in empty_ids)
        operating[str(threshold)] = {key: measured[key] for key in
                                    ("precision", "recall", "true_positives", "false_positives", "false_negatives")}
        operating[str(threshold)].update(empty_frame_false_positives=empty_fp,
                                         false_positives_per_empty_frame=empty_fp/len(empty_ids) if empty_ids else None)
    metrics["operating_points"] = operating
    bins = [(0., 15.), (15., 30.), (30., 50.000001)]
    totals = np.zeros(3, int);hits = np.zeros(3, int);dimension_errors = []
    for prediction in predictions:
        boxes = truth_by_id[(prediction["scene_id"], prediction["frame_id"])]
        radii = [float(np.linalg.norm(np.asarray(box["center"])[:2])) for box in boxes]
        categories = [next((i for i, (low, high) in enumerate(bins) if low <= radius < high), None) for radius in radii]
        for category in categories:
            if category is not None: totals[category] += 1
        used = set()
        for box in sorted(prediction["boxes"], key=lambda row: -row["score"]):
            if box["score"] < .25: continue
            candidates = [(float(np.linalg.norm(np.asarray(box["center"])[:2]-np.asarray(gt["center"])[:2])), i)
                          for i, gt in enumerate(boxes) if i not in used]
            distance, selected = min(candidates, default=(float("inf"), -1))
            if distance <= 2.:
                used.add(selected)
                if categories[selected] is not None: hits[categories[selected]] += 1
                dimension_errors.append(np.abs(np.asarray(box["size"])-np.asarray(boxes[selected]["size"])))
    metrics["visible_target_recall_by_range_at_score_0p25"] = {
        f"{int(low)}_{min(50,int(high))}m": {"targets": int(total), "matched": int(hit), "recall": float(hit/total) if total else None}
        for (low, high), total, hit in zip(bins, totals, hits)}
    metrics["matched_dimension_mae_lwh_m_at_score_0p25"] = np.mean(dimension_errors, axis=0).tolist() if dimension_errors else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pcdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Immutable original pretrained checkpoint")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Compare this frozen candidate with pretrained weights; perform no fitting")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("head", "full"), default="head")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Positive value selects a bounded gradient smoke")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-max-frames", type=int, default=0)
    parser.add_argument("--balance-scenes", action="store_true", help="Equal scene sampling with replacement, training split only")
    parser.add_argument("--baseline-cache", type=Path, help="Reuse identical frozen-input pretrained predictions; identities are verified")
    parser.add_argument("--mine-checkpoint", type=Path, help="Mine hard negatives on TRAINING EMPTY frames only; restore original weights before fitting")
    parser.add_argument("--mine-score", type=float, default=.2)
    parser.add_argument("--empty-loss-weight", type=float, default=1.)
    parser.add_argument("--negative-batch-fraction", type=float, default=0., help="Use homogeneous empty/positive batches;0 keeps ordinary frame batches")
    parser.add_argument("--backbone-lr", type=float, help="Separate full-mode backbone LR, at most1e-5; --lr remains head LR")
    parser.add_argument("--freeze-batchnorm", action="store_true", help="Freeze running statistics while retaining trainable affine parameters")
    parser.add_argument("--training-releases", type=Path, help="Exact immutable authorized source exception manifest")
    parser.add_argument("--qualification-authority", type=Path, help="Exact reserved-run authority required for qualification-only evaluation")
    parser.add_argument("--mine-all-training", action="store_true", help="Mine scored false positives on all training frames, not just empty frames")
    parser.add_argument("--mine-allow-training-subset", action="store_true", help="Allow a declared miner trained only on a subset of current training scenes")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lr) <= 0 or args.max_steps < 0 or args.eval_max_frames < 0:
        parser.error("Invalid epoch/batch/LR/step limits")
    if not 0 <= args.negative_batch_fraction <= 1 or args.empty_loss_weight <= 0 or not 0 <= args.mine_score <= 1:
        parser.error("Invalid hard-negative/empty-batch settings")
    if args.backbone_lr is not None and (args.mode != "full" or not 0 < args.backbone_lr <= 1e-5 or args.lr > 1e-4):
        parser.error("Separate backbone LR requires full mode, backbone<=1e-5, head<=1e-4")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new run directory; existing checkpoints/results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "preparing", "simulated_only": True, "a2rl_training_executed": False,
              "training_executed": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output/"status.json", status)
    try:
        import torch
        from easydict import EasyDict
        if not torch.cuda.is_available(): raise RuntimeError("Actual VoxelNeXt adaptation requires NVIDIA CUDA")
        torch.set_num_threads(2)
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(0)
        sys.path.insert(0, str(args.pcdet_root.resolve()))
        from pcdet.datasets.dataset import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils.spconv_utils import find_all_spconv_keys
        metadata = json.loads((args.dataset/"dataset.json").read_text())
        validate_evaluation_authority(metadata, args.evaluate_checkpoint, args.qualification_authority)
        source_eligibility = validate_export_training_sources(args.dataset, args.training_releases)
        diagnostic = bool(metadata.get("diagnostic_only"))
        if diagnostic and not args.max_steps and not args.evaluate_checkpoint:
            raise ValueError("Same-run diagnostic datasets require a bounded --max-steps gradient smoke")
        for split in ("train", "val"):
            if sha256(args.dataset/(split+".jsonl")) != metadata["manifest_sha256"][split]:
                raise ValueError("Frozen dataset manifest changed")
        train_rows, val_rows = read_split(args.dataset, "train"), read_split(args.dataset, "val")
        if {r["scene_id"] for r in train_rows}&{r["scene_id"] for r in val_rows} and not diagnostic:
            raise ValueError("Physical scenario groups overlap across splits")
        if {r["points_sha256"] for r in train_rows}&{r["points_sha256"] for r in val_rows} and not diagnostic:
            raise ValueError("Measured cloud hashes overlap across splits")
        if args.eval_max_frames: val_rows = val_rows[:args.eval_max_frames]
        config = EasyDict(load_yaml(args.config, args.pcdet_root/"tools"))
        if config.CLASS_NAMES != NUSCENES_CLASSES or config.MODEL.NAME != "VoxelNeXt":
            raise ValueError("Preserve the original ten-class nuScenes VoxelNeXt architecture")
        config.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights[-2:] = [0., 0.]
        for processor in config.DATA_CONFIG.DATA_PROCESSOR:
            if processor.NAME == "shuffle_points": processor.SHUFFLE_ENABLED.test = False
        # Evaluation-mode preprocessing still attaches numeric GT classes when
        # supplied, but avoids DatasetTemplate's recursive resampling of empty
        # training frames. Model train/eval and BN modes are managed separately.
        # No augmentation or GT database paste is performed in this adaptation.
        encoder = DatasetTemplate(config.DATA_CONFIG, config.CLASS_NAMES, training=False)
        model = build_network(config.MODEL, len(config.CLASS_NAMES), encoder)
        original_digest = sha256(args.checkpoint)
        disk = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = disk.get("model_state", disk.get("state_dict"))
        if state is None: raise ValueError("Checkpoint requires model_state or state_dict")
        transfer = transfer_state(model, state, find_all_spconv_keys(model))
        write_json(args.output/"checkpoint_transfer.json", transfer)
        del disk, state
        model.cuda().eval()
        status.update(status="running", gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      torch=torch.__version__, cuda=torch.version.cuda, mode=args.mode, seed=args.seed,
                      source_checkpoint_sha256=original_digest, source_config_sha256=sha256(args.config),
                      source_dataset_sha256=sha256(args.dataset/"dataset.json"), train_frames=len(train_rows), val_frames=len(val_rows),
                      preprocessing="official encoding; TRAIN-only quarter turns; evaluation/miner unchanged; zero time lag; no database paste",
                      velocity_loss_weights=[0., 0.], input_features=["x", "y", "z", "intensity", "zero_time_lag"])
        status.update(diagnostic_only=diagnostic, validation_independent=not diagnostic)
        status.update(validation_used_for_fit=False, validation_role=("reserved qualification only; no fitting or selection" if metadata.get("validation_purpose", "").startswith("qualification_only") else "development comparison only; qualification needs another frozen run"),
                      batch_size=args.batch_size, learning_rate=args.lr, weight_decay=1e-4, gradient_clip=10., max_steps=args.max_steps,
                      sampling="equal_physical_scenes_with_replacement" if args.balance_scenes else "uniform_shuffled_frames",
                      trainer_sha256=sha256(Path(__file__)), empty_batch_loss_weight=args.empty_loss_weight,
                      homogeneous_negative_batch_fraction=args.negative_batch_fraction,
                      backbone_learning_rate=args.backbone_lr, freeze_batchnorm_running_statistics=args.freeze_batchnorm,
                      training_source_eligibility=source_eligibility)
        shutil.copyfile(__file__, args.output/"trainer_source.py")
        write_json(args.output/"effective_config.json", config)
        write_json(args.output/"status.json", status)

        augmentation = QuarterTurnAugmentation(args.seed, args.output/'training_augmentation.jsonl')
        status.update(rotation_augmentation=augmentation.metadata(), parent_trainer_sha256='48b2546747bcb4a20977266da977fca9695c1786528d7bc5c653782ebb5cb6df',
                      augmentation_source_sha256=sha256(Path(__file__).resolve().parents[1]/'training/quarter_rotation.py'))

        def item(row, training=False):
            points = np.load(args.dataset/row["points_file"], allow_pickle=False).astype(np.float32)
            if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
                raise ValueError("Expected finite measured N×4 points")
            values = {"points": np.column_stack((points, np.zeros(len(points), np.float32))), "frame_id": row["frame_id"]}
            if training:
                values["gt_boxes"] = np.asarray([box["center"]+box["size"]+[box["yaw"], 0., 0.]
                                                 for box in row["gt_boxes"]], np.float32).reshape(-1, 9)
                values["gt_names"] = np.array(["car"]*len(row["gt_boxes"]))
                values["points"], values["gt_boxes"] = augmentation.apply(
                    values["points"], values["gt_boxes"], row["frame_id"], row["points_sha256"])
            prepared = encoder.prepare_data(values)
            if not training and "gt_boxes" in prepared: raise RuntimeError("GT leaked into evaluation forward input")
            return prepared

        output_score_floor = 0.
        def evaluate(label):
            model.eval()
            predictions, truth, latency = [], [], []
            with torch.inference_mode():
                for _ in range(2):
                    warmup = encoder.collate_batch([item(val_rows[0])]);load_data_to_gpu(warmup);model(warmup)
                torch.cuda.synchronize()
            with torch.inference_mode(), (args.output/(label+"_predictions.jsonl")).open("w") as handle:
                for row in val_rows:
                    batch = encoder.collate_batch([item(row)])
                    if "gt_boxes" in batch: raise RuntimeError("GT leaked into evaluation batch")
                    torch.cuda.synchronize();start = time.perf_counter();load_data_to_gpu(batch)
                    result, _ = model(batch);torch.cuda.synchronize();latency.append((time.perf_counter()-start)*1000)
                    prediction = {"scene_id": row["scene_id"], "frame_id": row["frame_id"], "boxes": []}
                    detected = result[0]
                    for box, score, cls in zip(detected["pred_boxes"].cpu().numpy(), detected["pred_scores"].cpu().numpy(), detected["pred_labels"].cpu().numpy()):
                        if int(cls) == 1 and score >= output_score_floor and np.linalg.norm(box[:2]) <= metadata["evaluation_radius_m"]:
                            prediction["boxes"].append({"center": box[:3].tolist(), "size": box[3:6].tolist(), "yaw": float(box[6]), "score": float(score)})
                    predictions.append(prediction);truth.append({"scene_id": row["scene_id"], "frame_id": row["frame_id"], "gt_boxes": row["gt_boxes"]})
                    handle.write(json.dumps(prediction, allow_nan=False)+"\n")
            metrics = comparison_metrics(predictions, truth)
            metrics.update(simulated_only=True, official_a2rl_metrics_executed=False, evaluation_radius_m=metadata["evaluation_radius_m"],
                           diagnostic_only=diagnostic, validation_independent=not diagnostic,
                           measured_point_visibility_threshold=metadata["min_measured_points_per_box"],
                           latency_median_ms=float(np.median(latency)), latency_p95_ms=float(np.percentile(latency, 95)),
                           warmup_passes=2, latency_scope="host-to-GPU plus model/NMS after2warmups; CPU voxelization excluded",
                           gt_removed_from_inference=True, effective_prediction_score_floor=output_score_floor)
            write_json(args.output/(label+"_metrics.json"), metrics)
            return metrics

        if args.baseline_cache:
            cached = json.loads((args.baseline_cache/"status.json").read_text())
            for key in ("source_checkpoint_sha256", "source_config_sha256", "source_dataset_sha256"):
                if cached.get(key) != status[key]: raise ValueError(f"Baseline cache identity mismatch:{key}")
            cached_rows = [json.loads(line) for line in (args.baseline_cache/"pretrained_predictions.jsonl").read_text().splitlines() if line.strip()]
            if {(r["scene_id"], r["frame_id"]) for r in cached_rows} != {(r["scene_id"], r["frame_id"]) for r in val_rows}:
                raise ValueError("Baseline cache frame set mismatch")
            for filename in ("pretrained_predictions.jsonl", "pretrained_metrics.json"):
                shutil.copyfile(args.baseline_cache/filename, args.output/filename)
            baseline = json.loads((args.output/"pretrained_metrics.json").read_text())
            status.update(baseline_cache_source=str(args.baseline_cache),
                          baseline_predictions_sha256=sha256(args.output/"pretrained_predictions.jsonl"))
        else:
            baseline = evaluate("pretrained")
        if args.evaluate_checkpoint:
            candidate = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=True)
            if candidate.get("source_checkpoint_sha256") != original_digest:
                raise ValueError("Frozen candidate does not declare this original source checkpoint")
            seen_scenes = set(candidate.get("training_scene_ids", []))
            if seen_scenes & {row["scene_id"] for row in val_rows}:
                raise ValueError("Frozen candidate was trained on a requested validation scenario")
            source_train_manifest = candidate.get("dataset_manifest_sha256", {}).get("train")
            if not seen_scenes and source_train_manifest != metadata["manifest_sha256"]["train"]:
                raise ValueError("Cannot verify candidate training lineage; retain its original training manifest in this dataset")
            candidate_state = candidate.get("model_state", candidate.get("state_dict"))
            if "race_detector_pipeline" in candidate:
                from race_lidar.pipeline_artifact import tensor_state_sha256
                pipeline = candidate["race_detector_pipeline"]
                if tensor_state_sha256(candidate_state) != pipeline["weights_state_sha256"]:
                    raise ValueError("Frozen pipeline tensor-state identity differs")
                output_score_floor = float(pipeline["effective_car_score_threshold"])
                if not math.isfinite(output_score_floor) or not 0 <= output_score_floor <= 1:
                    raise ValueError("Invalid frozen effective score threshold")
                status.update(effective_car_score_threshold=output_score_floor, detector_pipeline=pipeline)
            reloaded = transfer_state(model, candidate_state, find_all_spconv_keys(model))
            adapted = evaluate("adapted")
            if sha256(args.checkpoint) != original_digest: raise RuntimeError("Original pretrained checkpoint changed")
            status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), evaluation_only=True,
                          training_executed=False, training_steps=0, candidate_checkpoint=str(args.evaluate_checkpoint.resolve()),
                          candidate_checkpoint_sha256=sha256(args.evaluate_checkpoint), reload_verified_tensors=reloaded["loaded_tensors"],
                          baseline_metrics=baseline, adapted_metrics=adapted, live_services_modified=False,
                          accuracy_claim="frozen checkpoint comparison on held-out simulated runs; not A2RL/nuScenes benchmark AP")
            write_json(args.output/"status.json", status)
            print(json.dumps(status, indent=2), flush=True)
            return
        sample_weights = np.ones(len(train_rows), np.float64)
        if args.mine_checkpoint:
            mined_checkpoint = torch.load(args.mine_checkpoint, map_location="cpu", weights_only=True)
            miner_scenes = set(mined_checkpoint.get("training_scene_ids", []))
            subset_ok = (args.mine_allow_training_subset and miner_scenes and
                         miner_scenes <= {row["scene_id"] for row in train_rows} and
                         not miner_scenes & {row["scene_id"] for row in val_rows})
            if (mined_checkpoint.get("source_checkpoint_sha256") != original_digest or not
                    (mined_checkpoint.get("dataset_manifest_sha256", {}).get("train") == metadata["manifest_sha256"]["train"] or subset_ok)):
                raise ValueError("Mining checkpoint does not match the frozen original training lineage")
            transfer_state(model, mined_checkpoint["model_state"], find_all_spconv_keys(model));model.eval()
            del mined_checkpoint
            empty_indices = [i for i, row in enumerate(train_rows) if args.mine_all_training or not row["gt_boxes"]]
            mining = [];hard = 0
            with torch.inference_mode(), (args.output/"training_hard_negative_mining.jsonl").open("w") as handle:
                for count, index in enumerate(empty_indices, 1):
                    row = train_rows[index]
                    batch = encoder.collate_batch([item(row)]);assert "gt_boxes" not in batch;load_data_to_gpu(batch)
                    predictions, _ = model(batch);prediction = predictions[0]
                    detected = sorted([(box, float(score)) for box, score, cls in zip(prediction["pred_boxes"].cpu().numpy(),
                                      prediction["pred_scores"].cpu().numpy(), prediction["pred_labels"].cpu().numpy())
                                      if int(cls) == 1 and score >= args.mine_score and np.linalg.norm(box[:2]) <= metadata["evaluation_radius_m"]], key=lambda pair: -pair[1])
                    used, false_scores = set(), []
                    for box, score in detected:
                        choices = [(float(np.linalg.norm(box[:2]-np.asarray(gt["center"])[:2])), i)
                                   for i, gt in enumerate(row["gt_boxes"]) if i not in used]
                        error, selected = min(choices, default=(float("inf"), -1))
                        if error <= 2.: used.add(selected)
                        else: false_scores.append(score)
                    sample_weights[index] = 1.+4.*bool(false_scores)+min(4, len(false_scores))
                    hard += bool(false_scores)
                    record = {"training_index": index, "scene_id": row["scene_id"], "frame_id": row["frame_id"],
                              "points_sha256": row["points_sha256"], "false_car_scores": false_scores,
                              "sampling_weight": float(sample_weights[index])}
                    handle.write(json.dumps(record)+"\n")
                    if count % 500 == 0: handle.flush();print(json.dumps({"training_frames_mined": count, "hard_frames": hard}), flush=True)
            # Every new fit starts from the same original checkpoint, not from
            # the rejected miner. Validation and future groups never enter mining.
            restored = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            transfer_state(model, restored.get("model_state", restored.get("state_dict")), find_all_spconv_keys(model));del restored
            status["hard_negative_mining"] = {"checkpoint_sha256": sha256(args.mine_checkpoint),
                                              "training_frames": len(empty_indices), "all_training_frames": args.mine_all_training,
                                              "training_empty_frames": sum(not train_rows[i]["gt_boxes"] for i in empty_indices), "hard_frames": hard,
                                              "score_threshold": args.mine_score,
                                              "records_sha256": sha256(args.output/"training_hard_negative_mining.jsonl"),
                                              "original_weights_restored_before_fit": True, "validation_mined": False,
                                              "miner_training_subset_verified": bool(subset_ok)}
            write_json(args.output/"status.json", status)
        def training_mode():
            names = set_training_mode(model, args.mode)
            if args.freeze_batchnorm:
                for module in model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
            return names
        names = training_mode()
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters: raise RuntimeError("No trainable parameters")
        groups = [{"params": parameters, "lr": args.lr, "name": args.mode}]
        if args.backbone_lr is not None:
            groups = [{"params": [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("dense_head.")], "lr": args.backbone_lr, "name": "backbone"},
                      {"params": [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("dense_head.")], "lr": args.lr, "name": "head"}]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        write_json(args.output/"trainable_parameters.json", {"mode": args.mode, "names": names,
                   "optimizer_groups": [{"name": group["name"], "lr": group["lr"], "parameter_tensors": len(group["params"])} for group in groups]})
        rng = np.random.default_rng(args.seed);step = 0;gradient_report = None;started = time.perf_counter()
        empty_batches, positive_batches = 0, 0
        pools = {}
        if args.negative_batch_fraction:
            for empty in (False, True):
                allowed = [i for i, row in enumerate(train_rows) if bool(not row["gt_boxes"]) == empty]
                if not allowed: raise ValueError("Homogeneous sampling requires both positive and empty training frames")
                scenes = sorted({train_rows[i]["scene_id"] for i in allowed}) if args.balance_scenes else [None]
                pools[empty] = []
                for scene in scenes:
                    ids = np.array([i for i in allowed if scene is None or train_rows[i]["scene_id"] == scene])
                    weight = sample_weights[ids];probability = .7*weight/weight.sum()+.3/len(ids)
                    pools[empty].append((ids, probability))
        with (args.output/"losses.jsonl").open("w") as handle:
            for epoch in range(args.epochs):
                if args.negative_batch_fraction:
                    batches = []
                    for batch_index in range(math.ceil(len(train_rows)/args.batch_size)):
                        empty = math.floor((batch_index+1)*args.negative_batch_fraction) > math.floor(batch_index*args.negative_batch_fraction)
                        ids, probability = pools[empty][int(rng.integers(len(pools[empty])))]
                        batches.append(rng.choice(ids, size=args.batch_size, replace=True, p=probability))
                elif args.balance_scenes:
                    buckets = {scene: np.array([i for i, row in enumerate(train_rows) if row["scene_id"] == scene])
                               for scene in sorted({row["scene_id"] for row in train_rows})}
                    scenes = list(buckets);rng.shuffle(scenes)
                    indices = np.array([rng.choice(buckets[scenes[i % len(scenes)]]) for i in range(len(train_rows))])
                    batches = [indices[start:start+args.batch_size] for start in range(0, len(indices), args.batch_size)]
                else:
                    indices = rng.permutation(len(train_rows))
                    batches = [indices[start:start+args.batch_size] for start in range(0, len(indices), args.batch_size)]
                for indices in batches:
                    training_mode()
                    selected_rows = [train_rows[int(i)] for i in indices]
                    empty_batch = all(not row["gt_boxes"] for row in selected_rows)
                    empty_batches += int(empty_batch);positive_batches += int(not empty_batch)
                    multiplier = args.empty_loss_weight if empty_batch else 1.
                    batch = encoder.collate_batch([item(row, True) for row in selected_rows])
                    load_data_to_gpu(batch);optimizer.zero_grad(set_to_none=True)
                    output, _, _ = model(batch);unweighted_loss = output["loss"].mean();loss = unweighted_loss*multiplier
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    gradients = [p.grad for p in parameters if p.grad is not None]
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise RuntimeError("Missing or nonfinite gradients")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
                    if not math.isfinite(float(norm)): raise RuntimeError("Nonfinite gradient norm")
                    chosen = next((p for p in parameters if p.grad is not None and bool(p.grad.abs().max() > 0)), None)
                    if chosen is None: raise RuntimeError("Every gradient is zero")
                    backbone_choice = None
                    if gradient_report is None and args.mode == "full":
                        backbone = [(name, p) for name, p in model.named_parameters() if not name.startswith("dense_head.") and p.requires_grad]
                        nonzero_backbone = [(name, p) for name, p in backbone if p.grad is not None and bool(p.grad.abs().max() > 0)]
                        if not nonzero_backbone: raise RuntimeError("Full training requires actual nonzero backbone gradients")
                        backbone_name, backbone_choice = nonzero_backbone[0];backbone_before = backbone_choice.detach().clone()
                    before = chosen.detach().clone();optimizer.step()
                    if not bool((chosen.detach() != before).any()): raise RuntimeError("Optimizer did not update a nonzero-gradient parameter")
                    if hasattr(model, "update_global_step"): model.update_global_step()
                    step += 1
                    if gradient_report is None:
                        gradient_report = {"trainable_parameter_tensors": len(parameters), "gradient_tensors": len(gradients),
                                           "nonzero_gradient_tensors": sum(bool(g.abs().max() > 0) for g in gradients),
                                           "finite_gradients": True, "optimizer_changed_parameter": True}
                        if args.mode == "full":
                            changed = bool((backbone_choice.detach() != backbone_before).any())
                            if not changed: raise RuntimeError("Backbone optimizer did not change a nonzero-gradient tensor")
                            import hashlib
                            tensor_hash = lambda value: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                            gradient_report["backbone"] = {"trainable_parameter_tensors": len(backbone), "nonzero_gradient_tensors": len(nonzero_backbone),
                                "finite_gradients": True, "changed_parameter_name": backbone_name, "optimizer_changed_parameter": changed,
                                "before_sha256": tensor_hash(backbone_before), "after_first_step_sha256": tensor_hash(backbone_choice)}
                            write_json(args.output/"backbone_gradient_audit.json", gradient_report["backbone"])
                        write_json(args.output/"gradient_smoke.json", gradient_report)
                    handle.write(json.dumps({"step": step, "epoch": epoch+1, "loss": float(loss.detach()),
                                             "unweighted_loss": float(unweighted_loss.detach()), "loss_multiplier": multiplier,
                                             "empty_batch": empty_batch, "training_frame_ids": [row["frame_id"] for row in selected_rows],
                                             "training_augmentation_degrees": augmentation.last_angles(len(selected_rows)),
                                             "gradient_norm": float(norm)})+"\n");handle.flush()
                    status.update(training_executed=True, training_steps=step, rotation_augmentation=augmentation.metadata())
                    if step == 1 or step % 20 == 0:
                        print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
                        write_json(args.output/"status.json", status)
                    if args.max_steps and step >= args.max_steps: break
                checkpoint = args.output/(f"candidate_step_{step}.pth" if args.max_steps else f"candidate_epoch_{epoch+1}.pth")
                torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "epoch": epoch+1, "simulated_only": True, "source_checkpoint_sha256": original_digest,
                            "dataset_manifest_sha256": metadata["manifest_sha256"],
                            "training_scene_ids": sorted({row["scene_id"] for row in train_rows}),
                            "diagnostic_only": diagnostic, "training_steps": step,
                            "training_recipe": {"empty_batch_loss_weight": args.empty_loss_weight,
                                                "negative_batch_fraction": args.negative_batch_fraction,
                                                "mining": status.get("hard_negative_mining"), "lr": args.lr,
                                                "backbone_lr": args.backbone_lr, "freeze_batchnorm": args.freeze_batchnorm,
                                                "batch_size": args.batch_size, "seed": args.seed,
                                                "rotation_augmentation": augmentation.metadata()}}, checkpoint)
                if args.max_steps and step >= args.max_steps: break
        reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        reload_transfer = transfer_state(model, reloaded["model_state"], find_all_spconv_keys(model))
        write_json(args.output/"rotation_augmentation_summary.json", augmentation.metadata())
        if args.mode == "full":
            backbone_audit = gradient_report["backbone"]
            saved_name = backbone_audit["changed_parameter_name"]
            saved_hash = hashlib.sha256(reloaded["model_state"][saved_name].contiguous().numpy().tobytes()).hexdigest()
            if saved_hash == backbone_audit["before_sha256"]:
                raise RuntimeError("Saved final backbone tensor did not change from initialization")
            saved_proof = {"parameter_name": saved_name, "initial_sha256": backbone_audit["before_sha256"],
                           "saved_final_sha256": saved_hash, "changed_in_saved_checkpoint": True,
                           "first_step_nonzero_backbone_gradient_tensors": backbone_audit["nonzero_gradient_tensors"]}
            write_json(args.output/"saved_backbone_update.json", saved_proof)
            status["saved_backbone_update"] = saved_proof
        after = evaluate("adapted")
        if sha256(args.checkpoint) != original_digest: raise RuntimeError("Original pretrained checkpoint changed")
        status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), training_seconds=time.perf_counter()-started,
                      candidate_checkpoint=str(checkpoint.resolve()), candidate_checkpoint_sha256=sha256(checkpoint),
                      reload_verified_tensors=reload_transfer["loaded_tensors"], gradient_smoke=gradient_report,
                      baseline_metrics=baseline, adapted_metrics=after,
                      peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(), peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                      empty_training_batches=empty_batches, positive_training_batches=positive_batches,
                      live_services_modified=False,
                      accuracy_claim="same-run gradient smoke only; no independent validation" if diagnostic else "custom held-out simulated-run diagnostics only; not A2RL/nuScenes benchmark AP")
        write_json(args.output/"status.json", status)
        print(json.dumps(status, indent=2), flush=True)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output/"status.json", status)
        raise


if __name__ == "__main__":
    main()

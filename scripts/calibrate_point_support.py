#!/usr/bin/env python3
"""Calibrate sensor-only support using frozen TRAINING inputs and labels only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.batched_remote import load_batched_predictor
from race_lidar.point_support import config_sha256, filter_supported_boxes, validate_config
from qualify_sim_detector import matches
from scripts.cv_training_sources import validate_export_training_sources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def calibrate(args):
    source_eligibility = validate_export_training_sources(args.dataset, args.training_releases)
    import torch
    torch.set_num_threads(2)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new calibration output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    manifest = args.dataset/"train.jsonl"
    if sha(manifest) != protocol["training_manifest_sha256"] or sha(args.checkpoint) != protocol["parent_checkpoint_sha256"]:
        raise ValueError("Calibration requires the frozen original training manifest and selected parent weights")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    configs = [validate_config(config) for config in protocol["configurations"]]
    if len({config_sha256(config) for config in configs}) != len(configs):
        raise ValueError("Calibration configurations repeat")
    truth = {(row["scene_id"], row["frame_id"]): row for row in rows}
    if len(truth) != len(rows): raise ValueError("Training frame keys repeat")
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "training_only_calibration_not_independent_accuracy", "validation_inputs_opened": False,
              "training_frames": len(rows), "training_groups": sorted({row["scene_id"] for row in rows}),
              "training_manifest_sha256": sha(manifest), "parent_checkpoint_sha256": sha(args.checkpoint),
              "protocol_sha256": sha(args.protocol), "source_script_sha256": sha(Path(__file__)),
              "source_postprocess_sha256": sha(ROOT/"src/race_lidar/point_support.py"),
              "gpu_forward_contains_gt": False, "training_executed": False, "seed": args.seed}
    status["training_source_eligibility"] = source_eligibility
    write(args.output/"status.json", status)
    predictor = load_batched_predictor(args.pcdet_root, args.config, args.checkpoint)
    status["model"] = predictor.metadata
    write(args.output/"status.json", status)
    original, filtered = [], [[] for _ in configs]
    latency = [[] for _ in configs]
    score = protocol["operating_score_threshold"]
    with (args.output/"training_predictions.jsonl").open("w") as output:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start+args.batch_size]
            clouds = []
            for row in chunk:
                path = (args.dataset/row["points_file"]).resolve()
                if not path.is_relative_to(args.dataset.resolve()) or sha(path) != row["points_sha256"]:
                    raise ValueError("Frozen training cloud changed or escapes dataset root")
                clouds.append(np.load(path, allow_pickle=False))
            # Only measured point arrays enter the detector or support filter.
            predictions = predictor(clouds)
            for row, cloud, boxes in zip(chunk, clouds, predictions):
                prediction = {"scene_id": row["scene_id"], "frame_id": row["frame_id"], "boxes": boxes}
                original.append(prediction);output.write(json.dumps(prediction, allow_nan=False)+"\n")
                scored = [box for box in boxes if box["score"] >= score]
                for index, config in enumerate(configs):
                    tick = time.perf_counter();retained = filter_supported_boxes(cloud, scored, config)
                    latency[index].append((time.perf_counter()-tick)*1000)
                    filtered[index].append({**prediction, "boxes": retained})
            if start//args.batch_size % 100 == 0:
                output.flush();print(json.dumps({"training_frames_predicted": min(start+args.batch_size, len(rows))}), flush=True)
    distance = protocol["center_matching_threshold_m"]
    baseline_hits, baseline_fp = matches(original, truth, score, distance)
    categories = {name: set() for name in protocol["selection"]["maximum_loss_fraction_of_parent_matched_targets"]}
    for key, row in truth.items():
        for index, box in enumerate(row["gt_boxes"]):
            target = (*key, index);categories["all"].add(target)
            points = box["visible_points"]
            if 3 <= points <= 5: categories["sparse_3_5"].add(target)
            if 6 <= points <= 10: categories["sparse_6_10"].add(target)
            bearing = np.arctan2(box["center"][1], box["center"][0])
            if abs(np.cos(box["yaw"]-bearing)) <= .5: categories["side_view"].add(target)
    summaries = []
    for config, predictions, times in zip(configs, filtered, latency):
        hits, fp = matches(predictions, truth, score, distance)
        losses = {}
        for name, targets in categories.items():
            reference = set(baseline_hits)&targets
            lost = reference-set(hits)
            losses[name] = {"targets": len(targets), "parent_matched": len(reference), "filtered_matched": len(set(hits)&targets),
                            "lost_parent_matches": len(lost), "loss_fraction_of_parent_matches": len(lost)/len(reference) if reference else 0.}
        eligible = all(losses[name]["loss_fraction_of_parent_matches"] <= maximum+1e-12
                       for name, maximum in protocol["selection"]["maximum_loss_fraction_of_parent_matched_targets"].items())
        summaries.append({"config": config, "config_sha256": config_sha256(config), "eligible": eligible,
                          "true_positives": len(hits), "recall": len(hits)/len(categories["all"]), "false_positives": fp,
                          "false_positives_per_frame": fp/len(rows), "retention": losses,
                          "filter_latency_ms_median": float(np.median(times)), "filter_latency_ms_p95": float(np.percentile(times, 95))})
    eligible = [summary for summary in summaries if summary["eligible"]]
    eligible.sort(key=lambda row: (row["false_positives"], -row["true_positives"], row["config"]["above_ground_z_m"], row["config"]["minimum_xy_diameter_m"]))
    selected = eligible[0] if eligible else None
    report = {"scope": status["scope"], "protocol_sha256": sha(args.protocol), "parent_true_positives": len(baseline_hits),
              "parent_recall": len(baseline_hits)/len(categories["all"]), "parent_false_positives": baseline_fp,
              "frames": len(rows), "visible_targets": len(categories["all"]), "configurations": summaries,
              "selected": selected, "fresh_inputs_opened": False}
    write(args.output/"calibration_report.json", report)
    if selected:
        write(args.output/"selected_config.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(), "postprocess": selected["config"],
              "postprocess_sha256": selected["config_sha256"], "protocol_sha256": sha(args.protocol),
              "calibration_report_sha256": sha(args.output/"calibration_report.json"),
              "training_manifest_sha256": sha(manifest), "parent_checkpoint_sha256": sha(args.checkpoint), "fresh_inputs_opened": False})
    status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), selected=bool(selected),
                  training_predictions_sha256=sha(args.output/"training_predictions.jsonl"), report_sha256=sha(args.output/"calibration_report.json"))
    write(args.output/"status.json", status)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-releases", type=Path, help="Exact immutable authorized source exception manifest")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pcdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("Batch size must be positive")
    calibrate(args)


if __name__ == "__main__":
    main()

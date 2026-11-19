#!/usr/bin/env python3
"""Predeclared TRAIN-only score calibration; no gradients or reserved inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT));sys.path.insert(0, str(ROOT/"src"))
from race_lidar.batched_remote import load_batched_predictor
from scripts.cv_training_sources import validate_export_training_sources
from scripts.qualify_sim_detector import matches


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def select_threshold(official, learned, truth, protocol):
    baseline, baseline_fp = matches(official, truth, protocol["official_baseline_score_threshold"], protocol["matching_distance_m"])
    targets = sum(len(row["gt_boxes"]) for row in truth.values())
    if not targets: raise ValueError("Training calibration needs visible targets")
    summaries = []
    for threshold in protocol["raw_score_threshold_grid"]:
        hits, fp = matches(learned, truth, threshold, protocol["matching_distance_m"])
        eligible = len(hits) >= len(baseline) and fp/len(truth) <= protocol["objective"]["maximum_false_positives_per_training_frame"]+1e-12
        summaries.append({"effective_raw_score_threshold": threshold, "true_positives": len(hits), "false_positives": fp,
                          "recall": len(hits)/targets, "false_positives_per_frame": fp/len(truth), "eligible": eligible})
    eligible = sorted((row for row in summaries if row["eligible"]), key=lambda row: (-row["true_positives"], row["effective_raw_score_threshold"]))
    selected = eligible[0] if eligible else None
    return {"official_baseline_raw_threshold": protocol["official_baseline_score_threshold"],
            "official_baseline_true_positives": len(baseline), "official_baseline_recall": len(baseline)/targets,
            "official_baseline_false_positives": baseline_fp, "frames": len(truth), "visible_targets": targets,
            "grid": summaries, "selected": selected,
            "new_functional_candidate": bool(selected and selected["effective_raw_score_threshold"] != .35)}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-releases", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--learned-checkpoint", type=Path, required=True)
    parser.add_argument("--pcdet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists(): parser.error("Use a new calibration output directory")
    if args.batch_size < 1: parser.error("Batch size must be positive")
    protocol = json.loads(args.protocol.read_text())
    eligibility = validate_export_training_sources(args.dataset, args.training_releases)
    if (sha(args.dataset/"train.jsonl") != protocol["training_manifest_sha256"] or
        sha(args.official_checkpoint) != protocol["official_checkpoint_sha256"] or
        sha(args.learned_checkpoint) != protocol["learned_parent_checkpoint_sha256"]):
        raise ValueError("Calibration source identity differs from predeclared protocol")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    rows = [json.loads(line) for line in (args.dataset/"train.jsonl").read_text().splitlines() if line.strip()]
    truth = {(row["scene_id"], row["frame_id"]): row for row in rows}
    if len(truth) != len(rows) or len(rows) != 10233: raise ValueError("Calibration requires all10233unique original training frames")
    args.output.mkdir(parents=True)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(args.protocol),
              "training_source_eligibility": eligibility, "training_executed": False, "gradients_executed": False,
              "development_or_qualification_inputs_opened": False, "frames": len(rows), "training_groups": sorted({r["scene_id"] for r in rows}),
              "source_script_sha256": sha(Path(__file__)), "seed": args.seed}
    write(args.output/"status.json", status)
    official = load_batched_predictor(args.pcdet_root, args.config, args.official_checkpoint)
    learned = load_batched_predictor(args.pcdet_root, args.config, args.learned_checkpoint)
    status.update(official_model=official.metadata, learned_model=learned.metadata);write(args.output/"status.json", status)
    predictions = {"official": [], "learned": []}
    with (args.output/"official_training_predictions.jsonl").open("w") as old_file, (args.output/"learned_training_predictions.jsonl").open("w") as new_file:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start+args.batch_size];clouds = []
            for row in chunk:
                point = (args.dataset/row["points_file"]).resolve()
                if not point.is_relative_to(args.dataset.resolve()) or sha(point) != row["points_sha256"]: raise ValueError("Frozen training cloud changed")
                clouds.append(np.load(point, allow_pickle=False))
            # Only measured arrays enter either model. Labels are used later.
            for name, predictor, output in (("official", official, old_file), ("learned", learned, new_file)):
                boxes = predictor(clouds)
                if len(boxes) != len(chunk): raise RuntimeError("Detector batch differs from independent inputs")
                for row, detected in zip(chunk, boxes):
                    prediction = {"scene_id": row["scene_id"], "frame_id": row["frame_id"], "boxes": detected}
                    predictions[name].append(prediction);output.write(json.dumps(prediction, allow_nan=False)+"\n")
            if start//args.batch_size % 100 == 0:
                old_file.flush();new_file.flush();print(json.dumps({"training_frames_compared": min(start+args.batch_size, len(rows))}), flush=True)
    report = select_threshold(predictions["official"], predictions["learned"], truth, protocol)
    report.update(scope="TRAIN-only calibration; not independent accuracy or qualification", protocol_sha256=sha(args.protocol),
                  fresh_inputs_opened=False, training_manifest_sha256=sha(args.dataset/"train.jsonl"),
                  learned_checkpoint_sha256=sha(args.learned_checkpoint), official_checkpoint_sha256=sha(args.official_checkpoint))
    write(args.output/"calibration_report.json", report)
    if report["new_functional_candidate"]:
        write(args.output/"selected_config.json", {"frozen_utc": datetime.now(timezone.utc).isoformat(),
              "effective_car_score_threshold": report["selected"]["effective_raw_score_threshold"],
              "calibration_report_sha256": sha(args.output/"calibration_report.json"), "protocol_sha256": sha(args.protocol),
              "training_manifest_sha256": sha(args.dataset/"train.jsonl"), "weights_parent_checkpoint_sha256": sha(args.learned_checkpoint),
              "fresh_inputs_opened": False, "gradients_executed": False})
    decision = ("NEW_THRESHOLD_SELECTED_PENDING_FRESH_QUALIFICATION" if report["new_functional_candidate"] else
                "NO_EFFECTIVE_CHANGE_REJECTION_PRESERVED" if report["selected"] else "NO_ELIGIBLE_CONFIGURATION")
    status.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), decision=decision,
                  calibration_report_sha256=sha(args.output/"calibration_report.json"))
    write(args.output/"status.json", status);print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()

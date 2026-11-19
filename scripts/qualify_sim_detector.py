#!/usr/bin/env python3
"""Apply predeclared offline moving-run gates to frozen detector predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matches(prediction, ground_truth, score, distance):
    matched, false_positives = {}, 0
    for frame in prediction:
        frame_key = (frame["scene_id"], frame["frame_id"])
        boxes = ground_truth[frame_key]["gt_boxes"]
        used = set()
        for box in sorted(frame["boxes"], key=lambda value: -value["score"]):
            if box["score"] < score: continue
            available = [(float(np.linalg.norm(np.asarray(box["center"])[:2]-np.asarray(gt["center"])[:2])), i)
                         for i, gt in enumerate(boxes) if i not in used]
            error, index = min(available, default=(float("inf"), -1))
            if error <= distance:
                used.add(index);matched[(*frame_key, index)] = error
            else:
                false_positives += 1
    return matched, false_positives


def qualify(dataset, evaluation, definition, output=None):
    gate = json.loads(definition.read_text())
    status = json.loads((evaluation/"status.json").read_text())
    if status.get("training_executed") or not status.get("evaluation_only"):
        raise ValueError("Qualification must use an evaluation-only run with no fitting")
    if status.get("candidate_checkpoint_sha256") != gate["candidate_sha256"]:
        raise ValueError("Evaluation candidate differs from the predeclared frozen checkpoint")
    requested_groups = set(gate.get("qualification_runs") or [gate["qualification_run"]])
    truth = {(row["scene_id"], row["frame_id"]): row for row in rows(dataset/"val.jsonl") if row["scene_id"] in requested_groups}
    if {key[0] for key in truth} != requested_groups:
        raise ValueError("Held-out run differs from the predeclared qualification group")
    if status.get("source_dataset_sha256") != hashlib.sha256((dataset/"dataset.json").read_bytes()).hexdigest():
        raise ValueError("Qualification input manifest changed after evaluation")
    dataset_metadata = json.loads((dataset/"dataset.json").read_text())
    before, after = rows(evaluation/"pretrained_predictions.jsonl"), rows(evaluation/"adapted_predictions.jsonl")
    before = [row for row in before if row["scene_id"] in requested_groups]
    after = [row for row in after if row["scene_id"] in requested_groups]
    if set(truth) != {(row["scene_id"], row["frame_id"]) for row in before} or set(truth) != {(row["scene_id"], row["frame_id"]) for row in after}:
        raise ValueError("Qualification must evaluate every held-out frame in both models")
    old, old_fp = matches(before, truth, gate["operating_score_threshold"], gate["center_matching_threshold_m"])
    new, new_fp = matches(after, truth, gate["operating_score_threshold"], gate["center_matching_threshold_m"])
    target_count = sum(len(row["gt_boxes"]) for row in truth.values())
    common = set(old)&set(new)
    if not target_count or not common: raise ValueError("Need visible targets and common matched targets for qualification")
    old_center = float(np.mean([old[key] for key in common]));new_center = float(np.mean([new[key] for key in common]))
    far = {(*frame_key, i) for frame_key, row in truth.items() for i, box in enumerate(row["gt_boxes"])
           if gate["far_range_m"][0] <= np.linalg.norm(np.asarray(box["center"])[:2]) <= gate["far_range_m"][1]}
    old_far = len(set(old)&far)/len(far) if far else None
    new_far = len(set(new)&far)/len(far) if far else None
    adequate = len(far) >= gate["far_minimum_targets_for_additional_gate"]
    n = len(truth);fp_limit = gate["false_positives_per_frame"];center_limit = gate["center_error_common_matched_targets"]
    checks = {"recall_nonregression": len(new) >= len(old),
              "false_positive_increase_bound": new_fp/n <= old_fp/n+fp_limit["maximum_increase"]+1e-12,
              "false_positive_absolute_bound": new_fp/n <= fp_limit["absolute_maximum"]+1e-12,
              "common_center_absolute_bound": new_center <= center_limit["adapted_mean_max_m"]+1e-12,
              "common_center_degradation_bound": new_center-old_center <= center_limit["maximum_mean_degradation_m"]+1e-12,
              "far_recall_nonregression_when_adequate": not adequate or new_far >= old_far}
    excluded_inputs = [row for row in dataset_metadata.get("excluded_duplicate_validation_frames", []) if row["scene_id"] in requested_groups]
    group_counts = dataset_metadata.get("input_frame_counts_by_group", {}).get("val", {})
    raw_count = sum(group_counts[group] for group in requested_groups) if all(group in group_counts for group in requested_groups) else len(truth)+len(excluded_inputs)
    report = {"passed": all(checks.values()), "eligibility": "physical trial only; not formal promotion", "checks": checks,
              "original_effective_raw_score_threshold": gate["operating_score_threshold"],
              "candidate_effective_raw_score_threshold": max(gate["operating_score_threshold"], status.get("effective_car_score_threshold", gate["operating_score_threshold"])),
              "score_scope": "Original uses gate threshold; candidate also enforces its frozen pipeline's declared raw score floor.",
              "qualification_groups": sorted(requested_groups),
              "definition": gate, "definition_sha256": hashlib.sha256(definition.read_bytes()).hexdigest(),
              "input_scope": dataset_metadata.get("qualification_subset_scope", dataset_metadata["split_unit"]),
              "raw_qualification_frames": raw_count,
              "excluded_exact_training_clouds": len(excluded_inputs),
              "frames": n, "visible_targets": target_count, "common_matched_targets": len(common),
              "pretrained": {"recall": len(old)/target_count, "false_positives": old_fp, "false_positives_per_frame": old_fp/n,
                             "common_matched_center_mean_m": old_center},
              "adapted": {"recall": len(new)/target_count, "false_positives": new_fp, "false_positives_per_frame": new_fp/n,
                          "common_matched_center_mean_m": new_center},
              "far_targets": len(far), "far_count_adequate": adequate, "pretrained_far_recall": old_far,
              "adapted_far_recall": new_far, "far_scope": "frame-level targets may be temporally correlated; not independent physical scenes",
              "new_only_matches": len(set(new)-set(old)), "pretrained_only_matches": len(set(old)-set(new)),
              "fit_on_qualification": False, "metric_scope": "custom2mXY-center diagnostic on visible simulated cars; not A2RL/nuScenes benchmark AP"}
    (output or evaluation/"qualification_gate.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Separate named report for each predeclared qualification group")
    args = parser.parse_args()
    print(json.dumps(qualify(args.dataset, args.evaluation, args.definition, args.output), indent=2))


if __name__ == "__main__":
    main()

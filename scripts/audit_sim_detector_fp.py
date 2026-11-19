#!/usr/bin/env python3
"""Diagnostic-only FP attribution using full separate Supervisor ground truth.

This does not relabel data, rewrite a scored gate, or authorize deployment.
Range/sparsity/ego labels below explain scored false positives; they remain FPs
under the frozen visible-target scoring protocol.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from build_sim_detector_dataset import count_inside, world_pose_boxes


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a["center"])[:2]-np.asarray(b["center"])[:2]))


def scored_false_positives(predictions, truth, threshold):
    result = []
    for frame in predictions:
        key = (frame["scene_id"], frame["frame_id"])
        targets = truth[key]["gt_boxes"];used = set()
        for box in sorted(frame["boxes"], key=lambda row: -row["score"]):
            if box["score"] < threshold: continue
            choices = [(distance(box, gt), i) for i, gt in enumerate(targets) if i not in used]
            error, index = min(choices, default=(float("inf"), -1))
            if error <= 2.:
                used.add(index)
            else:
                any_target = min([(distance(box, gt), i) for i, gt in enumerate(targets)], default=(float("inf"), -1))
                result.append({"scene_id": key[0], "frame_id": key[1], "box": box,
                               "duplicate_visible_target": any_target[0] <= 2. and any_target[1] in used})
    return result


def audit(dataset, evaluation, output, threshold=.35):
    truth = {(r["scene_id"], r["frame_id"]): r for r in rows(dataset/"val.jsonl")}
    all_poses, configs = {}, {}
    results = {}
    for name in ("pretrained", "adapted"):
        false_positives = scored_false_positives(rows(evaluation/(name+"_predictions.jsonl")), truth, threshold)
        for item in false_positives:
            row = truth[(item["scene_id"], item["frame_id"])];run = Path(row["source_run"])
            if run not in all_poses:
                all_poses[run] = {p["timestamp_ns"]: p for p in rows(run/"gt_poses.jsonl")}
                configs[run] = json.loads((run/"config.json").read_text())
            full = world_pose_boxes(all_poses[run][row["timestamp_ns"]]["cars"], row["car_id"], configs[run],
                                    configs[run].get("ego_ground_z_m", 0.), include_ego=True)
            points = np.load(dataset/row["points_file"], allow_pickle=False)
            for box in full:
                box["visible_points"] = count_inside(points, box)
                box["range_m"] = float(np.linalg.norm(np.asarray(box["center"])[:2]))
            nearest = min(full, key=lambda box: distance(item["box"], box))
            error = distance(item["box"], nearest)
            if item["duplicate_visible_target"]:
                category = "duplicate_visible_car"
            elif error <= 2.:
                if nearest["car_id"] == row["car_id"]: category = "ego_self_return"
                elif nearest["range_m"] > 50.: category = "range_boundary_excluded_car"
                elif nearest["visible_points"] == 0: category = "zero_return_excluded_car"
                elif nearest["visible_points"] < 3: category = "sparse_return_excluded_car"
                else: category = "other_known_car_requires_review"
            elif error <= 4.:
                category = "mislocalized_known_car_2to4m"
            else:
                category = "background_no_GT_car_within4m"
            length, width, height = item["box"]["size"]
            item.update(category=category, car_id=row["car_id"], timestamp_ns=row["timestamp_ns"],
                        closest_GT=nearest, closest_GT_center_error_m=error,
                        runtime_size_filter_would_reject=not(3. <= length <= 7.5 and 1.25 <= width <= 2.9 and .55 <= height <= 2.5),
                        runtime_self_radius_would_reject=bool(np.linalg.norm(np.asarray(item["box"]["center"])[:2]) < 3.))
        results[name] = false_positives
    shared_old = set()
    for item in sorted(results["adapted"], key=lambda row: -row["box"]["score"]):
        choices = [(distance(item["box"], other["box"]), i) for i, other in enumerate(results["pretrained"])
                   if i not in shared_old and item["scene_id"] == other["scene_id"] and item["frame_id"] == other["frame_id"]]
        error, index = min(choices, default=(float("inf"), -1))
        item["shared_with_pretrained_FP"] = error <= 2.
        if error <= 2.: shared_old.add(index)
    report = {"scope": "diagnostic only; scored gate and visible-target definitions unchanged", "score_threshold": threshold,
              "all_GT_source": "separate exact-timestamp Supervisor poses, including ego and visibility/range-excluded cars",
              "heuristics": "2m center association;2–4m tagged mislocalization;farther tagged background. No inference that these tags change driving safety.",
              "counts": {name: dict(Counter(item["category"] for item in values)) for name, values in results.items()},
              "pretrained_FP": len(results["pretrained"]), "adapted_FP": len(results["adapted"]),
              "shared_FP": len(shared_old), "new_candidate_FP": len(results["adapted"])-len(shared_old),
              "removed_pretrained_FP": len(results["pretrained"])-len(shared_old),
              "new_candidate_FP_types": dict(Counter(item["category"] for item in results["adapted"] if not item["shared_with_pretrained_FP"])),
              "runtime_filter_tags": {name: {"size": sum(item["runtime_size_filter_would_reject"] for item in values),
                                               "self_radius": sum(bool(item["runtime_self_radius_would_reject"]) for item in values)} for name, values in results.items()},
              "instances": results}
    if (evaluation/"qualification_gate.json").exists():
        gate = json.loads((evaluation/"qualification_gate.json").read_text())
        if report["pretrained_FP"] != gate["pretrained"]["false_positives"] or report["adapted_FP"] != gate["adapted"]["false_positives"]:
            raise ValueError("Audit FP counts differ from the immutable scored gate")
    output.write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key != "instances"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.evaluation, args.output), indent=2))


if __name__ == "__main__":
    main()

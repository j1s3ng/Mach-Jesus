"""Custom center-distance diagnostics; NOT KITTI/RACE official AP or racing scores.

Matching is score-ranked and one-to-one within each (scene_id, frame_id).
AP uses 101-point interpolated precision and XY center distance <= threshold.
Heading error is the absolute directed yaw difference wrapped to [-pi, pi],
computed only on those center-matched pairs; a 180-degree flip remains an error.
Class labels, 3-D IoU, difficulty levels, track continuity and driving safety
are deliberately not assessed. Missing-annotation frames are excluded.
"""

from __future__ import annotations

import numpy as np

from .io import validate_box


def evaluate_frames(predictions: list[dict], ground_truth: list[dict], distance_threshold: float = 2.0) -> dict:
    """Accept frame dictionaries with `boxes` predictions and `gt_boxes` labels."""
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive")
    truth, prediction_map = {}, {}
    for frame in ground_truth:
        key = (frame["scene_id"], frame["frame_id"])
        if key in truth:
            raise ValueError(f"Duplicate ground-truth frame: {key}")
        if "gt_boxes" in frame:
            truth[key] = frame["gt_boxes"]
            for box in truth[key]:
                validate_box(box)
    for frame in predictions:
        key = (frame["scene_id"], frame["frame_id"])
        if key in prediction_map:
            raise ValueError(f"Duplicate prediction frame: {key}")
        prediction_map[key] = frame.get("boxes", [])
        for box in prediction_map[key]:
            validate_box(box)
    ranked = sorted([(float(box.get("score", 1)), key, i, box)
                     for key, boxes in prediction_map.items() if key in truth
                     for i, box in enumerate(boxes)], key=lambda item: (-item[0], item[1], item[2]))
    matched = {key: set() for key in truth}
    hits, errors, heading_errors = [], [], []
    for _, key, _, box in ranked:
        available = [(float(np.linalg.norm(np.asarray(box["center"][:2]) - np.asarray(gt["center"][:2]))), i)
                     for i, gt in enumerate(truth[key]) if i not in matched[key]]
        distance, index = min(available, default=(float("inf"), -1))
        hit = distance <= distance_threshold
        hits.append(int(hit))
        if hit:
            matched[key].add(index)
            errors.append(distance)
            delta = float(box["yaw"]) - float(truth[key][index]["yaw"])
            heading_errors.append(float(abs(np.arctan2(np.sin(delta), np.cos(delta)))))
    total_gt = sum(map(len, truth.values()))
    true_positive = int(sum(hits))
    false_positive = len(hits) - true_positive
    ap = None
    if total_gt:
        cumulative = np.cumsum(hits)
        recall = cumulative / total_gt
        precision = cumulative / np.arange(1, len(hits) + 1)
        ap = float(np.mean([precision[recall >= target].max(initial=0)
                            for target in np.linspace(0, 1, 101)]))
    return {
        "metric_type": "custom_xy_center_distance_non_official",
        "warning": "Not official KITTI/RACE AP, 3D IoU AP, or a measure of racing performance",
        "distance_threshold_m": float(distance_threshold), "annotated_frames": len(truth),
        "excluded_unannotated_prediction_frames": len(set(prediction_map) - set(truth)),
        "ground_truth_objects": total_gt, "predicted_objects": len(hits),
        "true_positives": true_positive, "false_positives": false_positive,
        "false_negatives": total_gt - true_positive, "ap_101": ap,
        "precision": true_positive / len(hits) if hits else None,
        "recall": true_positive / total_gt if total_gt else None,
        "matched_center_error_mean_m": float(np.mean(errors)) if errors else None,
        "matched_heading_error_mean_rad": float(np.mean(heading_errors)) if heading_errors else None,
        "matched_heading_error_p95_rad": float(np.percentile(heading_errors, 95)) if heading_errors else None,
        "heading_error_scope": "directed yaw wrapped to [-pi,pi], on custom center-matched pairs only",
    }

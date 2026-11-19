"""Sensor-only support filtering for upright predicted boxes in ego metres.

The leveled sensor datum has road ground near z=0. Ground truth, route geometry,
opponent identity, and tracking state are deliberately absent from this API.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def validate_config(config):
    expected = {"name", "version", "minimum_points", "above_ground_z_m", "box_margin_m", "minimum_xy_diameter_m"}
    if set(config) != expected or config["name"] != "measured_above_ground_box_support" or config["version"] != 1:
        raise ValueError("Unknown point-support configuration")
    if isinstance(config["minimum_points"], bool) or not isinstance(config["minimum_points"], int) or config["minimum_points"] < 1:
        raise ValueError("minimum_points must be a positive integer")
    if not np.isfinite(config["above_ground_z_m"]) or config["above_ground_z_m"] < 0:
        raise ValueError("above_ground_z_m must be finite and nonnegative")
    if not np.isfinite(config["box_margin_m"]) or config["box_margin_m"] < 0:
        raise ValueError("box_margin_m must be finite and nonnegative")
    if not np.isfinite(config["minimum_xy_diameter_m"]) or config["minimum_xy_diameter_m"] < 0:
        raise ValueError("minimum_xy_diameter_m must be finite and nonnegative")
    return dict(config)


def config_sha256(config):
    return hashlib.sha256(json.dumps(validate_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def xy_diameter_at_least(xy, threshold):
    """Exact, rotation-invariant pair-distance predicate with bounded memory."""
    if threshold <= 0:
        return True
    if len(xy) < 2:
        return False
    spans = np.ptp(xy, axis=0)
    # Both shortcuts imply the same Euclidean predicate for every rotation.
    if np.max(spans) >= threshold:
        return True
    if np.linalg.norm(spans) < threshold:
        return False
    squared = threshold*threshold
    for start in range(0, len(xy), 32):
        delta = xy[start:start+32, None, :]-xy[None, :, :]
        if np.any(np.sum(delta*delta, axis=2) >= squared):
            return True
    return False


def support_evidence(points, boxes, config):
    """Count returns and test optional XY diameter inside each predicted box."""
    config = validate_config(config)
    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[1] != 4 or cloud.dtype.kind != "f" or not np.isfinite(cloud).all():
        raise ValueError("Expected finite floating-point N×4 measured points")
    above = cloud[cloud[:, 2] > config["above_ground_z_m"], :3]
    evidence = []
    for box in boxes:
        center, size = np.asarray(box["center"], dtype=float), np.asarray(box["size"], dtype=float)
        yaw = float(box["yaw"])
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0) or not np.isfinite(yaw):
            raise ValueError("Predicted boxes require finite center/yaw and positive length,width,height")
        delta = above-center
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.column_stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1], delta[:, 2]))
        inside = np.all(np.abs(local) <= size/2+config["box_margin_m"], axis=1)
        evidence.append({"points": int(np.count_nonzero(inside)),
                         "xy_diameter_sufficient": xy_diameter_at_least(above[inside, :2], config["minimum_xy_diameter_m"])})
    return evidence


def support_counts(points, boxes, config):
    return [row["points"] for row in support_evidence(points, boxes, config)]


def filter_supported_boxes(points, boxes, config):
    """Preserve each original box and score iff measured support is sufficient."""
    config = validate_config(config)
    evidence = support_evidence(points, boxes, config)
    return [box for box, row in zip(boxes, evidence) if row["points"] >= config["minimum_points"] and row["xy_diameter_sufficient"]]


def wrap_predictor(predict_batch, config, *, artifact_sha256, parent_checkpoint_sha256, state_sha256):
    """Filter each independent car using only that car's corresponding cloud."""
    config = validate_config(config)
    def supported(point_clouds):
        predictions = predict_batch(point_clouds)
        if len(predictions) != len(point_clouds):
            raise RuntimeError("Detector returned a different number of examples")
        return [filter_supported_boxes(points, boxes, config) for points, boxes in zip(point_clouds, predictions)]
    supported.metadata = {**getattr(predict_batch, "metadata", {}),
                          "checkpoint_sha256": artifact_sha256,
                          "pipeline_artifact_sha256": artifact_sha256,
                          "weights_parent_checkpoint_sha256": parent_checkpoint_sha256,
                          "weights_state_sha256": state_sha256,
                          "postprocess": config, "postprocess_sha256": config_sha256(config)}
    if hasattr(predict_batch, "telemetry"):
        supported.telemetry = predict_batch.telemetry
    return supported

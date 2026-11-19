"""Exact XY quarter turns for supervised TRAIN points and numeric box labels.

This module imports no CUDA. Augmentation has its own Generator and never
touches global NumPy state, the frame sampler, evaluation or the TRAIN miner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ANGLES = (0, 90, -90, 180)
RNG_SALT = 0x524F5441


def _xy(array, degrees):
    result = array.copy()
    if degrees == 90:
        result[..., 0], result[..., 1] = -array[..., 1], array[..., 0]
    elif degrees == -90:
        result[..., 0], result[..., 1] = array[..., 1], -array[..., 0]
    elif degrees == 180:
        result[..., :2] = -array[..., :2]
    return result


def rotate_tensors(points, boxes, degrees):
    """Rotate XYZ[time/intensity] and [x,y,z,l,w,h,yaw,vx,vy] labels."""
    if degrees not in ANGLES:
        raise ValueError("Only the four predeclared exact quarter turns are allowed")
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] not in (4, 5) or not np.isfinite(points).all():
        raise ValueError("Expected finite float32 N×4/5 training points")
    if boxes.dtype != np.float32 or boxes.ndim != 2 or boxes.shape[1] != 9 or not np.isfinite(boxes).all():
        raise ValueError("Expected finite float32 M×9 supervised box labels")
    if len(boxes) and np.any(boxes[:, 3:6] <= 0):
        raise ValueError("Box sizes must be positive")
    rotated_points = _xy(points, degrees)
    rotated_boxes = _xy(boxes, degrees)
    if degrees:
        yaw = boxes[:, 6].astype(np.float64) + np.deg2rad(degrees)
        rotated_boxes[:, 6] = np.arctan2(np.sin(yaw), np.cos(yaw)).astype(np.float32)
        rotated_boxes[:, 7:9] = _xy(boxes[:, 7:9], degrees)
    return rotated_points, rotated_boxes


def corners(boxes):
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    local = signs[None] * boxes[:, None, 3:6].astype(np.float64) / 2.
    cosine = np.cos(boxes[:, 6].astype(np.float64))[:, None]
    sine = np.sin(boxes[:, 6].astype(np.float64))[:, None]
    result = local.copy()
    result[:, :, 0] = cosine * local[:, :, 0] - sine * local[:, :, 1]
    result[:, :, 1] = sine * local[:, :, 0] + cosine * local[:, :, 1]
    return result + boxes[:, None, :3].astype(np.float64)


def tensor_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def alignment_audit(points, boxes, rotated_points, rotated_boxes, degrees):
    inverse = {0: 0, 90: -90, -90: 90, 180: 180}[degrees]
    restored_points, restored_boxes = rotate_tensors(rotated_points, rotated_boxes, inverse)
    expected_corners = _xy(corners(boxes), degrees)
    error = float(np.max(np.abs(corners(rotated_boxes) - expected_corners))) if len(boxes) else 0.
    yaw_delta = restored_boxes[:, 6].astype(np.float64) - boxes[:, 6].astype(np.float64)
    yaw_error = float(np.max(np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))))) if len(boxes) else 0.
    result = {"point_inverse_exact": bool(np.array_equal(restored_points, points)),
        "z_intensity_time_exact": bool(np.array_equal(rotated_points[:, 2:], points[:, 2:])),
        "box_xyz_size_inverse_exact": bool(np.array_equal(restored_boxes[:, :6], boxes[:, :6])),
        "velocity_inverse_exact": bool(np.array_equal(restored_boxes[:, 7:9], boxes[:, 7:9])),
        "maximum_box_corner_alignment_error_m": error, "maximum_inverse_yaw_wrap_error_rad": yaw_error}
    if not all(result[key] for key in ("point_inverse_exact", "z_intensity_time_exact", "box_xyz_size_inverse_exact", "velocity_inverse_exact")) or error > 1e-5 or yaw_error > 1e-6:
        raise ValueError("Actual TRAIN point/label rigid alignment audit failed")
    return result


class QuarterTurnAugmentation:
    def __init__(self, seed, log_path):
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, RNG_SALT]))
        self.log_path = Path(log_path)
        if self.log_path.exists():
            raise ValueError("Preserve prior actual augmentation logs")
        self.angles = []
        self.counts = {angle: 0 for angle in ANGLES}
        self.max_corner_error = self.max_yaw_error = 0.

    def apply(self, points, boxes, frame_id, source_points_sha256):
        degrees = ANGLES[int(self.rng.integers(4))]
        rotated_points, rotated_boxes = rotate_tensors(points, boxes, degrees)
        audit = alignment_audit(points, boxes, rotated_points, rotated_boxes, degrees)
        self.angles.append(degrees)
        self.counts[degrees] += 1
        self.max_corner_error = max(self.max_corner_error, audit["maximum_box_corner_alignment_error_m"])
        self.max_yaw_error = max(self.max_yaw_error, audit["maximum_inverse_yaw_wrap_error_rad"])
        record = {"augmentation_index": len(self.angles), "frame_id": frame_id, "degrees": degrees,
            "source_points_sha256": source_points_sha256, "point_count": len(points), "gt_box_count": len(boxes),
            "input_points_tensor_sha256": tensor_sha(points), "augmented_points_tensor_sha256": tensor_sha(rotated_points),
            "input_box_tensor_sha256": tensor_sha(boxes), "augmented_box_tensor_sha256": tensor_sha(rotated_boxes),
            **audit}
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        return rotated_points, rotated_boxes

    def last_angles(self, count):
        return self.angles[-count:]

    def metadata(self):
        return {"version": "train_only_exact_quarter_turn_v1", "angles_degrees": list(ANGLES),
            "seed": self.seed, "independent_rng_salt": RNG_SALT, "draws": len(self.angles),
            "actual_angle_counts": {str(k): v for k, v in self.counts.items()},
            "maximum_box_corner_alignment_error_m": self.max_corner_error,
            "maximum_inverse_yaw_wrap_error_rad": self.max_yaw_error,
            "gt_velocity_xy_rotated": True, "evaluation_or_mining_augmented": False,
            "record_sha256": hashlib.sha256(self.log_path.read_bytes()).hexdigest() if self.log_path.exists() else None}
